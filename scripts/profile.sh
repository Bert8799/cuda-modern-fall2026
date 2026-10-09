#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <build-target-directory>\n' "$0" >&2
    printf 'Example: %s build/reduce\n' "$0" >&2
    exit 2
fi

target_dir=$1
target_name=$(basename -- "$target_dir")
executable="$target_dir/$target_name"
report_dir="$target_dir/profiling"

if [[ ! -x "$executable" ]]; then
    printf 'Error: executable not found or not executable: %s\n' "$executable" >&2
    exit 1
fi

for tool in nsys ncu; do
    command -v "$tool" >/dev/null 2>&1 ||
        { printf 'Error: required tool not found: %s\n' "$tool" >&2; exit 1; }
done

mkdir -p "$report_dir"
system_report="$report_dir/${target_name}-system"
compute_report="$report_dir/${target_name}-compute"

# Nsight Systems: run the application normally. Its GPU timestamps reflect the
# real execution, so this is the authoritative kernel time.
if ! nsys profile --force-overwrite true -o "$system_report" "$executable" >/dev/null 2>&1; then
    printf 'Error: Nsight Systems profiling failed\n' >&2
    exit 1
fi

# Nsight Compute: replay the kernel in isolation to collect hardware counters.
# It locks clocks to their base values by default, so its kernel time is a
# controlled reference; the DRAM byte counters are reused for both bandwidths.
if ! ncu --set basic --launch-count 1 --force-overwrite \
    --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum \
    -o "$compute_report" "$executable" >/dev/null 2>&1; then
    printf 'Error: Nsight Compute profiling failed\n' >&2
    exit 1
fi

# Export the profiled kernel name, duration (ns) and DRAM bytes from the report.
read -r ncu_time_ns read_bytes write_bytes kernel_name < <(
    ncu --import "$compute_report.ncu-rep" --csv --print-units base 2>/dev/null |
        awk -F'","' '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i == "gpu__time_duration.sum") { t = $(i + 2); name = $5 }
                    else if ($i == "dram__bytes_op_read.sum") r = $(i + 2)
                    else if ($i == "dram__bytes_op_write.sum") w = $(i + 2)
                }
            }
            END {
                gsub(/[",]/, "", t); gsub(/[",]/, "", r); gsub(/[",]/, "", w)
                if (t == "n/a") t = ""; if (r == "n/a") r = ""; if (w == "n/a") w = ""
                print t, r, w, name
            }'
) || true

# Kernel time in ns from the Nsight Systems summary (first matching TSV row).
nsys_time_ns=$(nsys stats --force-export=true --format tsv --report cuda_gpu_kern_sum \
    --output - "$system_report.nsys-rep" 2>/dev/null |
    awk -F'\t' -v kernel="$kernel_name" '$1 ~ /^[0-9.]+$/ && index($9, kernel) { print $2; exit }') || true

if [[ -z "$nsys_time_ns" || -z "$ncu_time_ns" || -z "$read_bytes" || -z "$write_bytes" ]]; then
    printf 'Error: failed to read the required metrics\n' >&2
    exit 1
fi

bytes=$((read_bytes + write_bytes))
awk -v nsys_time_ns="$nsys_time_ns" -v ncu_time_ns="$ncu_time_ns" -v bytes="$bytes" '
    function thousands(value, rest, result) {
        rest = sprintf("%.0f", value)
        result = ""
        while (length(rest) > 3) {
            result = "," substr(rest, length(rest) - 2) result
            rest = substr(rest, 1, length(rest) - 3)
        }
        return rest result
    }
    BEGIN {
        printf "Nsight Systems: kernel time %sns, bandwidth %.3fGB/s\n",
               thousands(nsys_time_ns), bytes / nsys_time_ns
        printf "Nsight Compute: kernel time %sns, bandwidth %.3fGB/s\n",
               thousands(ncu_time_ns), bytes / ncu_time_ns
    }'
