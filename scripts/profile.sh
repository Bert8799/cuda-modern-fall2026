#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: %s <executable> [argv...]\n' "$0" >&2
    printf 'Example: %s build/sgemm/sgemm 1\n' "$0" >&2
    exit 2
fi

target_file=$1
shift
app_args=("$@")

target_dir=$(dirname -- "$target_file")
target_name=$(basename -- "$target_file")
arg_tag=
if ((${#app_args[@]} > 0)); then
    printf -v arg_tag '%s_' "${app_args[@]}"
    arg_tag=${arg_tag%_}
    arg_tag=${arg_tag//[^[:alnum:]_.-]/_}
    arg_tag=${arg_tag:0:48}
    arg_tag="-$arg_tag"
fi
report_dir="$target_dir/profiling"

if [[ ! -f "$target_file" || ! -x "$target_file" ]]; then
    printf 'Error: executable not found or not executable: %s\n' "$target_file" >&2
    exit 1
fi

for tool in nsys ncu; do
    command -v "$tool" >/dev/null 2>&1 ||
        { printf 'Error: required tool not found: %s\n' "$tool" >&2; exit 1; }
done

mkdir -p "$report_dir"
system_report="$report_dir/${target_name}${arg_tag}-system"
compute_report="$report_dir/${target_name}${arg_tag}-compute"

# Nsight Systems: run the application normally. Its GPU timestamps reflect the
# real execution, so this is the authoritative kernel time.
if ! nsys profile --force-overwrite true -o "$system_report" \
    "$target_file" "${app_args[@]}" >/dev/null 2>&1; then
    printf 'Error: Nsight Systems profiling failed\n' >&2
    exit 1
fi

# Nsight Compute: replay the kernel in isolation to collect hardware counters.
# The FP32 SASS instruction counters let us count FADD + FMUL + 2 * FFMA FLOPs.
if ! ncu --launch-count 1 --force-overwrite \
    --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    -o "$compute_report" "$target_file" "${app_args[@]}" >/dev/null 2>&1; then
    printf 'Error: Nsight Compute profiling failed\n' >&2
    exit 1
fi

# Export the profiled kernel name, duration, DRAM bytes and FP32 instruction
# counts from the report. NCU's raw CSV puts metric names in the header and
# records units and launch values on the following rows.
read -r ncu_time_ns read_bytes write_bytes fadd_count fmul_count ffma_count kernel_name < <(
    ncu --import "$compute_report.ncu-rep" --csv --page raw --print-units base 2>/dev/null |
        awk -F'","' '
            function unquote(value) {
                sub(/^"/, "", value)
                sub(/"$/, "", value)
                return value
            }
            function clean_number(value) {
                value = unquote(value)
                gsub(/,/, "", value)
                gsub(/[[:space:]]/, "", value)
                if (value == "n/a" || value == "N/A") return ""
                return value
            }
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    metric = unquote($i)
                    if (metric == "gpu__time_duration.sum") time_column = i
                    else if (metric == "dram__bytes_op_read.sum") read_column = i
                    else if (metric == "dram__bytes_op_write.sum") write_column = i
                    else if (metric == "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") fadd_column = i
                    else if (metric == "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") fmul_column = i
                    else if (metric == "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") ffma_column = i
                }
                next
            }
            NF > 1 && unquote($1) ~ /^[0-9]+$/ {
                if (time_column > 0) t = clean_number($(time_column))
                if (read_column > 0) r = clean_number($(read_column))
                if (write_column > 0) w = clean_number($(write_column))
                if (fadd_column > 0) fadd = clean_number($(fadd_column))
                if (fmul_column > 0) fmul = clean_number($(fmul_column))
                if (ffma_column > 0) ffma = clean_number($(ffma_column))
                name = unquote($5)
                print t, r, w, fadd, fmul, ffma, name
                exit
            }'
) || true

# Kernel time in ns from the Nsight Systems summary (first matching TSV row).
nsys_time_ns=$(nsys stats --force-export=true --format tsv --report cuda_gpu_kern_sum \
    --output - "$system_report.nsys-rep" 2>/dev/null |
    awk -F'\t' -v kernel="$kernel_name" '
        $1 ~ /^[0-9.]+$/ {
            if (first_time == "") first_time = $2
            if (kernel != "" && index($9, kernel)) {
                print $2
                matched = 1
                exit
            }
        }
        END {
            if (!matched && first_time != "") print first_time
        }') || true

missing_metrics=()
[[ -n "$nsys_time_ns" ]] || missing_metrics+=("Nsight Systems kernel time")
[[ -n "$ncu_time_ns" ]] || missing_metrics+=("Nsight Compute kernel time")
[[ -n "$read_bytes" ]] || missing_metrics+=("DRAM read bytes")
[[ -n "$write_bytes" ]] || missing_metrics+=("DRAM write bytes")
[[ -n "$fadd_count" ]] || missing_metrics+=("FP32 FADD count")
[[ -n "$fmul_count" ]] || missing_metrics+=("FP32 FMUL count")
[[ -n "$ffma_count" ]] || missing_metrics+=("FP32 FFMA count")
if ((${#missing_metrics[@]} > 0)); then
    printf 'Error: failed to read required profiler data: %s\n' "${missing_metrics[*]}" >&2
    exit 1
fi

awk -v nsys_time_ns="$nsys_time_ns" -v ncu_time_ns="$ncu_time_ns" \
    -v read_bytes="$read_bytes" -v write_bytes="$write_bytes" \
    -v fadd_count="$fadd_count" -v fmul_count="$fmul_count" -v ffma_count="$ffma_count" '
    BEGIN {
        bytes = read_bytes + write_bytes
        flops = fadd_count + fmul_count + 2 * ffma_count
        printf "Nsight Systems: kernel time %.3f ms, bandwidth %.3f GB/s, TFLOPS %.3f\n",
               nsys_time_ns / 1000000, bytes / nsys_time_ns, flops / nsys_time_ns / 1000
        printf "Nsight Compute: kernel time %.3f ms, bandwidth %.3f GB/s, TFLOPS %.3f\n",
               ncu_time_ns / 1000000, bytes / ncu_time_ns, flops / ncu_time_ns / 1000
    }'
