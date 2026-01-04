#!/bin/bash
set -e

LOG="/home/jjschirle/PIDC/output/hESC_unperturbed_edges.log"

echo "=== Starting FastPIDC job at $(date) ===" | tee -a "$LOG"

# Set the threads you want
export JULIA_NUM_THREADS=16

# Start memory sampling in background ------------------------------
MEMLOG="/home/jjschirle/PIDC/output/hESC_unperturbed_edges_memtrace.log"

(
    echo "=== Memory trace started at $(date) ==="
    while true; do
        date
        ps -u $USER -o pid,ppid,cmd,%mem,%cpu,rss,vsz --sort=-%mem | head -n 10
        echo "--------------------------------------------------------"
        sleep 5
    done
) >> "$MEMLOG" &
MEM_PID=$!

echo "Memory monitor PID = $MEM_PID" | tee -a "$LOG"

# Run the actual job with /usr/bin/time -v -------------------------
(
    /usr/bin/time -v julia --project=/home/jjschirle/PIDC/FastPIDC.jl --color=yes \
        "/home/jjschirle/PIDC/FastPIDC.jl/CLI_fastpidc.jl" \
        --infile "/home/jjschirle/PIDC/data/hESC_unperturbed_pidc_input.txt" \
        --outfile "/home/jjschirle/PIDC/output/hESC_unperturbed_edges.tsv" \
        --delim "space" \
        --discretizer "uniform_width" \
        --estimator "maximum_likelihood" \
        --n-bins 10 \
        --base 2 \
        --triplet-block-k 30 \
        --neighbor-mode "target" \
        --triplet-backend "threads" \
        --verbose true
) 2>&1 | tee -a "$LOG"
# (
#     /usr/bin/time -v julia --project=/home/jjschirle/PIDC/FastPIDC.jl -p 8 --color=yes \
#         "/home/jjschirle/PIDC/FastPIDC.jl/CLI_fastpidc.jl" \
#         --infile "/home/jjschirle/PIDC/data/hESC_unperturbed_pidc_input.txt" \
#         --outfile "/home/jjschirle/PIDC/output/hESC_unperturbed_edges.tsv" \
#         --delim "space" \
#         --discretizer "uniform_width" \
#         --estimator "maximum_likelihood" \
#         --n-bins 10 \
#         --base 2 \
#         --triplet-block-k 30 \
#         --neighbor-mode "target" \
#         --triplet-backend "threads" \
#         --verbose true
# ) 2>&1 | tee -a "$LOG"

# Kill memory sampler when done ------------------------------------
echo "Stopping memory monitor…" | tee -a "$LOG"
kill $MEM_PID 2>/dev/null || true

echo "=== FastPIDC job finished at $(date) ===" | tee -a "$LOG"
