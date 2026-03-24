#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/config.yaml"
BENCHMARK_LOG="${ROOT_DIR}/data/logs/benchmarks-logs.csv"

FRAMEWORKS=("CUDA" "OpenCL")
DIM_MS=(2024 3024 4024 5024 6024 7024 8024 9024 10024)
DIM_KS=(2024 3024 4024 5024 6024 7024 8024 9024 10024)
DIM_N="${DIM_N:-256}"
BATCH_SIZE="${BATCH_SIZE:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
EPSILON="${EPSILON:-1e-08}"

if [[ ! -x "${ROOT_DIR}/build/app" ]]; then
  echo "Missing executable: ${ROOT_DIR}/build/app"
  echo "Build first with: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

mkdir -p "$(dirname "${BENCHMARK_LOG}")"
rm -f "${BENCHMARK_LOG}"

for framework in "${FRAMEWORKS[@]}"; do
  for i in "${!DIM_MS[@]}"; do
    dim_m="${DIM_MS[$i]}"
    dim_k="${DIM_KS[$i]}"

    cat > "${CONFIG_FILE}" <<EOF
runtime:
  workload: "GEMM"
  optimizer: "Adam"
  framework: "${framework}"

optimizer:
  learning_rate: ${LEARNING_RATE}
  beta_1: ${BETA1}
  beta_2: ${BETA2}
  epsilon: ${EPSILON}
  dim_m: ${dim_m}
  dim_k: ${dim_k}
  dim_n: ${DIM_N}
  batch_size: ${BATCH_SIZE}
EOF

    echo "Running framework=${framework} workload=${dim_m}x${dim_k}x${DIM_N} batch_size=${BATCH_SIZE}"
    (cd "${ROOT_DIR}" && ./build/app)
  done
done

echo
echo "GEMM benchmark runs finished."
echo "Results written to: ${BENCHMARK_LOG}"
