#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/config.yaml"
VALIDATION_LOG="${ROOT_DIR}/data/logs/validation-benchmark-logs.csv"

FRAMEWORKS=("CUDA" "OpenCL")
BATCH_SIZES=(32 64 128 256)
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
EPSILON="${EPSILON:-1e-08}"
DATASET_DIR="${DATASET_DIR:-data/mnist}"

if [[ ! -x "${ROOT_DIR}/build/app" ]]; then
  echo "Missing executable: ${ROOT_DIR}/build/app"
  echo "Build first with: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${DATASET_DIR}/train-images-idx3-ubyte" ]]; then
  echo "Missing MNIST training data in ${ROOT_DIR}/${DATASET_DIR}"
  echo "Download with: python3 scripts/load_download_mnist.py -d ${DATASET_DIR}"
  exit 1
fi

mkdir -p "$(dirname "${VALIDATION_LOG}")"
rm -f "${VALIDATION_LOG}"

for framework in "${FRAMEWORKS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    cat > "${CONFIG_FILE}" <<EOF
runtime:
  workload: "Training"
  optimizer: "Adam"
  framework: "${framework}"

optimizer:
  learning_rate: ${LEARNING_RATE}
  beta_1: ${BETA1}
  beta_2: ${BETA2}
  epsilon: ${EPSILON}
  dim_m: 10024
  dim_k: 10024
  dim_n: 256
  batch_size: ${batch_size}

workload:
  dataset_dir: "${DATASET_DIR}"
  max_samples: ${MAX_SAMPLES}
  num_epochs: ${NUM_EPOCHS}
EOF

    echo "Running framework=${framework} batch_size=${batch_size} max_samples=${MAX_SAMPLES} num_epochs=${NUM_EPOCHS}"
    (cd "${ROOT_DIR}" && ./build/app)
  done
done

echo
echo "Validation runs finished."
echo "Results written to: ${VALIDATION_LOG}"
