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
    {
      printf 'runtime:\n'
      printf '  workload: "Training"\n'
      printf '  optimizer: "Adam"\n'
      printf '  framework: "%s"\n' "${framework}"
      printf '\n'
      printf 'optimizer:\n'
      printf '  learning_rate: %s\n' "${LEARNING_RATE}"
      printf '  beta_1: %s\n' "${BETA1}"
      printf '  beta_2: %s\n' "${BETA2}"
      printf '  epsilon: %s\n' "${EPSILON}"
      printf '  dim_m: 10024\n'
      printf '  dim_k: 10024\n'
      printf '  dim_n: 256\n'
      printf '  batch_size: %s\n' "${batch_size}"
      printf '\n'
      printf 'workload:\n'
      printf '  dataset_dir: "%s"\n' "${DATASET_DIR}"
      printf '  max_samples: %s\n' "${MAX_SAMPLES}"
      printf '  num_epochs: %s\n' "${NUM_EPOCHS}"
    } > "${CONFIG_FILE}"

    echo "Running framework=${framework} batch_size=${batch_size} max_samples=${MAX_SAMPLES} num_epochs=${NUM_EPOCHS}"
    (cd "${ROOT_DIR}" && ./build/app)
  done
done

echo
echo "Validation runs finished."
echo "Results written to: ${VALIDATION_LOG}"
