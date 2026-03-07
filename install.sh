check() {
    local command=("$@")

    if "${command[@]}"; then
        echo notify user OK >&2
    else
        echo notify user FAIL >&2
        exit 1
    fi
}

# Before execution, make sure you have a GPU attached to your host.
is_hpc=0
  if [[ -n "${SLURM_JOB_ID:-}" || -n "${LMOD_SYSTEM_NAME:-}" || -d /sw || -d /etc/modulefiles ]]; then
    is_hpc=1
  fi

if (( is_hpc )); then
	# clean suspicious local prepend
	export PATH="/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:$PATH"

	# clean environment
	module purge

	# load modules
	check module load gcc/13.2.0
	check module load cuda/12.6.2

	CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
	CUDA_ARCH="${CUDA_ARCH:-75}"

	rm -rf build
	cmake -S . -B build \
   		-DCMAKE_BUILD_TYPE=Release \
   		-DBUILD_TESTS=OFF \
   		-DBUILD_BENCHMARKS=ON \
   		-DLIBTORCH_ROOT="$PWD/lib/libtorch" \
   		-DCUDAToolkit_ROOT="$CUDA_ROOT" \
   		-DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
   		-DCMAKE_CXX_FLAGS="-I$HOME/.local/include" \
   		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
 
	# pick a filesystem with quota/space (examples)
	mkdir -p "$HOME/tmp" "$HOME/.nv/tmp"

	export TMPDIR="$HOME/tmp"
	export TEMP="$TMPDIR"
	export TMP="$TMPDIR"
	export CUDA_CACHE_PATH="$HOME/.nv/tmp"

	# optional: reduce parallel pressure
	cmake --build build -j 2
fi
