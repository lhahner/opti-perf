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

	# ensure vendored deps are present
	check git submodule update --init --recursive

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
   		-DCUDAToolkit_ROOT="$CUDA_ROOT" \
   		-DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
		-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-11 \
   		-DCMAKE_CXX_FLAGS="-I$HOME/.local/include" \
   		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
		-DENABLE_YAML=ON \
		-DUSE_SYSTEM_YAMLCPP=OFF \
		-DYAMLCPP_ROOT="$PWD/lib/yaml-cpp"
 
	# pick a filesystem with quota/space (examples)
	mkdir -p "$HOME/tmp" "$HOME/.nv/tmp"

	export TMPDIR="$HOME/tmp"
	export TEMP="$TMPDIR"
	export TMP="$TMPDIR"
	export CUDA_CACHE_PATH="$HOME/.nv/tmp"

	# optional: reduce parallel pressure
	cmake --build build -j 2

else
	# Non-HPC systems (no module environment)
	export PATH="/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:$PATH"
	CUDA_ARCH="${CUDA_ARCH:-52}" # Change for different GPU architectures (e.g., 80 for Ampere, 90 for Hopper)

	# ensure vendored deps are present
	check git submodule update --init --recursive

	if command -v nvcc >/dev/null 2>&1; then
		CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
	else
		CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
	fi
	if command -v gcc-11 >/dev/null 2>&1; then
		HOST_GCC="$(command -v gcc-11)"
		HOST_GXX="$(command -v g++-11)"
	else
		echo "gcc-11/g++-11 not found. Install GCC 11 or set HOST_GCC/HOST_GXX." >&2
		exit 1
	fi

	rm -rf build
	cmake -S . -B build \
   		-DCMAKE_BUILD_TYPE=Release \
   		-DBUILD_TESTS=OFF \
   		-DBUILD_BENCHMARKS=ON \
   		-DCUDAToolkit_ROOT="$CUDA_ROOT" \
   		-DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
		-DCMAKE_C_COMPILER="$HOST_GCC" \
		-DCMAKE_CXX_COMPILER="$HOST_GXX" \
		-DCMAKE_CUDA_HOST_COMPILER="$HOST_GCC" \
   		-DCMAKE_CXX_FLAGS="-I$HOME/.local/include" \
   		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
		-DENABLE_YAML=ON \
		-DUSE_SYSTEM_YAMLCPP=OFF \
		-DYAMLCPP_ROOT="$PWD/lib/yaml-cpp"

	# pick a filesystem with quota/space (examples)
	mkdir -p "$HOME/tmp" "$HOME/.nv/tmp"

	export TMPDIR="$HOME/tmp"
	export TEMP="$TMPDIR"
	export TMP="$TMPDIR"
	export CUDA_CACHE_PATH="$HOME/.nv/tmp"

	# optional: reduce parallel pressure
	cmake --build build -j 2
fi
