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

		# load lib
		check mkdir -p lib
		if [[ ! -d "lib/yaml-cpp/.git" ]]; then
			check git clone https://github.com/jbeder/yaml-cpp.git lib/yaml-cpp
		else
			echo "yaml-cpp repository already exists at lib/yaml-cpp; skipping clone." >&2
		fi
		check mkdir -p lib/yaml-cpp/build
		if ! cmake -S lib/yaml-cpp -B lib/yaml-cpp/build \
			-DCMAKE_BUILD_TYPE=Release \
			-DYAML_BUILD_SHARED_LIBS=OFF \
			-DYAML_CPP_BUILD_TESTS=OFF \
			-DYAML_CPP_BUILD_TOOLS=OFF \
			-DCMAKE_INSTALL_PREFIX="$PWD/lib/yaml-cpp/install"; then
			echo "Failed to configure yaml-cpp in lib/yaml-cpp/build." >&2
			exit 1
		fi
		if ! cmake --build lib/yaml-cpp/build -j 2; then
			echo "Failed to build yaml-cpp from lib/yaml-cpp/build." >&2
			exit 1
		fi
		if ! cmake --install lib/yaml-cpp/build; then
			echo "Failed to install yaml-cpp into $PWD/lib/yaml-cpp/install." >&2
			exit 1
		fi
		if [[ ! -d "lib/benchmark/.git" ]]; then
			check git clone https://github.com/google/benchmark.git lib/benchmark
		else
			echo "google-benchmark repository already exists at lib/benchmark; skipping clone." >&2
		fi
		check mkdir -p lib/benchmark/build
		if ! cmake -S lib/benchmark -B lib/benchmark/build \
			-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
			-DCMAKE_BUILD_TYPE=Release \
			-DBENCHMARK_ENABLE_TESTING=OFF \
			-DBENCHMARK_ENABLE_GTEST_TESTS=OFF \
			-DBENCHMARK_ENABLE_INSTALL=ON \
			-DCMAKE_INSTALL_PREFIX="$PWD/lib/benchmark/install"; then
			echo "Failed to configure google-benchmark in lib/benchmark/build." >&2
			exit 1
		fi
		if ! cmake --build lib/benchmark/build --config Release -j 2; then
			echo "Failed to build google-benchmark from lib/benchmark/build." >&2
			exit 1
		fi
		if ! cmake --install lib/benchmark/build; then
			echo "Failed to install google-benchmark into $PWD/lib/benchmark/install." >&2
			exit 1
		fi


	CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
	CUDA_ARCH="${CUDA_ARCH:-75}"
	if [[ -n "${HOST_GCC:-}" && -n "${HOST_GXX:-}" ]]; then
		:
	elif command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
		HOST_GCC="$(command -v gcc)"
		HOST_GXX="$(command -v g++)"
	else
		echo "No CUDA host compiler found. Set HOST_GCC/HOST_GXX or load a GCC module." >&2
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
