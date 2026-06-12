# Private Heavy-Hitters Protocol

## Requirements

- [GMP](https://gmplib.org/)
- [NTL](https://libntl.org/) (>= 11.5.1)
- [Boost](https://www.boost.org/) (>= 1.83.0)
- [Benchmark](https://github.com/google/benchmark) (>= 1.9.0)

All of these should be available via Homebrew using the `gmp`, `ntl`, `boost`, and `google-benchmark` formulas.

## Usage

```sh
# Commands are assumed to be run from root directory of the repo.

# Create build directory.
mkdir build && cd build

# Create Makefiles.
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release ..

# Compile all targets.
# To only compile tests or benchmarks run 'make -j tests' and 'make -j benches'
# respectively.
make -j

# Run unit tests.
ctest
# Alternatively, run specific unit tests binaries using optional command line
# arguments. For example, to run the unit test suite for encryption with 
# complete log information, execute
./tests/encryption_test -l all

# Run benchmarks.
# Simply execute the benchmark file to run benchmarks. A number of command line
# options are available to change time units, save files, or change format.
# For example, to run the decryption benchmark, execute
./benches/decryption
```
