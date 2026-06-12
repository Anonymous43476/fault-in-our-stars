#include <NTL/ZZ.h>
#include <NTL/BasicThreadPool.h>
#include <benchmark/benchmark.h>
#include <secureagg/encryption.h>

#include <vector>

using namespace secureagg;

static void BM_Decryption(benchmark::State& state) {
  double space_param = static_cast<double>(state.range(0)) / 100;
  uint32_t num_bits_ptx_mod = state.range(1);
  uint32_t num_bits_ctx_mod = state.range(2);

  NTL::SetNumThreads(1);

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod, space_param);
  auto pk = sk.getPublicKey();
  auto ctx = pk.encrypt(pk.plaintextModulus() - 1);

  for (auto _ : state) {
    auto val = sk.decrypt(ctx);
    benchmark::DoNotOptimize(val);
  }
}

BENCHMARK(BM_Decryption)
    ->ArgsProduct({{50, 75, 100},
                   benchmark::CreateDenseRange(10, 30, /*step=*/10),
                   {2048, 3072}});

BENCHMARK_MAIN();
