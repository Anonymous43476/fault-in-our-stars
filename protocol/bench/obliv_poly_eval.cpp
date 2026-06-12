#include <NTL/ZZ.h>
#include <NTL/BasicThreadPool.h>
#include <benchmark/benchmark.h>
#include <secureagg/encryption.h>

#include <vector>

using namespace secureagg;

static void BM_OblivPolyEval(benchmark::State& state) {
  size_t degree = state.range(0);
  uint32_t num_bits_ptx_mod = state.range(1);
  uint32_t num_bits_ctx_mod = state.range(2);

  NTL::SetNumThreads(1);

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto pk = BESecretKey(ptx_mod, num_bits_ctx_mod).getPublicKey();

  std::vector<NTL::ZZ> coeffx_ctx(degree + 1);
  PtxSpace pt{0};

  for (auto& i : coeffx_ctx) {
    i = pk.sampleCiphertext();
  }
  NTL::RandomBnd(pt, pk.plaintextModulus());

  for (auto _ : state) {
    auto res = pk.poly_eval(coeffx_ctx, pt);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_OblivPolyEval)
    ->ArgsProduct({{349},        // degree = t_priv from BM_ClientMessage
                   {17},         // ~log2(100003), matching ptx_mod bit-width
                   {2048, 3072}});

// BENCHMARK(BM_OblivPolyEval)
//     ->ArgsProduct({benchmark::CreateRange(128, 8192, /*multi=*/2),
//                    benchmark::CreateDenseRange(10, 30, /*step=*/5),
//                    {2048, 3072}});

BENCHMARK_MAIN();
