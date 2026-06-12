#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <benchmark/benchmark.h>
#include <secureagg/encryption.h>
#include <secureagg/obliv_share_sample.h>

using namespace secureagg;

static void BM_ClientMessage(benchmark::State& state) {
  uint32_t ptx_mod = state.range(0);
  unsigned int t_priv = state.range(1);
  unsigned int mdss_reps = state.range(2);
  uint32_t num_bits_ctx_mod = state.range(3);
  unsigned int threads = state.range(4);

  NTL::SetNumThreads(threads);

  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod, 0.75);
  auto pk = sk.getPublicKey();

  auto client = Client(std::move(pk), mdss_reps, t_priv);

  PtxSpace input = NTL::RandomBnd(ptx_mod);
  PtxSpace pt = ptx_mod - 1;  // Worst case input

  for (auto _ : state) {
    auto client_message = client.initial_message(input, pt);
    benchmark::DoNotOptimize(client_message);
  }

  auto client_message = client.initial_message(input, pt);
  state.counters["MessageSize"] = static_cast<double>(client_message.size());
}

BENCHMARK(BM_ClientMessage)
    ->ArgsProduct({{100003}, {349}, {138}, {2048, 3072}, {1, 4, 8}});

static void BM_ServerMessage(benchmark::State& state) {
  uint32_t ptx_mod = state.range(0);
  unsigned int t_priv = state.range(1);
  unsigned int mdss_reps = state.range(2);
  uint32_t num_bits_ctx_mod = state.range(3);
  unsigned int threads = state.range(4);

  NTL::SetNumThreads(threads);

  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod, 0.75);
  auto pk = sk.getPublicKey();

  auto client = Client(std::move(pk), mdss_reps, t_priv);
  auto server = Server(std::move(sk), mdss_reps);

  PtxSpace input = NTL::RandomBnd(ptx_mod);
  PtxSpace pt = ptx_mod - 1;
  auto client_message = client.initial_message(input, pt);

  for (auto _ : state) {
    auto server_message = server.read_message(client_message);
    benchmark::DoNotOptimize(server_message);
  }

  auto server_message = server.read_message(client_message);
  state.counters["MessageSize"] = static_cast<double>(server_message.size());
}

BENCHMARK(BM_ServerMessage)
    ->ArgsProduct({{100003}, {349}, {138}, {2048, 3072}, {1, 4, 8}});

BENCHMARK_MAIN();
