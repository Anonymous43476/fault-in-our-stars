#define BOOST_TEST_MODULE protocol

#include <secureagg/encryption.h>
#include <secureagg/obliv_share_sample.h>

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace secureagg;
namespace bdata = boost::unit_test::data;

Report run_obliv_share_sample_protocol(Server& server, Client& client,
                                       PtxSpace input, PtxSpace pt) {
  auto client_message = client.initial_message(input, pt);
  auto server_message = server.read_message(client_message);
  auto report = client.read_message(server_message);

  return report;
}

BOOST_AUTO_TEST_SUITE(obliv_share_sample)

BOOST_AUTO_TEST_CASE(execute_with_one_mdss_rep) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  const unsigned int mdss_reps = 1;
  const unsigned int t_priv = 5;

  PtxSpace input{11};
  PtxSpace pt{2};

  auto server = Server(std::move(sk), mdss_reps);
  auto client = Client(std::move(pk), mdss_reps, t_priv);
  auto report = run_obliv_share_sample_protocol(server, client, input, pt);

  BOOST_TEST(pt == report.pt);
  BOOST_TEST(report.share.size() == mdss_reps);
}

BOOST_AUTO_TEST_CASE(execute_with_ten_mdss_rep) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  const unsigned int mdss_reps = 10;
  const unsigned int t_priv = 5;

  PtxSpace input{11};
  PtxSpace pt{2};

  auto server = Server(std::move(sk), mdss_reps);
  auto client = Client(std::move(pk), mdss_reps, t_priv);
  auto report = run_obliv_share_sample_protocol(server, client, input, pt);

  BOOST_TEST(pt == report.pt);
  BOOST_TEST(report.share.size() == mdss_reps);
}

BOOST_AUTO_TEST_CASE(reports_are_deterministic) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  const unsigned int mdss_reps = 10;
  const unsigned int t_priv = 5;

  PtxSpace input{11};
  PtxSpace id{2};

  auto server = Server(std::move(sk), mdss_reps);
  auto client = Client(std::move(pk), mdss_reps, t_priv);
  auto report1 = run_obliv_share_sample_protocol(server, client, input, id);
  auto report2 = run_obliv_share_sample_protocol(server, client, input, id);

  BOOST_TEST(report1.share == report2.share);
}

BOOST_AUTO_TEST_CASE(reports_lie_on_the_same_polynomial) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  const unsigned int mdss_reps = 2;
  const unsigned int t_priv = 1;

  PtxSpace input{11};
  PtxSpace id1{2};
  PtxSpace id2{3};

  auto server = Server(std::move(sk), mdss_reps);
  auto client = Client(std::move(pk), mdss_reps, t_priv);
  auto report1 = run_obliv_share_sample_protocol(server, client, input, id1);
  auto report2 = run_obliv_share_sample_protocol(server, client, input, id2);

  for (auto i = 0; i < mdss_reps; i++) {
    auto recon_inp =
        (report1.share[i] - (report2.share[i] - report1.share[i]) * id1) %
        ptx_mod;
    if (recon_inp < 0) {
      recon_inp += ptx_mod;
    }

    BOOST_TEST(recon_inp == input);
  }
}

BOOST_AUTO_TEST_SUITE_END()
