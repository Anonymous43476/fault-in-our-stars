#define BOOST_TEST_MODULE encryption

#include <NTL/ZZ.h>
#include <secureagg/encryption.h>

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/included/unit_test.hpp>
#include <vector>

using namespace secureagg;
namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(dlog)
BOOST_AUTO_TEST_CASE(only_lookup) {
  const NTL::ZZ base{4};
  const NTL::ZZ mod{227};
  PtxSpace order{113};
  const NTL::ZZ exp{58};
  const NTL::ZZ val = NTL::PowerMod(base, exp, mod);

  auto dlog = BruteForceDL(base, NTL::ZZ(mod), order, 1);
  auto dl_val = dlog(val);

  BOOST_TEST(dl_val == exp);
}

BOOST_AUTO_TEST_CASE(no_lookup) {
  const NTL::ZZ base{4};
  const NTL::ZZ mod{227};
  PtxSpace order{113};
  const NTL::ZZ exp{58};
  const NTL::ZZ val = NTL::PowerMod(base, exp, mod);

  auto dlog = BruteForceDL(base, NTL::ZZ(mod), order, 0);
  auto dl_val = dlog(val);

  BOOST_TEST(dl_val == exp);
}

BOOST_AUTO_TEST_CASE(sqrt_lookup) {
  const NTL::ZZ base{4};
  const NTL::ZZ mod{227};
  PtxSpace order{113};
  const NTL::ZZ exp{58};
  const NTL::ZZ val = NTL::PowerMod(base, exp, mod);

  auto dlog = BruteForceDL(base, NTL::ZZ(mod), order, 0.5);
  auto dl_val = dlog(val);

  BOOST_TEST(dl_val == exp);
}

BOOST_AUTO_TEST_CASE(one_fifth_lookup) {
  const NTL::ZZ base{4};
  const NTL::ZZ mod{227};
  PtxSpace order{113};
  const NTL::ZZ exp{58};
  const NTL::ZZ val = NTL::PowerMod(base, exp, mod);

  auto dlog = BruteForceDL(base, NTL::ZZ(mod), order, 0.2);
  auto dl_val = dlog(val);

  BOOST_TEST(dl_val == exp);
}

BOOST_DATA_TEST_CASE(dlog_computation, bdata::xrange(112), exp_int) {
  const NTL::ZZ base{4};
  const NTL::ZZ mod{227};
  PtxSpace order{113};
  const NTL::ZZ exp{exp_int};
  const NTL::ZZ val = NTL::PowerMod(base, exp, mod);

  auto dlog = BruteForceDL(base, NTL::ZZ(mod), order, 0.5);
  auto dl_val = dlog(val);

  BOOST_TEST(dl_val == exp);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(bescheme)

BOOST_AUTO_TEST_CASE(gen_secret_key_small) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);

  auto ctx_mod = sk.ciphertextModulus();
  auto factor1 = sk.ctxModulusFactor1();
  auto factor2 = sk.ctxModulusFactor2();

  BOOST_TEST(NTL::NumBits(ptx_mod) == num_bits_ptx_mod);
  BOOST_TEST(NTL::NumBits(ctx_mod) >= num_bits_ctx_mod);

  BOOST_TEST_MESSAGE("Plaintext bit length: " << NTL::NumBits(ptx_mod));
  BOOST_TEST_MESSAGE("Ciphertext bit length: " << NTL::NumBits(ctx_mod));

  BOOST_TEST(((factor1 - 1) % ptx_mod) != 0);
  BOOST_TEST(((factor2 - 1) % ptx_mod) == 0);
  BOOST_TEST(((factor2 - 1) % (ptx_mod * ptx_mod)) != 0);
}

BOOST_AUTO_TEST_CASE(gen_secret_key_large) {
  const uint32_t num_bits_ptx_mod = 20;
  const uint32_t num_bits_ctx_mod = 2048;

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);

  auto ctx_mod = sk.ciphertextModulus();
  auto factor1 = sk.ctxModulusFactor1();
  auto factor2 = sk.ctxModulusFactor2();

  BOOST_TEST(NTL::NumBits(ptx_mod) == num_bits_ptx_mod);
  BOOST_TEST(NTL::NumBits(ctx_mod) >= num_bits_ctx_mod);

  BOOST_TEST_MESSAGE("Plaintext bit length: " << NTL::NumBits(ptx_mod));
  BOOST_TEST_MESSAGE("Ciphertext bit length: " << NTL::NumBits(ctx_mod));

  BOOST_TEST(((factor1 - 1) % ptx_mod) != 0);
  BOOST_TEST(((factor2 - 1) % ptx_mod) == 0);
  BOOST_TEST(((factor2 - 1) % (ptx_mod * ptx_mod)) != 0);
}

BOOST_AUTO_TEST_CASE(enc_idempotent_small_params) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  PtxSpace msg(11);

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  auto ctx = pk.encrypt(msg);
  auto dec_msg = sk.decrypt(ctx);

  BOOST_TEST(dec_msg == msg);
}

BOOST_AUTO_TEST_CASE(enc_idempotent_large_params) {
  const uint32_t num_bits_ptx_mod = 20;
  const uint32_t num_bits_ctx_mod = 2048;
  PtxSpace msg(1023);

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  auto ctx = pk.encrypt(msg);
  auto dec_msg = sk.decrypt(ctx);

  BOOST_TEST(dec_msg == msg);
}

BOOST_AUTO_TEST_CASE(rand_stream_is_updated_and_deterministic) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  PtxSpace msg(11);

  std::array<char, NTL_PRG_KEYLEN> stream_key{};
  std::string seed{"random"};
  auto stream_key_ptr = reinterpret_cast<unsigned char*>(stream_key.data());
  NTL::DeriveKey(stream_key_ptr, NTL_PRG_KEYLEN,
                 reinterpret_cast<unsigned char*>(seed.data()), seed.size());
  NTL::RandomStream stream(stream_key_ptr);
  auto stream_cpy = stream;

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  auto ctx1 = pk.encrypt(msg, pk.sampleRand(stream));
  auto ctx2 = pk.encrypt(msg, pk.sampleRand(stream));
  auto ctx3 = pk.encrypt(msg, pk.sampleRand(stream_cpy));
  auto ctx4 = pk.encrypt(msg, pk.sampleRand(stream_cpy));

  BOOST_TEST(ctx1 != ctx2);
  BOOST_TEST(ctx1 == ctx3);
  BOOST_TEST(ctx2 == ctx4);
}

BOOST_AUTO_TEST_CASE(sample_ctx_oblivious) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  auto ctx1 = pk.sampleCiphertext();
  auto ctx2 = pk.sampleCiphertext();
  auto val1 = sk.decrypt(ctx1);
  auto val2 = sk.decrypt(ctx2);

  auto ctx_mod = pk.ciphertextModulus();
  PtxSpace x = NTL::RandomBnd(ptx_mod);
  auto ctx = NTL::MulMod(ctx1, NTL::PowerMod(ctx2, x, ctx_mod), ctx_mod);
  auto dec_val = sk.decrypt(ctx);
  auto val = NTL::AddMod(val1, NTL::MulMod(val2, x, ptx_mod), ptx_mod);

  BOOST_TEST(dec_val == val);
}

BOOST_AUTO_TEST_CASE(poly_eval) {
  const uint32_t num_bits_ptx_mod = 5;
  const uint32_t num_bits_ctx_mod = 32;
  const uint32_t degree = 20;

  auto ptx_mod = NTL::GenPrime_long(num_bits_ptx_mod);
  auto sk = BESecretKey(ptx_mod, num_bits_ctx_mod);
  auto pk = sk.getPublicKey();

  auto ctx_mod = pk.ciphertextModulus();

  std::vector<PtxSpace> coeffs(degree + 1);
  NTL::VectorRandomBnd(ptx_mod, coeffs.data(), degree + 1);
  PtxSpace pt = NTL::RandomBnd(ptx_mod);

  PtxSpace val{0};
  for (uint32_t i = 0; i < coeffs.size(); i++) {
    auto term = NTL::MulMod(coeffs[i], NTL::PowerMod(pt, i, ptx_mod), ptx_mod);
    val = NTL::AddMod(val, term, ptx_mod);
  }

  std::vector<NTL::ZZ> coeffs_ctx(coeffs.size());
  for (size_t i = 0; i < coeffs.size(); i++) {
    coeffs_ctx[i] = pk.encrypt(coeffs[i]);
  }
  auto eval_ctx = pk.poly_eval(coeffs_ctx, pt);
  auto dec_val = sk.decrypt(eval_ctx);

  BOOST_TEST(dec_val == val);
}

BOOST_AUTO_TEST_SUITE_END()
