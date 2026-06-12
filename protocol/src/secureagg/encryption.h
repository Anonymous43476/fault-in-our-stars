#pragma once

#include <NTL/ZZ.h>

#include <map>
#include <optional>
#include <vector>

namespace secureagg {
// Plaintext space
typedef long PtxSpace;

// Benaloh Encryption Scheme's Public Key
class BEPublicKey {
  PtxSpace ptx_mod_;
  NTL::ZZ ctx_mod_;
  NTL::ZZ gen_;

 public:
  // Create a public key from scheme parameters.
  BEPublicKey(PtxSpace ptx_mod, NTL::ZZ&& ctx_mod, NTL::ZZ&& gen);

  // Getters for basic parameters of the scheme.
  PtxSpace plaintextModulus() const;
  const NTL::ZZ& ciphertextModulus() const;
  const NTL::ZZ& ptxSubGrpGen() const;

  // Generate a pseudorandom value that is co-prime to ctx_mod.
  // This uses the thread-local NTL random stream.
  void sampleRand(NTL::ZZ& rand) const;
  NTL::ZZ sampleRand() const;
  // Generate a pseudorandom value that is co-prime to ctx_mod, using
  // the output of stream. stream is updated accordingly.
  void sampleRand(NTL::ZZ& rand, NTL::RandomStream& stream) const;
  NTL::ZZ sampleRand(NTL::RandomStream& stream) const;

  // Encrypt 0 <= val < ptx_mod.
  // This uses the thread-local NTL random stream.
  void encrypt(NTL::ZZ& ctx, PtxSpace val) const;
  NTL::ZZ encrypt(PtxSpace val) const;
  // Encrypt 0 <= val < ptx_mod using randomness rand, where rand is a uniformly
  // random value co-prime to ctx_mod.
  void encrypt(NTL::ZZ& ctx, PtxSpace val, const NTL::ZZ& rand) const;
  NTL::ZZ encrypt(PtxSpace val, const NTL::ZZ& rand) const;

  // Obliviously sample a ciphertext pseudorandomly.
  // This uses the thread-local NTL random stream.
  void sampleCiphertext(NTL::ZZ& ctx) const;
  NTL::ZZ sampleCiphertext() const;
  // Obliviously sample a ciphertext pseudorandomly, using the output of stream.
  // stream is updated accordingly.
  void sampleCiphertext(NTL::ZZ& ctx, NTL::RandomStream& stream) const;
  NTL::ZZ sampleCiphertext(NTL::RandomStream& stream) const;

  // Given the encryptions of the coefficients coeff_ctx of a polynomial,
  // evaluate the polynomial at a point pt.
  // coeff_ctx[i] is the co-efficient of X^i.
  void poly_eval(NTL::ZZ& output, const std::vector<NTL::ZZ>& coeff_ctx, PtxSpace pt) const;
  NTL::ZZ poly_eval(const std::vector<NTL::ZZ>& coeff_ctx,
                    PtxSpace pt) const;
};

// Brute force computation of discrete-logarithm.
// Currently uses the Babe-step Giant-step algorithm.
class BruteForceDL {
  NTL::ZZ mod_;
  NTL::ZZ jump_;
  PtxSpace order_;
  PtxSpace num_jumps_;
  PtxSpace num_entries_;
  // TODO: Check if unordered_map improves performance.
  std::map<NTL::ZZ, PtxSpace> lookup_;

 public:
  // Initialization to compute DL with respect to 'base'. 'base' generates a
  // subgroup of order 'order' modulo 'mod'.
  //
  // space_param must be between 0 and 1 and indicates the size of the lookup
  // table. A smaller space_param will lead to larger runtime for each discrete
  // log computation.
  BruteForceDL(const NTL::ZZ& base, NTL::ZZ&& mod, PtxSpace order,
               double space_param);

  // Returns the discrete log of val with respect to base. The returned value
  // is between 0 and order - 1.
  PtxSpace operator()(const NTL::ZZ& val) const;
};

// Benaloh Encryption Scheme's Secret Key
class BESecretKey {
  PtxSpace ptx_mod_;
  NTL::ZZ factor1_, factor2_, ctx_mod_;
  NTL::ZZ dec_exp_;
  NTL::ZZ gen_;
  std::optional<BruteForceDL> dlog_;

  // Use the plaintext modulus ptx_mod_ to generate a pair of primes, the
  // product of which will be the ciphertext modulus ctx_mod_. ctx_mod_ has bit
  // length at least num_bits_ctx_mod.
  //
  // Requires ptx_mod_ to be well-defined.
  void genCiphertextModulus(uint32_t num_bits_ctx_mod);

  // Use the factors of the ciphertext modulus to compute a "non ptx_mod_
  // residue" gen_ as well as the decryption key dec_exp_. 0 <= space_param <= 1
  // denotes the space-time trade-off for computing discrete-log.
  //
  // Requires ptx_mod_, factor1_, factor2_, and ctx_mod_ to be well-defined.
  void genKey(double space_param);

 public:
  // Generate secret key for the Benaloh Encryption scheme with a given prime
  // and num_bits_ctx_mod bits ciphertext modulus length.
  //
  // space_param must be between 0 and 1 and indicates the size of the lookup
  // table for decryption. A smaller space_param will lead to larger runtime for
  // each decryption.
  explicit BESecretKey(PtxSpace ptx_mod, uint32_t num_bits_ctx_mod,
                       double space_param = 0.5);

  // Getters for basic parameters of the scheme.
  PtxSpace plaintextModulus() const;
  const NTL::ZZ& ptxSubGrpGen() const;
  const NTL::ZZ& ciphertextModulus() const;
  const NTL::ZZ& ctxModulusFactor1() const;
  const NTL::ZZ& ctxModulusFactor2() const;

  // Get corresponding public key
  BEPublicKey getPublicKey() const;

  // Decrypt ciphertext
  PtxSpace decrypt(const NTL::ZZ& ctx) const;
};
}  // namespace secureagg
