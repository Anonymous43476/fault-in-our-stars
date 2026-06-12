#include "obliv_share_sample.h"

#include <NTL/BasicThreadPool.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "secureagg/encryption.h"

namespace secureagg {
Report::Report() : share{}, pt{} {}

Report::Report(std::vector<PtxSpace>&& share) : share(std::move(share)), pt{} {}

std::vector<uint8_t> ZZ_to_bytes(const std::vector<NTL::ZZ>& vals,
                                 uint32_t num_bytes_per_val) {
  std::vector<uint8_t> message_buffer(vals.size() * num_bytes_per_val);

  for (size_t i = 0; i < vals.size(); i++) {
    NTL::BytesFromZZ(message_buffer.data() + i * num_bytes_per_val, vals[i],
                     num_bytes_per_val);
  }

  return message_buffer;
}

std::vector<NTL::ZZ> bytes_to_ZZ(const std::vector<uint8_t>& message,
                                 uint32_t num_bytes_per_val) {
  auto num_elements = message.size() / num_bytes_per_val;
  std::vector<NTL::ZZ> vals(num_elements);

  for (size_t i = 0; i < num_elements; i++) {
    auto offset = i * num_bytes_per_val;
    NTL::ZZFromBytes(vals[i], message.data() + offset, num_bytes_per_val);
  }

  return vals;
}

std::vector<uint8_t> ptx_to_bytes(const std::vector<PtxSpace>& vals,
                                  uint32_t num_bytes_per_val) {
  std::vector<uint8_t> message_buffer(vals.size() * num_bytes_per_val);

  for (size_t i = 0; i < vals.size(); i++) {
    auto inp_it =
        static_cast<const uint8_t*>(static_cast<const void*>(&vals[i]));
    std::copy(inp_it, inp_it + num_bytes_per_val,
              message_buffer.data() + i * num_bytes_per_val);
  }

  return message_buffer;
}

std::vector<PtxSpace> bytes_to_ptx(const std::vector<uint8_t>& message,
                                   uint32_t num_bytes_per_val) {
  auto num_elements = message.size() / num_bytes_per_val;
  std::vector<PtxSpace> vals(num_elements);

  auto it = message.begin();
  for (size_t i = 0; i < num_elements; i++) {
    std::copy(it, it + num_bytes_per_val,
              static_cast<uint8_t*>(static_cast<void*>(&vals[i])));
    it += num_bytes_per_val;
  }

  return vals;
}

std::array<uint8_t, SHA256_DIGEST_LENGTH> derive_prg_seed(PtxSpace input,
                                                          long idx) {
  auto buffer =
      ptx_to_bytes({input, static_cast<PtxSpace>(idx)}, sizeof(PtxSpace));

  std::array<uint8_t, SHA256_DIGEST_LENGTH> output{};
  SHA256(buffer.data(), buffer.size(), output.data());

  return output;
}

Client::Client(BEPublicKey&& pk, unsigned int mdss_reps,
               unsigned int privacy_threshold)
    : pk_{std::move(pk)},
      mdss_reps_{mdss_reps},
      t_priv_{privacy_threshold},
      masks_(mdss_reps),
      pt_{} {
  auto ctx_mod = pk_.ciphertextModulus();
  num_bytes_ctx_ = NTL::NumBytes(ctx_mod);

  auto ptx_mod = pk_.plaintextModulus();
  num_bytes_ptx_ = static_cast<uint32_t>(std::ceil(std::log2(ptx_mod) / 8.0));
}

std::vector<uint8_t> Client::initial_message(PtxSpace input, PtxSpace pt) {
  auto ptx_mod = pk_.plaintextModulus();
  pt_ = pt;

  NTL::VectorRandomBnd(static_cast<long>(masks_.size()), masks_.data(),
                       ptx_mod);

  std::vector<NTL::ZZ> masked_share_ctxs(mdss_reps_);

  NTL_EXEC_RANGE(mdss_reps_, first, last)
  std::vector<NTL::ZZ> coeffs(t_priv_ + 1);

  for (long rep = first; rep < last; rep++) {
    auto seed = derive_prg_seed(input, rep);
    // The ideal way to set the RandomStream seed is to first call
    // NTL::DeriveKey on the RO output. However, this is not currently necessary
    // since the NTL PRG key length is equal to 256 bits.
    NTL::RandomStream stream(seed.data());

    for (long i = 1; i < t_priv_ + 1; i++) {
      // Sample random ciphertexts using stream
      pk_.sampleCiphertext(coeffs[i], stream);
    }
    // Mask and encrypt the input
    // Note that we don't need to use stream
    pk_.encrypt(coeffs[0], (masks_[rep] + input) % ptx_mod);

    pk_.poly_eval(masked_share_ctxs[rep], coeffs, pt_);
  }
  NTL_EXEC_RANGE_END

  // Serialize ciphertexts into message_buffer
  return ZZ_to_bytes(masked_share_ctxs, num_bytes_ctx_);
}

Report Client::read_message(const std::vector<uint8_t>& message) {
  auto report = Report(bytes_to_ptx(message, num_bytes_ptx_));
  report.pt = pt_;

  auto ptx_mod = pk_.plaintextModulus();
  for (size_t i = 0; i < mdss_reps_; i++) {
    report.share[i] -= masks_[i];
    if (report.share[i] < 0) {
      report.share[i] += ptx_mod;
    }
    report.share[i] = report.share[i] % ptx_mod;
  }

  return report;
}

Server::Server(BESecretKey&& sk, unsigned int mdss_reps)
    : sk_{std::move(sk)}, mdss_reps_{mdss_reps} {
  auto ctx_mod = sk_.ciphertextModulus();
  num_bytes_ctx_ = NTL::NumBytes(ctx_mod);

  auto ptx_mod = sk_.plaintextModulus();
  num_bytes_ptx_ = static_cast<uint32_t>(std::ceil(std::log2(ptx_mod) / 8.0));
}

std::vector<uint8_t> Server::read_message(
    const std::vector<uint8_t>& client_message) const {
  auto ctxs = bytes_to_ZZ(client_message, num_bytes_ctx_);

  std::vector<PtxSpace> vals(mdss_reps_);
  NTL_EXEC_RANGE(mdss_reps_, first, last)
  for (long i = first; i < last; i++) {
    vals[i] = sk_.decrypt(ctxs[i]);
  }
  NTL_EXEC_RANGE_END

  return ptx_to_bytes(vals, num_bytes_ptx_);
}

}  // namespace secureagg
