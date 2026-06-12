#pragma once

#include <NTL/ZZ.h>

#include <cstdint>
#include <vector>

#include "encryption.h"

namespace secureagg {
struct Report {
  PtxSpace pt;
  std::vector<PtxSpace> share;

  Report();
  Report(std::vector<PtxSpace>&& share);
};

// Oblivious share sampling protocol client algorithms
class Client {
  BEPublicKey pk_;
  std::vector<PtxSpace> masks_;
  unsigned int mdss_reps_;
  unsigned int t_priv_;
  PtxSpace pt_;
  uint32_t num_bytes_ptx_;
  uint32_t num_bytes_ctx_;

 public:
  // A single client instance can be re-used to compute reports for different
  // inputs.
  //
  // pk is the public key corresponding to the secret key held by the
  // randomness server, mdss_reps is the number of polynomials for each MDSS
  // instance and privacy_threshold is the privacy threshold of the MDSS scheme.
  Client(BEPublicKey&& pk, unsigned int mdss_reps,
         unsigned int privacy_threshold);

  // Outputs the message to be sent to the randomness server when
  // computing the report for a given value. pt denotes the
  // point at which the client evaluates polynomials.
  std::vector<uint8_t> initial_message(PtxSpace input, PtxSpace pt);

  // Read the message sent by the randomness server and return the report.
  Report read_message(const std::vector<uint8_t>& message);
};

// Oblivious share sampling protocol server algorithms
class Server {
  BESecretKey sk_;
  unsigned int mdss_reps_;
  uint32_t num_bytes_ptx_;
  uint32_t num_bytes_ctx_;

 public:
  Server(BESecretKey&& sk, unsigned int mdss_reps);

  // Read client message from client_message and write the response into
  // response_buffer.
  std::vector<uint8_t> read_message(
      const std::vector<uint8_t>& client_message) const;
};
}  // namespace secureagg
