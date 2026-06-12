#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>
#include <NTL/lzz_pXFactoring.h>
#include <NTL/matrix.h>
#include <driver_types.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "array3d.h"
#include "ntl_utils.h"

using namespace std;
using namespace NTL;

using json = nlohmann::json;

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

constexpr uint32_t FIELD_SIZE = 100003;
inline void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess)
    throw std::runtime_error((std::string("CUDA error at ") + file + ":" +
                              std::to_string(line) + " — " +
                              cudaGetErrorString(err))
                                 .c_str());
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

inline void cudaCheckKernel(const char *file, int line) {
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess)
    throw std::runtime_error(std::string("Kernel launch error at ") + file +
                             ":" + std::to_string(line) + " - " +
                             cudaGetErrorString(launchErr));

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess)
    throw std::runtime_error(std::string("Kernel execution error at ") + file +
                             ":" + std::to_string(line) + " - " +
                             cudaGetErrorString(syncErr));
}

#define CUDA_CHECK_KERNEL() cudaCheckKernel(__FILE__, __LINE__)

vector<uint32_t> build_inv_table(uint32_t p) {
  vector<uint32_t> inv(p);

  inv[0] = 0;
  inv[1] = 1;

  for (size_t i = 2; i < p; i++) {
    inv[i] = p - (p / i) * inv[p % i] % p;
  }

  return inv;
}

// JSON utils
json readJsonFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file:" + filePath);
  }
  json jsonData;
  file >> jsonData;
  file.close();
  return jsonData;
}

optional<json> create_json() {
  try {
    return json{};
  } catch (exception &e) {
    cout << "Failed to initialize json object: " << e.what() << endl;
    return nullopt;
  }
}

// Main Structs
struct ColMetadata {
  int degree;
  uint32_t lead_coeff;
};

struct pivotInfo {
  int row;
  int degree;
  uint32_t lead_coeff;
};

struct pivotInfoAndTiming {
  pivotInfo pi;
  chrono::duration<double, milli> timeGRM;
  chrono::duration<double, milli> timeMemcpy;
  chrono::duration<double, milli> timeCalcPivot;
};

struct LatticeParams {
  zz_pX N;
  Vec<zz_pX> Ls;
  Vec<zz_p> w;
  unsigned short c;
  unsigned long ell;
};

struct InstanceParams {
  long field_size;
  unsigned short c;
  unsigned short ell;
  long n;
  long agreement;
};

struct Codeword {
  Vec<zz_p> x_coords;
  Mat<zz_p> y_coords;
};

struct Config {
  vector<unsigned long> batch_sizes;
  vector<unsigned long> c_vals;
  int num_threads = 1;
  bool continue_on_failure = false;
  bool terminate_after_last_config = false;
};

InstanceParams load_instance_parameters(json instance) {
  json params = instance["parameters"];
  return InstanceParams{params["field_size"], params["c"], params["ell"],
                        params["n"], params["agreement"]};
}

Codeword load_codeword(json instance, unsigned short c) {
  // Extract coordinates
  Vec<zz_p> x_coords;
  Mat<zz_p> y_coords;
  y_coords.SetDims(instance["parameters"]["n"], c);
  int row_counter(0);

  // Codeword points combine x and y coordinates, we separate them out here.
  for (const auto &elem : instance["codeword"]) {
    x_coords.append(zz_p(elem[0]));
    int col_counter = 0;
    for (const auto &y_val : elem[1]) {
      y_coords[row_counter][col_counter] = zz_p(y_val);
      col_counter++;
    }
    row_counter++;
  }

  return Codeword{x_coords, y_coords};
}

Config load_config(json config_data) {
  return Config{config_data["batch_sizes"], config_data["c_vals"],
                config_data.value("threads", 1),
                config_data.value("continue_on_failure", false),
                config_data.value("terminate_after_last_config", false)};
}

Vec<zz_p> get_dealer_value(InstanceParams params, Vec<zz_pX> poly_set,
                           Vec<zz_p> x_coords, Mat<zz_p> y_coords) {
  Vec<zz_p> dealer_x_coords;
  Vec<Vec<zz_p>> dealer_y_coords;

  for (int i = 0; i < x_coords.length(); i++) {
    bool matches_all = true;
    for (int j = 0; j < poly_set.length(); j++) {
      if (NTL::eval(poly_set[j], x_coords[i]) != y_coords[i][j]) {
        matches_all = false;
        break;
      }
    }
    if (matches_all) {
      dealer_x_coords.append(x_coords[i]);
      dealer_y_coords.append(y_coords[i]);
    }
  }
  Vec<zz_p> value;
  for (const auto &poly : poly_set) {
    value.append(ConstTerm(poly));
  }
  cout << "Interpolating Polynomials..." << endl;
  for (long j = poly_set.length(); j < y_coords.NumCols(); j++) {
    cout << "\r" << j - poly_set.length() << "/"
         << y_coords.NumCols() - poly_set.length() - 1 << flush;
    Vec<zz_p> interp_x_coords;
    Vec<zz_p> interp_y_coords;

    for (int k = 0; k < params.ell + 1; k++) {
      interp_x_coords.append(dealer_x_coords[k]);
      interp_y_coords.append(dealer_y_coords[k][j]);
    }
    zz_pX interp_poly = NTL::interpolate(interp_x_coords, interp_y_coords);
    value.append(ConstTerm(interp_poly));
  }
  cout << endl << "Done interpolating polynomials" << endl;

  return value;
}

unsigned long min_t(float batch_size, float c, float ell) {
  return ceil((1 / (c + 1)) * (batch_size + (c * ell)));
}

void remove_points(const Vec<zz_p> &x_coords, const Mat<zz_p> &y_coords,
                   const Vec<zz_pX> &resp_polys, Vec<zz_p> &x_indics) {
  for (long i = 0; i < x_coords.length(); i++) {
    zz_p x_coord = x_coords[i];
    Vec<zz_p> y_row = y_coords[i];

    bool matches_all = true;

    for (int j = 0; j < resp_polys.length(); j++) {
      zz_p eval_point = eval(resp_polys[j], x_coord);

      if (eval_point != y_row[j]) {
        matches_all = false;
        break;
      }
    }
    if (matches_all) {
      x_indics[i] = 0;
    }
  }
}

pair<Vec<zz_p>, Mat<zz_p>> remove_points_2(const Vec<zz_p> &x_coords,
                                           const Mat<zz_p> &y_coords,
                                           const Vec<zz_pX> &resp_polys) {
  Vec<zz_p> new_x_coords;
  Mat<zz_p> new_y_coords;
  new_y_coords.SetDims(y_coords.NumRows(), y_coords.NumCols());

  long row_insertion_counter = 0;
  for (long i = 0; i < x_coords.length(); i++) {
    zz_p x_coord = x_coords[i];
    Vec<zz_p> y_row = y_coords[i];

    bool matches_all = true;

    for (int j = 0; j < resp_polys.length(); j++) {
      zz_p eval_point = eval(resp_polys[j], x_coord);

      if (eval_point != y_row[j]) {
        matches_all = false;
        break;
      }
    }
    if (!matches_all) {
      new_x_coords.append(x_coord);
      new_y_coords[row_insertion_counter] = y_row;
      row_insertion_counter++;
    }
  }
  new_y_coords.SetDims(new_x_coords.length(), new_y_coords.NumCols());
  return make_pair(new_x_coords, new_y_coords);
}

long compute_norm(Vec<zz_pX> &row) {
  long max_degree = 0;
  for (auto &poly : row) {
    long current_degree = deg(poly);
    if (current_degree > max_degree) {
      max_degree = current_degree;
    }
  }
  return max_degree;
}

long find_sv_len(Mat<zz_pX> &matrix) {
  Vec<zz_pX> &row_0 = matrix[0];
  long sv_len = compute_norm(row_0);

  for (long i = 0; i < matrix.NumRows() - 1; i++) {
    Vec<zz_pX> &row_i = matrix[i + 1];
    long current_len = compute_norm(row_i);
    if (current_len < sv_len) {
      sv_len = current_len;
    }
  }
  return sv_len;
}

Vec<zz_pX> add_shortest_vectors(Mat<zz_pX> &matrix) {
  long shortest_len = find_sv_len(matrix);

  Vec<zz_pX> add_all_shortest_vecs;
  add_all_shortest_vecs.SetLength(matrix.NumCols());

  for (int itr = 0; itr < matrix.NumRows(); itr++) {
    Vec<zz_pX> itr_vec = matrix[itr];

    if (compute_norm(itr_vec) == shortest_len) {
      add_to_vec(add_all_shortest_vecs, itr_vec);
    }
  }

  return add_all_shortest_vecs;
}

// NTL operations to set up the matrix

zz_pX barycentric_interpolate(Vec<zz_pX> &lags, Vec<zz_p> &w, Mat<zz_p> &ys,
                              Vec<zz_p> &x_indics, long poly_num) {
  zz_pX result;
  for (long i = 0; i < lags.length(); i++) {
    result += x_indics[i] * lags[i] * w[i] * ys[i][poly_num];
  }
  return result;
}

Vec<zz_pX> create_interpols(Mat<zz_p> &y_coords, Vec<zz_p> &x_indics,
                            LatticeParams &params) {
  Vec<zz_pX> result;
  for (long i = 0; i < params.c; i++) {
    auto current =
        barycentric_interpolate(params.Ls, params.w, y_coords, x_indics, i);
    result.append(current);
  }
  return result;
}

LatticeParams lagrange_basis(Vec<zz_p> &x_coords, unsigned short c,
                             unsigned long ell) {
  zz_pX L = BuildFromRoots(x_coords);

  vec_zz_pX lags;
  lags.SetLength(x_coords.length());
  vec_zz_p w;

  for (int i = 0; i < x_coords.length(); i++) {
    auto &elem = x_coords[i];
    Vec<zz_p> single_root;
    single_root.append(elem);
    zz_pX root_i = BuildFromRoots(single_root);

    zz_pX lag_i = L / root_i;

    lags[i] = lag_i;
  }

  int counter(0);
  for (const auto &elem : lags) {
    zz_p lag_i_eval = eval(lags[counter], x_coords[counter]);
    w.append(inv(lag_i_eval));
    counter++;
  }

  return {L, lags, w, c, ell};
}

Mat<zz_pX> make_basis(zz_pX &base_N, unsigned short c, unsigned long ell,
                      Vec<zz_pX> &lagr_polys, Vec<zz_p> &x_coords,
                      Vec<zz_p> &locs) {
  Mat<zz_pX> M_D;
  M_D.SetDims(c + 1, c + 1);

  for (long i = 0; i < locs.length(); i++) {
    if (locs[i] == 0) {
      Vec<zz_p> single_root;
      single_root.append(x_coords[i]);
      zz_pX divisor = BuildFromRoots(single_root);
      base_N = base_N / divisor;
    }
  }

  for (int i = 0; i < c + 1; i++) {
    for (int j = 0; j < c + 1; j++) {
      if (i == 0) {
        if (j == 0) {
          zz_pX zero_zero;
          SetCoeff(zero_zero, (long)ell, 1);
          M_D[i][j] = zero_zero;
        } else {
          M_D[i][j] = lagr_polys[j - 1];
        }
      } else {
        if (i == j) {
          M_D[i][j] = base_N;
        } else {
          zz_pX zero;
          M_D[i][j] = zero;
        }
      }
    }
  }

  return M_D;
}

// GPU LAND

// HELPERS
__device__ __forceinline__ uint32_t dev_add(uint32_t a, uint32_t b) {
  uint32_t s = a + b;
  return s >= FIELD_SIZE ? s - FIELD_SIZE : s;
}

__device__ __forceinline__ uint32_t dev_sub(uint32_t a, uint32_t b) {
  return a >= b ? a - b : a + FIELD_SIZE - b;
}

__device__ __forceinline__ uint32_t dev_mul(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * b) % FIELD_SIZE);
}

__device__ uint32_t mod_pow(uint64_t base, uint32_t exp) {
  uint64_t res = 1;
  base %= FIELD_SIZE;

  while (exp > 0) {
    if (exp % 2 == 1) res = (res * base) % FIELD_SIZE;
    base = (base * base) % FIELD_SIZE;
    exp /= 2;
  }
  return (uint32_t)res;
}

__device__ uint32_t mod_inverse(uint32_t n) {
  return mod_pow(n, FIELD_SIZE - 2);
}

__device__ uint32_t calculate_zeroing_coeff_device(uint32_t hlc, uint32_t llc) {
  if (llc == 0) return 0;

  uint32_t inv_llc = mod_inverse(llc);
  uint64_t product = ((uint64_t)hlc * inv_llc) % FIELD_SIZE;

  if (product == 0) return 0;
  return (uint32_t)(FIELD_SIZE - product);
}

__global__ void add_multiple_of_row_kernel(uint32_t *d_data, int target_row,
                                           int source_row,
                                           uint32_t zeroing_coeff, int shift,
                                           int cols, int degree) {
  unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int col = blockIdx.y;

  if (z < degree && col < cols && (z + shift < degree)) {
    unsigned int target_idx =
        target_row * cols * degree + col * degree + (z + shift);
    unsigned int source_idx = source_row * cols * degree + col * degree + z;

    uint64_t mult_val = ((uint64_t)d_data[source_idx]) * zeroing_coeff;
    uint32_t reduced_mult_val = mult_val % FIELD_SIZE;

    uint32_t val = d_data[target_idx] + reduced_mult_val;

    if (val >= FIELD_SIZE) {
      val -= FIELD_SIZE;
    }

    d_data[target_idx] = val;
  }
}

__global__ void get_row_metadata_kernel(const uint32_t *d_data, long row,
                                        long cols, long degree,
                                        ColMetadata *d_out_metadata) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < cols) {
    long highest_deg = -1;
    uint32_t lead_coeff = 0;

    for (long z = degree - 1; z >= 0; z--) {
      size_t idx = (size_t)row * cols * degree + col * degree + z;
      int val = d_data[idx];
      if (val > 0) {
        highest_deg = z;
        lead_coeff = val;
        break;
      }
    }

    d_out_metadata[col].degree = highest_deg;
    d_out_metadata[col].lead_coeff = lead_coeff;
  }
}

__device__ pivotInfo find_pivot_device(ColMetadata *d_metadata, long row,
                                       long cols) {
  int pivot_col = -1;
  int pivot_degree = -1;
  uint32_t pivot_lead_coeff = 0;

  for (int col = 0; col < cols; col++) {
    if (d_metadata[col].degree >= pivot_degree) {
      pivot_col = col;
      pivot_degree = d_metadata[col].degree;
      pivot_lead_coeff = d_metadata[col].lead_coeff;
    }
  }

  return {pivot_col, pivot_degree, pivot_lead_coeff};
}

pivotInfoAndTiming get_pivot_gpu(uint32_t *d_data, long row, long cols,
                                 long degree, ColMetadata *d_metadata,
                                 ColMetadata *h_metadata) {
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double, std::milli>;

  Duration timeGRM{}, timeMemcpy{}, timeCalcPivot{};

  auto t0 = Clock::now();
  int threads = 256;
  long blocks = (cols + threads - 1) / threads;

  get_row_metadata_kernel<<<blocks, threads>>>(d_data, row, cols, degree,
                                               d_metadata);
  auto t1 = Clock::now();

  cudaMemcpy(h_metadata, d_metadata, cols * sizeof(ColMetadata),
             cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();

  int pivot_col = -1;
  int pivot_degree = -1;
  uint32_t pivot_lead_coeff = 0;

  for (int col = 0; col < cols; col++) {
    if (h_metadata[col].degree >= pivot_degree) {
      pivot_col = col;
      pivot_degree = h_metadata[col].degree;
      pivot_lead_coeff = h_metadata[col].lead_coeff;
    }
  }

  auto t3 = Clock::now();

  timeGRM = t1 - t0;
  timeMemcpy = t2 - t1;
  timeCalcPivot = t3 - t2;

  pivotInfo pi = {pivot_col, pivot_degree, pivot_lead_coeff};

  return {pi, timeGRM, timeMemcpy, timeCalcPivot};
}

struct gpuPivotInfo {
  uint32_t row_idx;
  long degree;
};

struct EROConfig {
  uint32_t target_row_idx;
  uint32_t source_row_idx;
  uint32_t zeroing_coefficient;
  long shift;
};

__device__ inline void push_row_with_pivot(gpuPivotInfo *rows_with_pivot,
                                           uint32_t *rwp_tops,
                                           uint32_t num_rows, uint32_t col,
                                           uint32_t row, long degree) {
  uint32_t top = rwp_tops[col];

  // Optional safety check (remove in release for perf)
  if (top >= num_rows) return;

  rows_with_pivot[col * num_rows + top] = {row, degree};
  rwp_tops[col] = top + 1;
}

__device__ gpuPivotInfo pop_rows_with_pivot(gpuPivotInfo *rows_with_pivot,
                                            uint32_t *rwp_tops,
                                            uint32_t num_rows, uint32_t col) {
  uint32_t top = rwp_tops[col] - 1;
  rwp_tops[col] = top;
  return rows_with_pivot[col * num_rows + top];
}

void print_rows_with_pivot(gpuPivotInfo *rows_with_pivot, uint32_t *rwp_tops,
                           uint32_t num_cols, uint32_t num_rows) {
  gpuPivotInfo *h_rwp = new gpuPivotInfo[num_cols * num_rows];
  uint32_t *h_rwp_tops = new uint32_t[num_cols];

  cudaMemcpy(h_rwp, rows_with_pivot, num_cols * num_rows * sizeof(gpuPivotInfo),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_rwp_tops, rwp_tops, num_cols * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  for (uint32_t col = 0; col < num_cols; col++) {
    cout << "Col " << col << " (size=" << h_rwp_tops[col] << "): ";

    for (uint32_t i = 0; i < h_rwp_tops[col]; i++) {
      auto &entry = h_rwp[col * num_rows + i];
      cout << "(" << entry.row_idx << ", " << entry.degree << ") ";
    }

    cout << endl;
  }

  delete[] h_rwp;
  delete[] h_rwp_tops;
}

__device__ void update_metadata_for_row(
    uint32_t *d_matrix_data, uint32_t num_rows, uint32_t num_cols,
    uint32_t max_degree, uint32_t *d_conflicts_stack,
    uint32_t *d_conflicts_stack_top, gpuPivotInfo *rows_with_pivot,
    uint32_t *rwp_tops, uint32_t target_row) {
  long best_pivot_col = -1;
  long best_degree = -1;

  for (long current_col = 0; current_col < num_cols; current_col++) {
    long current_degree = -1;

    for (long z_idx = max_degree - 1; z_idx >= 0; z_idx--) {
      if (d_matrix_data[(target_row * num_cols * max_degree) +
                        (current_col * max_degree) + z_idx] > 0) {
        current_degree = z_idx;
        break;
      }
    }

    if (current_degree >= best_degree) {
      best_degree = current_degree;
      best_pivot_col = current_col;
    }
  }

  if (best_degree >= 0) {
    push_row_with_pivot(rows_with_pivot, rwp_tops, num_rows, best_pivot_col,
                        target_row, best_degree);

    if (rwp_tops[best_pivot_col] > 1) {
      uint32_t idx = atomicAdd(&d_conflicts_stack_top[0], 1);
      d_conflicts_stack[idx] = best_pivot_col;
    }
  }
}

__global__ void update_metadata_for_row_kernel(

    uint32_t *d_matrix_data, uint32_t num_rows, uint32_t num_cols,
    uint32_t max_degree, uint32_t *d_conflicts_stack,
    uint32_t *d_conflicts_stack_top, gpuPivotInfo *rows_with_pivot,
    uint32_t *rwp_tops, uint32_t target_row) {
  update_metadata_for_row(d_matrix_data, num_rows, num_cols, max_degree,
                          d_conflicts_stack, d_conflicts_stack_top,
                          rows_with_pivot, rwp_tops, target_row);
}

__global__ void update_metadata_post_ero_CLAUDE(
    uint32_t *d_matrix_data, uint32_t num_rows, uint32_t num_cols,
    uint32_t max_degree, uint32_t *d_conflicts_stack,
    uint32_t *d_conflicts_stack_top, gpuPivotInfo *rows_with_pivot,
    uint32_t *rwp_tops, EROConfig *cERO, long *d_col_best_degree) {
  if (cERO[0].shift < 0) return;

  uint32_t target_row = cERO[0].target_row_idx;
  uint32_t col = blockIdx.x;
  if (col >= num_cols) return;

  extern __shared__ long s_warp_best[];

  uint32_t tid = threadIdx.x;
  uint32_t lane = tid % 32;
  uint32_t warp_id = tid / 32;
  uint32_t num_warps = (blockDim.x + 31) / 32;

  uint32_t *col_data = d_matrix_data +
                       (size_t)target_row * num_cols * max_degree +
                       col * max_degree;

  long local_best = -1;
  for (long z = (long)max_degree - 1 - tid; z >= 0; z -= blockDim.x) {
    if (col_data[z] > 0) {
      local_best = z;
      break;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    long other = __shfl_down_sync(0xffffffff, local_best, offset);
    if (other > local_best) local_best = other;
  }

  if (lane == 0) s_warp_best[warp_id] = local_best;
  __syncthreads();

  if (tid == 0) {
    long best_degree = -1;
    for (uint32_t w = 0; w < num_warps; w++) {
      if (s_warp_best[w] > best_degree) best_degree = s_warp_best[w];
    }

    d_col_best_degree[col] = best_degree;
    // if (best_degree >= 0) {
    //   push_row_with_pivot(rows_with_pivot, rwp_tops, num_rows, col,
    //   target_row,
    //                       best_degree);
    //   if (rwp_tops[col] > 1) {
    //     uint32_t idx = atomicAdd(&d_conflicts_stack_top[0], 1);
    //     d_conflicts_stack[idx] = col;
    //   }
    // }
  }
}

__global__ void push_best_pivot(uint32_t num_rows, uint32_t num_cols,
                                uint32_t *d_conflicts_stack,
                                uint32_t *d_conflicts_stack_top,
                                gpuPivotInfo *rows_with_pivot,
                                uint32_t *rwp_tops, EROConfig *cERO,
                                long *d_col_best_degree) {
  if (cERO[0].shift < 0) return;

  // Single thread finds best col and does exactly one push
  uint32_t target_row = cERO[0].target_row_idx;
  long best_degree = -1;
  uint32_t best_col = 0;

  for (uint32_t col = 0; col < num_cols; col++) {
    if (d_col_best_degree[col] >= best_degree) {
      best_degree = d_col_best_degree[col];
      best_col = col;
    }
  }

  if (best_degree >= 0) {
    push_row_with_pivot(rows_with_pivot, rwp_tops, num_rows, best_col,
                        target_row, best_degree);
    if (rwp_tops[best_col] > 1) {
      uint32_t idx = atomicAdd(&d_conflicts_stack_top[0], 1);
      d_conflicts_stack[idx] = best_col;
    }
  }
}

__global__ void set_current_ero(uint32_t *d_matrix_data, uint32_t num_rows,
                                uint32_t num_cols, uint32_t max_degree,
                                uint32_t *d_conflicts_stack,
                                uint32_t *d_conflicts_stack_top,
                                gpuPivotInfo *rows_with_pivot,
                                uint32_t *rwp_tops, EROConfig *cERO) {
  if (d_conflicts_stack_top[0] != 0) {
    uint32_t top = d_conflicts_stack_top[0];
    uint32_t conflict_col = d_conflicts_stack[top - 1];
    d_conflicts_stack_top[0] = top - 1;  // (optional, if you're popping)

    gpuPivotInfo gpi_a =
        pop_rows_with_pivot(rows_with_pivot, rwp_tops, num_rows, conflict_col);
    gpuPivotInfo gpi_b =
        pop_rows_with_pivot(rows_with_pivot, rwp_tops, num_rows, conflict_col);

    gpuPivotInfo higher;
    gpuPivotInfo lower;

    if (gpi_a.degree >= gpi_b.degree) {
      higher = gpi_a;
      lower = gpi_b;
    } else {
      higher = gpi_b;
      lower = gpi_a;
    }

    push_row_with_pivot(rows_with_pivot, rwp_tops, num_rows, conflict_col,
                        lower.row_idx, lower.degree);

    uint32_t higher_lc =
        d_matrix_data[(higher.row_idx * num_cols * max_degree) +
                      (conflict_col * max_degree) + higher.degree];

    uint32_t lower_lc =
        d_matrix_data[(lower.row_idx * num_cols * max_degree) +
                      (conflict_col * max_degree) + lower.degree];
    uint32_t zeroing_coeff =
        calculate_zeroing_coeff_device(higher_lc, lower_lc);

    cERO[0] = {higher.row_idx, lower.row_idx, zeroing_coeff,
               higher.degree - lower.degree};
  } else {
    cERO[0] = {0, 0, 0, -1};
  }
}

__global__ void perform_current_ero(uint32_t *d_matrix_data, uint32_t num_rows,
                                    uint32_t num_cols, uint32_t max_degree,
                                    EROConfig *cERO) {
  if (cERO[0].shift < 0) {
    return;
  }

  unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int col = blockIdx.y;

  EROConfig ero = cERO[0];

  if (z < max_degree && col < num_cols && (z + ero.shift < max_degree)) {
    unsigned int target_idx = (ero.target_row_idx * num_cols * max_degree) +
                              (col * max_degree) + (z + ero.shift);
    unsigned int source_idx =
        (ero.source_row_idx * num_cols * max_degree) + (col * max_degree) + z;

    uint64_t mult_val =
        ((uint64_t)d_matrix_data[source_idx]) * ero.zeroing_coefficient;
    uint32_t reduced_mult_val = mult_val % FIELD_SIZE;

    uint32_t val = d_matrix_data[target_idx] + reduced_mult_val;
    if (val >= FIELD_SIZE) {
      val -= FIELD_SIZE;
    }

    d_matrix_data[target_idx] = val;
  }
}

__global__ void update_metadata_post_ero(uint32_t *d_matrix_data,
                                         uint32_t num_rows, uint32_t num_cols,
                                         uint32_t max_degree,
                                         uint32_t *d_conflicts_stack,
                                         uint32_t *d_conflicts_stack_top,
                                         gpuPivotInfo *rows_with_pivot,
                                         uint32_t *rwp_tops, EROConfig *cERO) {
  if (cERO[0].shift < 0) {
    return;
  }
  update_metadata_for_row(d_matrix_data, num_rows, num_cols, max_degree,
                          d_conflicts_stack, d_conflicts_stack_top,
                          rows_with_pivot, rwp_tops, cERO[0].target_row_idx);
}

void weak_popov_only_gpu(Array3D<uint32_t> &arr, unsigned long ell,
                         uint32_t row_0_pivot_idx, long row_0_pivot_degree,
                         vector<vector<gpuPivotInfo>> initial_rows_with_pivot) {
  if (row_0_pivot_idx == 0) {
    return;
  }
  Duration dur_mem_allocation, dur_graph_setup, dur_main_loop, dur_final_memcpy;

  auto t0 = Clock::now();

  uint32_t num_rows = arr.dim_x();
  uint32_t num_cols = arr.dim_y();
  uint32_t max_degree = arr.dim_z();

  long n = max_degree - 1;
  long c = num_rows - 1;

  const int THREADS = 256;
  const int NUM_WARPS = (THREADS + 31) / 32;
  size_t smem = NUM_WARPS * sizeof(long);

  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks((max_degree + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 num_cols);

  dim3 meta_blocks(num_cols);
  dim3 meta_threads(THREADS);

  cout << "Num Blocks: (" << numBlocks.x << ", " << numBlocks.y << ", "
       << numBlocks.z << ")" << endl;

  unsigned long expected_steps = (n - ell) * c;
  cout << "Expected number of steps: " << expected_steps << endl;
  cout << "Starting reduction..." << endl;

  // --------------------------------- //
  // Set up data structures on device  //
  // --------------------------------- //

  // Matrix
  cout << num_rows << " " << num_cols << " " << max_degree << " "
       << sizeof(uint32_t) << endl;
  uint64_t array_size = num_rows * num_cols * max_degree * sizeof(uint32_t);
  uint32_t *d_matrix_data = nullptr;
  cudaError_t err = cudaMalloc(&d_matrix_data, array_size);
  if (err != cudaSuccess) {
    throw runtime_error("GPU Out of Memory: " +
                        string(cudaGetErrorString(err)));
  }
  CUDA_CHECK(cudaMemcpy(d_matrix_data, arr.data(), array_size,
                        cudaMemcpyHostToDevice));
  cout << "Array size: " << array_size << " bytes" << endl;

  // Conflicts stack
  uint32_t *d_conflicts_stack = nullptr;
  uint32_t *d_conflicts_stack_top = nullptr;
  CUDA_CHECK(cudaMalloc(&d_conflicts_stack, num_cols * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_conflicts_stack_top, sizeof(uint32_t)));
  uint32_t h_conflicts_stack_top = 1;
  CUDA_CHECK(cudaMemcpy(d_conflicts_stack_top, &h_conflicts_stack_top,
                        sizeof(uint32_t), cudaMemcpyHostToDevice));
  uint32_t *h_conflicts_stack = (uint32_t *)malloc(num_cols * sizeof(uint32_t));
  h_conflicts_stack[0] = row_0_pivot_idx;
  CUDA_CHECK(cudaMemcpy(d_conflicts_stack, h_conflicts_stack,
                        num_cols * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Col -> rows with col as pivot
  gpuPivotInfo *h_rows_with_pivot =
      (gpuPivotInfo *)malloc(num_cols * num_rows * sizeof(gpuPivotInfo));
  uint32_t *h_rwp_tops = (uint32_t *)malloc(num_cols * sizeof(uint32_t));
  for (int i = 0; i < initial_rows_with_pivot.size(); i++) {
    vector<gpuPivotInfo> cur_vec = initial_rows_with_pivot[i];
    for (int j = 0; j < cur_vec.size(); j++) {
      h_rows_with_pivot[i * num_rows + j] = cur_vec[j];
    }
    h_rwp_tops[i] = cur_vec.size();
  }

  gpuPivotInfo *rows_with_pivot = nullptr;
  uint32_t *rwp_tops = nullptr;
  CUDA_CHECK(
      cudaMalloc(&rows_with_pivot, num_cols * num_rows * sizeof(gpuPivotInfo)));
  CUDA_CHECK(cudaMalloc(&rwp_tops, num_cols * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(rows_with_pivot, h_rows_with_pivot,
                        num_cols * num_rows * sizeof(gpuPivotInfo),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rwp_tops, h_rwp_tops, num_cols * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  free(h_conflicts_stack);
  free(h_rows_with_pivot);
  free(h_rwp_tops);

  // Current ero
  EROConfig *cERO = nullptr;
  CUDA_CHECK(cudaMalloc(&cERO, sizeof(EROConfig)));

  // Scratch buffer for calculating row metadata
  long *d_col_best_degree = nullptr;
  CUDA_CHECK(cudaMalloc(&d_col_best_degree, num_cols * sizeof(long)));

  auto t1 = Clock::now();
  dur_mem_allocation = t1 - t0;
  auto t2 = Clock::now();

  // --------------------------------- //
  // Perform Lattice Reduction         //
  // --------------------------------- //

  // Setup GPU Stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  set_current_ero<<<1, 1, 0, stream>>>(
      d_matrix_data, num_rows, num_cols, max_degree, d_conflicts_stack,
      d_conflicts_stack_top, rows_with_pivot, rwp_tops, cERO);
  perform_current_ero<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_matrix_data, num_rows, num_cols, max_degree, cERO);
  update_metadata_post_ero_CLAUDE<<<meta_blocks, meta_threads, smem, stream>>>(
      d_matrix_data, num_rows, num_cols, max_degree, d_conflicts_stack,
      d_conflicts_stack_top, rows_with_pivot, rwp_tops, cERO,
      d_col_best_degree);

  push_best_pivot<<<1, 1, 0, stream>>>(num_rows, num_cols, d_conflicts_stack,
                                       d_conflicts_stack_top, rows_with_pivot,
                                       rwp_tops, cERO, d_col_best_degree);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  auto t3 = Clock::now();
  dur_graph_setup = t3 - t2;

  // Main redution loop;
  for (int i = 0; i < expected_steps; i++) {
    cudaGraphLaunch(graphExec, stream);
  }
  CUDA_CHECK_KERNEL();

  auto t4 = Clock::now();
  dur_main_loop = t4 - t3;

  CUDA_CHECK(cudaMemcpy(arr.data(), d_matrix_data, array_size,
                        cudaMemcpyDeviceToHost));
  auto t5 = Clock::now();
  dur_final_memcpy = t5 - t4;

  cout << "---- WPF Timings ----" << endl;
  cout << "Initial allocation: \t" << dur_mem_allocation.count() << "ms"
       << endl;
  cout << "Graph setup: \t\t" << dur_graph_setup.count() << "ms" << endl;
  cout << "Main loop: \t\t" << dur_main_loop.count() << "ms" << endl;
  cout << "Final memcpy: \t\t" << dur_final_memcpy.count() << "ms" << endl
       << endl;

  // Free/Teardown everything
  cudaFree(d_matrix_data);
  cudaFree(d_conflicts_stack);
  cudaFree(d_conflicts_stack_top);
  cudaFree(rows_with_pivot);
  cudaFree(rwp_tops);
  cudaFree(cERO);
  cudaFree(d_col_best_degree);

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

// Basis Setup Kernels

__global__ void kernel_divide_L(
    const uint32_t *__restrict__ d_L,  // [n+1]
    const uint32_t *__restrict__ d_x,  // [n]
    uint32_t *d_lags,                  // [n * n]  (coeff 0..n-1 for each i)
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  uint32_t xi = d_x[i];
  uint32_t *qi = d_lags + (size_t)i * n;  // output slice for this i

  // Synthetic division: start from the leading coefficient of L.
  // L is monic of degree n, so L[n] = 1.
  // The quotient L_i has degree n-1, so n coefficients (indices 0..n-1).
  uint32_t carry = d_L[n];  // == 1
  // We fill qi from the highest degree down.
  for (int k = n - 1; k >= 0; --k) {
    qi[k] = carry;
    // carry for next step: L[k] + xi * qi[k]
    carry = dev_add(d_L[k], dev_mul(xi, carry));
    // (carry at k=0 is the remainder, should be 0 if xi is a root of L)
  }
}

__global__ void kernel_eval_and_invert(
    const uint32_t *__restrict__ d_lags,  // [n * n]
    const uint32_t *__restrict__ d_x,     // [n]
    uint32_t *d_w,                        // [n]  output weights
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  uint32_t xi = d_x[i];
  const uint32_t *qi = d_lags + (size_t)i * n;

  // Horner evaluation of degree-(n-1) polynomial at xi.
  // Coefficients stored as qi[0] + qi[1]*x + ... + qi[n-1]*x^{n-1}.
  uint32_t val = qi[n - 1];
  for (int k = n - 2; k >= 0; --k) {
    val = dev_add(dev_mul(val, xi), qi[k]);
  }

  d_w[i] = mod_inverse(val);
}

struct LatticeParamsGPU {
  // Polynomial N(x) = L(x): host-side coefficients, degree n (length n+1).
  std::vector<uint32_t> N;  // N[d] = coeff of x^d

  // Lagrange basis polynomials: row-major [n][n], each L_i has degree n-1.
  std::vector<uint32_t> lags;  // lags[i*n + d] = coeff of x^d in L_i

  // Barycentric weights.
  std::vector<uint32_t> w;  // w[i] = 1 / L_i(x_i)

  uint32_t c;
  uint32_t ell;
};

static std::vector<uint32_t> build_product_poly(
    const std::vector<uint32_t> &x) {
  int n = (int)x.size();
  // poly starts as "1"
  std::vector<uint32_t> L(n + 1, 0);
  L[0] = 1;
  int cur_deg = 0;

  for (int i = 0; i < n; ++i) {
    uint32_t neg_xi = (x[i] == 0) ? 0 : FIELD_SIZE - x[i];
    // multiply L by (X - x[i]):  L = L * X + L * (-x[i])
    for (int d = cur_deg; d >= 0; --d) {
      uint32_t c = L[d];
      L[d + 1] = (L[d + 1] + c) % FIELD_SIZE;
      L[d] = (uint32_t)(((uint64_t)c * neg_xi) % FIELD_SIZE);
    }
    ++cur_deg;
  }
  return L;  // length n+1, L[n] == 1 (monic)
}

LatticeParamsGPU lagrange_basis_gpu(
    const std::vector<uint32_t> &x_coords,  // n ≈ 40000
    uint32_t c, uint32_t ell) {
  int n = (int)x_coords.size();
  if (n == 0) throw std::invalid_argument("x_coords must be non-empty");

  // 1. Build L on the CPU (product of all (x - x_i)).
  std::vector<uint32_t> L = build_product_poly(x_coords);  // length n+1

  // 2. Allocate device memory.
  uint32_t *d_L = nullptr, *d_x = nullptr, *d_lags = nullptr, *d_w = nullptr;

  size_t lag_bytes = (size_t)n * n * sizeof(uint32_t);  // n rows × n coeffs
  CUDA_CHECK(cudaMalloc(&d_L, (n + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_lags, lag_bytes));
  CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(uint32_t)));

  // 3. Copy inputs to device.
  CUDA_CHECK(cudaMemcpy(d_L, L.data(), (n + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x_coords.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // 4. Kernel 1: divide L by each (x - x_i) in parallel.
  //    Each thread does O(n) work, so total GPU work is O(n^2) — same as CPU,
  //    but parallelized across n ≈ 40000 threads.
  {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_divide_L<<<blocks, threads>>>(d_L, d_x, d_lags, n);
    cudaDeviceSynchronize();
  }
  CUDA_CHECK_KERNEL();

  // 5. Kernel 2: evaluate L_i(x_i) and invert, yielding w[i].
  {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_eval_and_invert<<<blocks, threads>>>(d_lags, d_x, d_w, n);
    cudaDeviceSynchronize();
  }
  CUDA_CHECK_KERNEL();

  // 6. Copy results back to host.
  LatticeParamsGPU params;
  params.N = L;  // N = L(x), already on host
  params.lags.resize((size_t)n * n);
  params.w.resize(n);
  params.c = c;
  params.ell = ell;

  CUDA_CHECK(cudaMemcpy(params.lags.data(), d_lags, lag_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(params.w.data(), d_w, n * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  // 7. Cleanup.
  cudaFree(d_L);
  cudaFree(d_x);
  cudaFree(d_lags);
  cudaFree(d_w);

  return params;
}

template <int TILE>
__global__ void gemm_transpose_A(const uint32_t *__restrict__ A,  // [K × M]
                                 const uint32_t *__restrict__ B,  // [K × N]
                                 uint32_t *C,                     // [M × N]
                                 int M, int N, int K) {
  // Shared memory tiles.
  // sA holds a TILE×TILE strip of A^T → we load A[k][m] into sA[tx][ty]
  // sB holds a TILE×TILE strip of B   → we load B[k][n] into sB[ty][tx]
  __shared__ uint32_t sA[TILE][TILE];  // sA[m_local][k_local]
  __shared__ uint32_t sB[TILE][TILE];  // sB[k_local][n_local]

  int row = blockIdx.y * TILE + threadIdx.y;  // output row  (degree d)
  int col = blockIdx.x * TILE + threadIdx.x;  // output col  (poly   j)

  // We accumulate in 64-bit to defer modular reduction inside the k-loop.
  // Each partial sum is at most TILE * (p-1)^2 ≈ 32 * (100002)^2 ≈ 3.2e11,
  // which fits in a uint64_t (max ~1.8e19).  We reduce mod p every tile.
  uint64_t acc = 0;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int k_A = t * TILE + threadIdx.x;  // k index when loading A for sA
    int k_B = t * TILE + threadIdx.y;  // k index when loading B for sB

    // Load A[k_A][row] into sA[threadIdx.y][threadIdx.x]
    // A is [K×M], element A[k][m] = A[k*M + m]
    sA[threadIdx.y][threadIdx.x] =
        (k_A < K && row < M) ? A[(size_t)k_A * M + row] : 0u;

    // Load B[k_B][col] into sB[threadIdx.y][threadIdx.x]
    // B is [K×N], element B[k][n] = B[k*N + col]
    sB[threadIdx.y][threadIdx.x] =
        (k_B < K && col < N) ? B[(size_t)k_B * N + col] : 0u;

    __syncthreads();

// Accumulate dot product over the tile strip.
#pragma unroll
    for (int u = 0; u < TILE; ++u) {
      acc += (uint64_t)sA[threadIdx.y][u] * sB[u][threadIdx.x];
    }
    // Reduce mod p once per tile to keep acc < 2^64.
    acc %= FIELD_SIZE;

    __syncthreads();
  }

  if (row < M && col < N) {
    C[(size_t)row * N + col] = (uint32_t)(acc % FIELD_SIZE);
  }
}

// ── kernel: build the scalar matrix s[i][j] = x_indics[i]*w[i]*ys[i][j] ─────
//
// ys      : [n × c]  (row-major: ys[i*c + j])
// x_indics: [n]
// w       : [n]
// s_out   : [n × c]  (row-major, same layout)

__global__ void kernel_build_s(const uint32_t *__restrict__ ys,  // [n × c]
                               const uint32_t *__restrict__ x_indics,  // [n]
                               const uint32_t *__restrict__ w,         // [n]
                               uint32_t *s_out,  // [n × c]
                               int n, int c) {
  // One thread per (i, j) pair.
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // point index
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // poly  index

  if (i >= n || j >= c) return;

  uint32_t xi = x_indics[i];
  if (xi == 0) {
    // x_indics[i] == 0 means point i is excluded (zeroed out in CPU code).
    s_out[i * c + j] = 0;
    return;
  }

  uint32_t wi = w[i];
  uint32_t yij = ys[i * c + j];

  // s[i][j] = x_indics[i] * w[i] * ys[i][j]  (mod p)
  // x_indics[i] is 0 or 1 in the original code, so the multiply is free,
  // but we handle the general case for correctness.
  uint64_t val = (uint64_t)xi * wi % FIELD_SIZE;
  val = val * yij % FIELD_SIZE;
  s_out[i * c + j] = (uint32_t)val;
}

std::vector<uint32_t> create_interpols_gpu(
    const std::vector<uint32_t> &lags,  // [n × n], from lagrange_basis_gpu
    const std::vector<uint32_t> &w,     // [n]
    const std::vector<uint32_t> &ys,    // [n × c], caller flattens y_coords
    const std::vector<uint32_t> &x_indics,  // [n]
    int n, int c) {
  if ((int)lags.size() != n * n)
    throw std::invalid_argument("lags size mismatch");
  if ((int)w.size() != n) throw std::invalid_argument("w size mismatch");
  if ((int)ys.size() != n * c) throw std::invalid_argument("ys size mismatch");
  if ((int)x_indics.size() != n)
    throw std::invalid_argument("x_indics size mismatch");

  // ── allocate device buffers ───────────────────────────────────────────────
  uint32_t *d_lags = nullptr, *d_w = nullptr, *d_ys = nullptr, *d_xi = nullptr,
           *d_s = nullptr, *d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_lags, (size_t)n * n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_ys, (size_t)n * c * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_xi, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_s, (size_t)n * c * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * c * sizeof(uint32_t)));

  // ── copy inputs to device ─────────────────────────────────────────────────
  CUDA_CHECK(cudaMemcpy(d_lags, lags.data(), (size_t)n * n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_w, w.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), (size_t)n * c * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xi, x_indics.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // ── step 1: build scalar matrix s[i][j] ──────────────────────────────────
  {
    dim3 block(16, 16);
    dim3 grid((c + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    kernel_build_s<<<grid, block>>>(d_ys, d_xi, d_w, d_s, n, c);
    cudaDeviceSynchronize();
  }
  CUDA_CHECK_KERNEL();

  // ── step 2: out = lags^T * s  (GF(p) GEMM) ───────────────────────────────
  // lags is [n×n] (K=n, M=n), s is [n×c] (N=c), out is [n×c]
  {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((c + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gemm_transpose_A<TILE><<<grid, block>>>(d_lags, d_s, d_out, n, c, n);
    cudaDeviceSynchronize();
  }
  CUDA_CHECK_KERNEL();

  // ── copy result back ──────────────────────────────────────────────────────
  std::vector<uint32_t> out_polys((size_t)n * c);
  cudaMemcpy(out_polys.data(), d_out, (size_t)n * c * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  // ── cleanup ───────────────────────────────────────────────────────────────
  cudaFree(d_lags);
  cudaFree(d_w);
  cudaFree(d_ys);
  cudaFree(d_xi);
  cudaFree(d_s);
  cudaFree(d_out);

  return out_polys;
  // out_polys[d * c + j] = coefficient of x^d in the j-th interpolated
  // polynomial
}

// More decoding steps

Mat<zz_pX> first_step(Vec<zz_p> &x_coords, Mat<zz_p> &y_coords,
                      Vec<zz_pX> &a_list, Vec<zz_p> &x_indics, zz_pX &base_N,
                      unsigned short c, unsigned long ell) {
  Duration dur_mb, dur_convert_array, dur_wpf, dur_convert_array_back;

  long one_count = std::count(x_indics.begin(), x_indics.end(), zz_p(1));
  Vec<zz_pX> lagr_polys;
  if (one_count == x_coords.length()) {
    lagr_polys = a_list;
  } else {
    // TODO: allow for less than all 1s again
    throw invalid_argument("Expected all 1s in x_indics");
    // lagr_polys = create_interpols(y_coords, x_indics, params);
  }

  auto t0 = Clock::now();
  auto M_D = make_basis(base_N, c, ell, a_list, x_coords, x_indics);
  auto t1 = Clock::now();
  dur_mb = t1 - t0;

  vector<vector<gpuPivotInfo>> rows_with_pivot;
  for (int i = 0; i < M_D.NumRows(); i++) {
    vector<gpuPivotInfo> empty_vec;
    rows_with_pivot.push_back(empty_vec);
  }
  uint32_t row_0_pivot_idx = 0;
  long row_0_pivot_degree = -1;

  for (uint32_t current_col = 0; current_col < M_D.NumCols(); current_col++) {
    long current_degree = deg(M_D[0][current_col]);

    if (current_degree >= row_0_pivot_degree) {
      row_0_pivot_degree = current_degree;
      row_0_pivot_idx = current_col;
    }
  }
  rows_with_pivot[row_0_pivot_idx].push_back({0, row_0_pivot_degree});
  for (uint32_t row_idx = 1; row_idx < M_D.NumRows(); row_idx++) {
    rows_with_pivot[row_idx].push_back({row_idx, x_coords.length()});
  }

  auto converted_M_D = mat_to_array3d(M_D);
  auto t2 = Clock::now();
  dur_convert_array = t2 - t1;

  weak_popov_only_gpu(converted_M_D, ell, row_0_pivot_idx, row_0_pivot_degree,
                      rows_with_pivot);
  auto t3 = Clock::now();
  dur_wpf = t3 - t2;

  auto converted_back = array3d_to_mat(converted_M_D);
  auto t4 = Clock::now();
  dur_convert_array_back = t4 - t3;

  cout << "---- First Step Timings ----" << endl;
  cout << "Make Basis:\t\t" << dur_mb.count() << "ms" << endl;
  cout << "Convert Array:\t\t" << dur_convert_array.count() << "ms" << endl;
  cout << "WPF:\t\t\t" << dur_wpf.count() << "ms" << endl;
  cout << "Convert Array 2:\t" << dur_convert_array_back.count() << "ms" << endl
       << endl;

  return converted_back;
}

vector<Vec<zz_pX>> list_decode(Vec<zz_p> &x_coords, Mat<zz_p> &y_coords,
                               unsigned short c, unsigned long ell,
                               unsigned long agreement,
                               bool stop_after_first_solution = false) {
  if (x_coords.length() < ell) {
    cout << "Cannot decode with " << x_coords.length()
         << " points when ell = " << ell << "." << endl;
    vector<Vec<zz_pX>> empty;
    return empty;
  }

  Duration dur_lgb, dur_ci, dur_alist_trans, dur_N_trans, dur_first_step,
      dur_extract_dealer;

  auto t0 = Clock::now();

  vector<uint32_t> gpu_x_coords;
  for (zz_p xc : x_coords) {
    gpu_x_coords.push_back((uint32_t)rep(xc));
  }
  LatticeParamsGPU params_gpu = lagrange_basis_gpu(gpu_x_coords, c, ell);
  auto t1 = Clock::now();
  dur_lgb = t1 - t0;

  // Make a giant vector (a_list) based on x and y coordinates
  Vec<zz_p> x_indics;
  for (size_t i = 0; i < x_coords.length(); i++) {
    x_indics.append(zz_p(1));
  }

  vector<uint32_t> gpu_y_coords(y_coords.NumRows() * y_coords.NumCols());
  for (int i = 0; i < y_coords.NumRows(); i++) {
    for (int j = 0; j < c; j++) {
      gpu_y_coords[i * c + j] = (uint32_t)rep(y_coords[i][j]);
    }
  }
  vector<uint32_t> x_indics_gpu;
  for (auto xi : x_indics) {
    x_indics_gpu.push_back((uint32_t)rep(xi));
  }
  vector<uint32_t> gpu_a_list =
      create_interpols_gpu(params_gpu.lags, params_gpu.w, gpu_y_coords,
                           x_indics_gpu, y_coords.NumRows(), c);
  auto t2 = Clock::now();
  dur_ci = t2 - t1;

  Vec<zz_pX> new_a_list;
  for (int cur_c = 0; cur_c < c; cur_c++) {
    zz_pX cur_poly;
    for (int deg = 0; deg < y_coords.NumRows(); deg++) {
      SetCoeff(cur_poly, deg, gpu_a_list[deg * c + cur_c]);
    }
    new_a_list.append(cur_poly);
  }
  auto t3 = Clock::now();
  dur_alist_trans = t3 - t2;

  zz_pX new_N;
  for (int deg = 0; deg <= y_coords.NumRows(); deg++) {
    SetCoeff(new_N, deg, params_gpu.N[deg]);
  }
  auto t4 = Clock::now();
  dur_N_trans = t4 - t3;

  // Vector to store output polynomials
  vector<Vec<zz_pX>> outputs;

  // Where all the decoding magic is done.
  // Follows a pattern of:
  //  1) Build and reduce a matrix
  //  2) Check the matrix for a specific structure, if present extract a
  //  solution and return 3) If not present, end decoding.
  bool clfs = true;
  while (clfs) {
    // Build the matrix and perform the lattice reduction
    Mat<zz_pX> A =
        first_step(x_coords, y_coords, new_a_list, x_indics, new_N, c, ell);
    auto t5 = Clock::now();
    dur_first_step = t5 - t4;

    // Check for structure
    long sv_len = find_sv_len(A);
    unsigned long ub_on_sol =
        std::count(x_indics.begin(), x_indics.end(), zz_p(1)) - agreement + ell;

    if (sv_len > ub_on_sol) {
      // No more solutions at all, exit.
      clfs = false;
      continue;
    }

    Vec<zz_pX> comb_svs = add_shortest_vectors(A);

    zz_pX divisor;
    SetCoeff(divisor, (long)ell, 1);

    comb_svs[0] = comb_svs[0] / divisor;

    if (a_divides_b(comb_svs[0], comb_svs[1])) {
      // Success! There's an easy solution here.

      // Copy over the polynomials
      Vec<zz_pX> resp_polys;
      resp_polys.SetLength(comb_svs.length() - 1);
      for (long i = 1; i < comb_svs.length(); i++) {
        resp_polys[i - 1] = comb_svs[i] / comb_svs[0];
      }

      outputs.push_back(resp_polys);

      // To do iterative decoding we allow an early exit.
      // In the original code it would continue until all solutions are found.
      // We need to exit early as we change the batch size/c on each decode
      // attempt.
      // TODO: Refactor so that both are easily doable.
      if (stop_after_first_solution) {
        clfs = false;
      } else {
        remove_points(x_coords, y_coords, resp_polys, x_indics);
      }
    } else if (sv_len == ub_on_sol) {
      clfs = false;
    } else {
      // Just ignore this for now. This is the "unhappy path" which happens
      // extremely rarely and costs a fair amount of time to go down.
      // TODO: Calculate the exact probability of hitting this path.
      cout << "Unhappy path :(" << endl;
      clfs = false;
    }

    auto t6 = Clock::now();
    dur_extract_dealer = t6 - t5;
  }
  // Duration dur_lgb, dur_ci, dur_alist_trans, dur_N_trans, dur_first_step,
  //     dur_extract_dealer;

  cout << "---- List Decode Timings ----" << endl;
  cout << "Lagrange Basis:\t\t" << dur_lgb.count() << "ms" << endl;
  cout << "Create Interpols:\t" << dur_ci.count() << "ms" << endl;
  cout << "a_list transform:\t" << dur_alist_trans.count() << "ms" << endl;
  cout << "N transform:\t\t" << dur_N_trans.count() << "ms" << endl;
  cout << "First Step:\t\t" << dur_first_step.count() << "ms" << endl;
  cout << "Extract Dealer:\t\t" << dur_extract_dealer.count() << "ms" << endl;
  cout << "--------" << endl;

  return outputs;
}

pair<vector<Vec<zz_pX>>, chrono::duration<double>> time_decode(
    Vec<zz_p> &x_coords, Mat<zz_p> &y_coords, unsigned short c,
    unsigned long ell, unsigned long agreement,
    bool stop_after_first_solution = false) {
  auto list_decode_start = std::chrono::high_resolution_clock::now();
  auto output_polys = list_decode(x_coords, y_coords, c, ell, agreement,
                                  stop_after_first_solution);
  auto list_decode_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = list_decode_end - list_decode_start;
  return make_pair(output_polys, duration);
}

pair<Vec<zz_p>, Mat<zz_p>> subsample(Vec<zz_p> x_coords, Mat<zz_p> y_coords,
                                     unsigned long new_n,
                                     unsigned short new_c) {
  long N = x_coords.length();

  if (new_n > N) {
    throw invalid_argument("new array cannot be longer than original array");
  }
  if (new_c > y_coords.NumCols()) {
    throw invalid_argument("cannot use a larger value for new_c");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::set<long> indices;
  while (indices.size() < new_n) {
    indices.insert(std::uniform_int_distribution<>(0, (int)N - 1)(gen));
  }

  Vec<zz_p> new_x_coords;
  Mat<zz_p> new_y_coords;

  new_x_coords.SetLength((long)new_n);
  new_y_coords.SetDims((long)new_n, new_c);

  long curr_idx = 0;
  for (long idx : indices) {
    new_x_coords[curr_idx] = x_coords[idx];

    for (long j = 0; j < new_c; j++) {
      new_y_coords[curr_idx][j] = y_coords[idx][j];
    }

    curr_idx++;
  }

  return make_pair(new_x_coords, new_y_coords);
}

optional<Vec<zz_pX>> sample_and_decode(unsigned long batch_size,
                                       unsigned short c, unsigned short ell,
                                       Vec<zz_p> &x_coords,
                                       Mat<zz_p> &y_coords) {
  cout << "Starting decode" << endl;
  auto current_agreement = min_t((float)batch_size, c, ell);
  auto new_coords = subsample(x_coords, y_coords, batch_size, c);
  auto result = time_decode(new_coords.first, new_coords.second, c, ell,
                            current_agreement, true);
  auto output = result.first;

  if (output.size() > 0) {
    cout << "Success!" << endl;
    cout << "Decoded in " << result.second.count() << " seconds" << endl;
    return output[0];
  }

  cout << "Failed to recover a dealer." << endl;
  cout << "Decoded in " << result.second.count() << " seconds" << endl;
  return nullopt;
}

int main(int argc, char *argv[]) {
  auto t0 = Clock::now();
  if (argc < 3) {
    std::cout << "Must provide a path to an instance file and a config file"
              << endl;
    return 1;
  }
  vector<string> args(argv + 1, argv + argc);

  auto o_instance = create_json();
  auto o_config = create_json();
  auto o_raw_present_dealers = create_json();
  if (!o_instance.has_value() || !o_config.has_value() ||
      !o_raw_present_dealers.has_value()) {
    cout << "Json initialization failed, quitting" << endl;
    return 1;
  }
  json instance = o_instance.value();
  json config_data = o_config.value();
  json raw_present_dealers = o_raw_present_dealers.value();

  try {
    instance = readJsonFile(args[0]);
  } catch (exception &e) {
    cout << "Error reading instance file: " << e.what() << endl;
    return 1;
  }

  InstanceParams params{};
  try {
    params = load_instance_parameters(instance);
  } catch (exception &e) {
    cout << "Error loading instance parameters: " << e.what() << endl;
    return 1;
  }

  cout << "Processing instance with n = " << params.n
       << ", ell = " << params.ell << ", and agreement = " << params.agreement
       << endl;

  // Initialize prime field
  // zz_p::init(params.field_size);
  zz_p::init(FIELD_SIZE);

  Codeword codeword{};
  try {
    codeword = load_codeword(instance, params.c);
  } catch (exception &e) {
    cout << "Error loading codeword: " << e.what() << endl;
    return 1;
  }
  auto x_coords = codeword.x_coords;
  auto y_coords = codeword.y_coords;

  // Read in config and compute agreements
  try {
    config_data = readJsonFile(args[1]);
  } catch (exception &e) {
    cout << "Error reading config file: " << e.what() << endl;
    return 1;
  }
  Config config;
  try {
    config = load_config(config_data);
  } catch (exception &e) {
    cout << "Error loading config file: " << e.what() << endl;
    return 1;
  }

  auto t1 = Clock::now();
  Duration load_files = t1 - t0;
  cout << "Time to load files: " << load_files.count() << "ms" << endl;

  // We iteratively decode until no solution is found, at which point we end.
  // bool found_solution = true;

  // Vector to store all the found polynomials.
  vector<NTL::vec_zz_pX> output_polys;

  auto total_start = std::chrono::high_resolution_clock::now();

  int current_index = 0;
  vector<json> decode_results;
  while (true) {
    json decode_result;

    auto current_start = std::chrono::high_resolution_clock::now();
    cout << endl
         << "Decoding with configuration at index: " << current_index + 1
         << endl;

    auto batch_size = config.batch_sizes[current_index];
    auto current_c = config.c_vals[current_index];

    auto ss_batch_size = batch_size;
    if (ss_batch_size > x_coords.length()) {
      ss_batch_size = x_coords.length();
      cout << "Cannot subsample " << batch_size << " from " << ss_batch_size
           << " points. Using " << ss_batch_size << " points instead." << endl;
    }

    cout << "Using c = " << current_c << ", and batch_size = " << ss_batch_size
         << endl;

    decode_result["batch_size"] = ss_batch_size;
    decode_result["c"] = current_c;
    decode_result["field_size"] = params.field_size;

    optional<Vec<zz_pX>> output;
    try {
      output = sample_and_decode(ss_batch_size, current_c, params.ell, x_coords,
                                 y_coords);
    } catch (const exception &e) {
      cout << "Error while decoding: " << e.what() << std::endl;
      cout << "Quitting..." << endl;
      decode_result["status"] = "error";
      decode_result["error"] = e.what();
      decode_results.push_back(decode_result);
      break;
    }

    auto current_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> current_duration =
        current_end - current_start;
    decode_result["duration"] = current_duration.count();

    if (output.has_value() > 0)  // We found a polynomial set!
    {
      cout << "Success! ";
      auto output_poly_set = output.value();
      cout << "Adjusting points and moving to next rank" << endl;

      // Remove all points that agree with the polynomial set
      auto new_points = remove_points_2(x_coords, y_coords, output_poly_set);

      // Exit if we discovered an insufficient dealer.
      auto difference = x_coords.length() - new_points.first.length();
      if (difference < params.agreement) {
        cout << "Recovered dealer has " << difference << " < "
             << params.agreement << " points, quitting." << endl;
        decode_result["status"] = "insufficient_dealer";
        decode_results.push_back(decode_result);
        break;
      }
      decode_result["status"] = "success";
      decode_results.push_back(decode_result);

      // Otherwise, add the found poly set to the output and continue
      output_polys.push_back(output_poly_set);
      x_coords = new_points.first;
      y_coords = new_points.second;
    } else {
      cout << "Failed to recover rank, ";
      decode_result["status"] = "failure";
      decode_results.push_back(decode_result);  // TODO: Refactor and DRY out
      if (config.continue_on_failure) {
        if (current_index == config.batch_sizes.size() - 1) {
          cout << "reached final index, quitting." << endl;
          break;
        }
        cout << "continiuing anyway" << endl;
      } else {
        cout << "quitting." << endl;
        break;
      }
    }

    if (current_index < config.batch_sizes.size() - 1) {
      current_index++;
    } else if (config.terminate_after_last_config) {
      cout << "Finished with last config, quitting." << endl;
      break;
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = total_end - total_start;
  std::cout << endl
            << "Total time spent decoding: " << total_duration.count() << endl;

  cout << "Output len: " << output_polys.size() << endl;

  // Check if our output was correct
  map<int, Vec<zz_p>> present_values;

  try {
    raw_present_dealers = instance["present_dealers"];
  } catch (exception &e) {
    cout << "Error when loading present dealers: " << e.what() << endl;
  }

  try {
    for (const auto &[key, stalker_poly_set] : raw_present_dealers.items()) {
      // The instance file contains a list of the correct polynomials, i.e.
      // all those that are at least t-present in the full instance. We deduce
      // the sent "value" of each dealer, i.e. the concatenation of the
      // constant terms of each polynomial.

      Vec<zz_p> value;
      for (const auto &coefs : stalker_poly_set) {
        value.append(zz_p(coefs[0]));
      }
      present_values[stoi(key)] = value;
    }
  } catch (exception &e) {
    cout << "Error processing present dealers: " << e.what() << endl;
    return 1;
  }

  Vec<Vec<zz_p>> recovered_values;
  int counter = 0;
  for (const auto &dealer : output_polys) {
    cout << "Recovering value for dealer: " << counter + 1 << endl;
    auto value =
        get_dealer_value(params, dealer, codeword.x_coords, codeword.y_coords);
    recovered_values.append(value);
    counter++;
  }

  // Loop over each present dealer to check if we recovered it.
  bool missed_dealer = false;
  vector<int> missed_ranks;
  std::set<int> matched_indexes;
  for (const auto &[key, dealer_value] : present_values) {
    bool recovered_dealer = false;
    for (int i = 0; i < recovered_values.length(); i++) {
      auto rv = recovered_values[i];
      if (rv == dealer_value) {
        recovered_dealer = true;
        matched_indexes.insert(i);
        break;
      }
    }
    if (!recovered_dealer) {
      cout << "Failed to recover value at rank " << key << "." << endl;
      missed_dealer = true;
      missed_ranks.push_back(key);
    } else {
      cout << "Successfully recovered value at rank " << key << "." << endl;
    }
  }

  if (missed_dealer) {
    cout << "Did not recover all values." << endl;
  } else {
    cout << "Successfully recovered all present values!" << endl;
  }

  // Loop over each output value to see if it corresponds to a present
  // dealer.
  bool found_extra_dealer = false;
  vector<vector<long>> extra_values;
  for (int i = 0; i < recovered_values.length(); i++) {
    if (matched_indexes.count(i) < 1) {
      found_extra_dealer = true;
      cout << endl
           << "Output value at position " << i << " was not in present_values"
           << endl;
      cout << "Printing value: " << endl;
      cout << "[";

      vector<long> extra_val;
      for (int j = 0; j < recovered_values[i].length(); j++) {
        long rvj = 0;
        conv(rvj, recovered_values[i][j]);
        extra_val.push_back(rvj);
        cout << recovered_values[i][j];
        if (j < recovered_values[i].length() - 1) {
          cout << ", ";
        }
      }
      extra_values.push_back(extra_val);
      cout << "]" << endl;
    }
  }

  if (argc == 4) {
    // An output filename was passed
    json output_data;
    output_data["total_decode_time"] = total_duration.count();
    output_data["decode_results"] = decode_results;
    output_data["missed_dealer"] = missed_dealer;
    output_data["found_extra_dealer"] = found_extra_dealer;
    output_data["succeeded"] = !missed_dealer && !found_extra_dealer;

    if (missed_dealer) {
      output_data["missed_ranks"] = missed_ranks;
    }

    if (found_extra_dealer) {
      output_data["extra_dealers"] = extra_values;
    }

    ofstream file(args[2]);
    if (file.is_open()) {
      file << output_data.dump(4);
      file.close();
    }
  }

  return 0;
}
