#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>
#include <NTL/lzz_pXFactoring.h>
#include <NTL/matrix.h>

#include <algorithm>
#include <chrono>
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

using namespace std;
using namespace NTL;

using json = nlohmann::json;

using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double, milli>;

void dump_ntl_matrix_binary(const Mat<zz_pX> &mat, size_t max_degree,
                            const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open binary file for writing: " +
                             filename);
  }

  size_t rows = mat.NumRows();
  size_t cols = mat.NumCols();

  // Write Header
  out.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));
  out.write(reinterpret_cast<const char *>(&max_degree), sizeof(size_t));

  std::vector<int> buffer(cols * max_degree);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      const zz_pX &poly = mat[i][j];
      for (size_t k = 0; k < max_degree; k++) {
        // coeff() returns 0 if k > deg(poly)
        // rep(zz_p) returns the underlying long
        buffer[j * max_degree + k] = static_cast<int>(rep(coeff(poly, k)));
      }
    }
    out.write(reinterpret_cast<const char *>(buffer.data()),
              buffer.size() * sizeof(int));
  }
}

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

Vec<zz_pX> zero_vec(long len) {
  Vec<zz_pX> v;
  for (long i = 0; i < len; i++) {
    zz_pX p;
    v.append(p);
  }
  return v;
}

void pretty_print_vec(const vector<unsigned long> vec) {
  cout << "[";
  for (auto &elem : vec) {
    cout << elem << ", ";
  }
  cout << "]" << endl;
}

void pretty_print_pair_vec(const vector<vector<pair<long, long>>> vec) {
  cout << "[";
  for (auto &sub_vec : vec) {
    cout << "[";
    for (auto &pair : sub_vec) {
      cout << "(" << pair.first << ", " << pair.second << "), ";
    }
    cout << "], ";
  }
  cout << "]" << endl;
}

std::string pretty_print(const zz_pX &poly) {
  if (IsZero(poly)) return "0";

  std::ostringstream oss;

  for (long i = deg(poly); i >= 0; i--) {
    zz_p coeff = poly[i];
    if (IsZero(coeff)) continue;

    stringstream ss;
    ss << coeff;
    string coeff_str = ss.str();

    if (coeff_str != "1") {
      oss << coeff_str;
    }

    if (i > 0) {
      oss << "x"
          << "^" << i << " + ";
    }
  }

  return oss.str();
}

void pretty_print_matrix(const Mat<zz_p> &mat) {
  for (long i = 0; i < mat.NumRows(); i++) {
    for (long j = 0; j < mat.NumCols(); j++) {
      cout << mat[i][j] << " ";
    }
    cout << endl;
  }
}

void pretty_print_matrix(const Mat<zz_pX> &mat) {
  for (long i = 0; i < mat.NumRows(); i++) {
    cout << "\nRow " << i << endl;
    for (long j = 0; j < mat.NumCols(); j++) {
      cout << pretty_print(mat[i][j]) << endl;
    }
  }
}

void print_matrix_degrees(const Mat<zz_pX> &mat) {
  for (long i = 0; i < mat.NumRows(); i++) {
    for (long j = 0; j < mat.NumCols(); j++) {
      cout << deg(mat[i][j]) << "\t";
    }
    cout << endl;
  }
}

template <typename T>
T pop_set(std::set<T> &s) {
  if (s.empty()) {
    throw std::out_of_range("Set is empty");
  }

  T smallest = *s.begin();
  s.erase(s.begin());
  return smallest;
}

bool point_in(zz_p point, Vec<zz_p> vec) {
  auto it = find(vec.begin(), vec.end(), point);

  if (it != vec.end()) {
    return true;
  }

  return false;
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

void add_to_vec(Vec<zz_pX> &target, const Vec<zz_pX> &to_add) {
  for (long i = 0; i < target.length(); i++) {
    add(target[i], target[i], to_add[i]);
  }
}

bool a_divides_b(const zz_pX &a, const zz_pX &b) {
  // Returns true if a divides b

  if (IsZero(a)) {
    throw 20;
  }

  zz_pX quotient, remainder;
  DivRem(quotient, remainder, b, a);
  return IsZero(remainder);
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
      cout << "Adding SV at row: " << itr << endl;
      add_to_vec(add_all_shortest_vecs, itr_vec);
    }
  }

  return add_all_shortest_vecs;
}

size_t get_polynomial_size(const zz_pX &poly) {
  return (deg(poly) + 1) * sizeof(zz_p);
}

size_t get_matrix_size(const Mat<zz_pX> &matrix) {
  size_t total_size = 0;

  for (long i = 0; i < matrix.NumRows(); i++) {
    for (long j = 0; j < matrix.NumCols(); j++) {
      total_size += get_polynomial_size(matrix[i][j]);
    }
  }

  return total_size;
}

Vec<zz_pX> mul_row(Vec<zz_pX> &row, zz_pX &val) {
  for (long i = 0; i < row.length(); i++) {
    row[i] *= val;
  }
  return row;
}

void add_to_row(Mat<zz_pX> &matrix, long target, Vec<zz_pX> &val) {
  for (long i = 0; i < matrix.NumCols(); i++) {
    matrix[target][i] += val[i];
  }
}

void add_multiple_of_row(Mat<zz_pX> &matrix, long target, long source,
                         zz_pX &val) {
  zz_pContext context;
  context.save();

  NTL_EXEC_RANGE(matrix.NumCols(), first, last)
  context.restore();

  zz_pX temp;
  for (long i = first; i < last; i++) {
    mul(temp, matrix[source][i], val);
    matrix[target][i] += temp;
  }
  NTL_EXEC_RANGE_END
}

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

Mat<zz_pX> make_basis(LatticeParams &params, Vec<zz_pX> &lagr_polys,
                      Vec<zz_p> &x_coords, Vec<zz_p> &locs) {
  Mat<zz_pX> M_D;
  M_D.SetDims(params.c + 1, params.c + 1);

  zz_pX base_N = params.N;

  for (long i = 0; i < locs.length(); i++) {
    if (locs[i] == 0) {
      Vec<zz_p> single_root;
      single_root.append(x_coords[i]);
      zz_pX divisor = BuildFromRoots(single_root);
      base_N = base_N / divisor;
    }
  }

  for (int i = 0; i < params.c + 1; i++) {
    for (int j = 0; j < params.c + 1; j++) {
      if (i == 0) {
        if (j == 0) {
          zz_pX zero_zero;
          SetCoeff(zero_zero, (long)params.ell, 1);
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

void weak_popov_form(Mat<zz_pX> &matrix) {
  // cout << "\n MAT:\n";
  // pretty_print_matrix(matrix);
  // cout << endl;

  unsigned long m = matrix.NumRows();
  unsigned long n = matrix.NumCols();

  vector<vector<pair<long, long>>> to_row;
  vector<long> conflicts;

  for (size_t c = 0; c < n; c++) {
    vector<pair<long, long>> vec;
    to_row.push_back(vec);
  }

  Duration timeSetup{}, timeATR{}, timePivotComp{};
  for (long i = 0; i < m; i++) {
    long bestp = -1;
    long best = -1;

    // Find pivot index for this row
    for (long c = 0; c < n; c++) {
      const zz_pX &current = matrix[i][c];
      const long d = deg(current);

      if (d >= best) {
        bestp = c;
        best = d;
      }
    }

    if (best >= 0) {  // Check if a non-zero row
      to_row[bestp].push_back(make_pair(i, best));
      if (to_row[bestp].size() > 1) {
        conflicts.push_back(bestp);
      }
    }
  }

  int expected_steps = (deg(matrix[0][1]) - deg(matrix[0][0])) * (m - 1);
  int step_counter = 0;

  auto last_print = Clock::now();
  auto loop_start = Clock::now();
  while (conflicts.size() > 0) {
    // cout << "---" << endl;
    // print_matrix_degrees(matrix);
    // if (step_counter > 50) break;
    auto t0 = Clock::now();
    step_counter++;

    if (t0 - last_print > std::chrono::seconds(30)) {
      double elapsed = std::chrono::duration<double>(t0 - loop_start).count();
      double rate = (step_counter + 1) / elapsed;
      double eta = (expected_steps - step_counter - 1) / rate;

      int eta_h = static_cast<int>(eta) / 3600;
      int eta_m = (static_cast<int>(eta) % 3600) / 60;
      int eta_s = static_cast<int>(eta) % 60;

      cout << "\rStep " << step_counter << " / " << expected_steps << " -- "
           << "Expected time remaining: " << std::setw(2) << std::setfill('0')
           << eta_h << ":" << std::setw(2) << eta_m << ":" << std::setw(2)
           << eta_s << std::flush;

      last_print = t0;
    }
    auto c = conflicts.back();
    conflicts.pop_back();

    auto &row = to_row[c];

    auto i_pair = row.back();
    row.pop_back();
    auto j_pair = row.back();
    row.pop_back();

    if (j_pair.second > i_pair.second) {
      swap(i_pair, j_pair);
    }
    long i = i_pair.first;
    long ideg = i_pair.second;
    long j = j_pair.first;
    long jdeg = j_pair.second;

    zz_pX &num_poly = matrix[i][c];
    zz_pX &denom_poly = matrix[j][c];
    auto coef = -1 * (LeadCoeff(num_poly) / LeadCoeff(denom_poly));

    zz_pX s;
    NTL::set(s);
    s <<= ideg - jdeg;
    s *= coef;

    // cout << "Shift: " << ideg - jdeg << endl;
    cout << "Target: " << i << ", Source " << j << endl;

    auto t1 = Clock::now();
    add_multiple_of_row(matrix, i, j, s);
    auto t2 = Clock::now();

    // print_matrix_degrees(matrix);

    row.push_back(make_pair(j, jdeg));

    long bestp = -1;
    long best = -1;

    for (long c = 0; c < n; c++) {
      zz_pX &current = matrix[i][c];
      auto d = deg(current);

      if (d >= best) {
        bestp = c;
        best = d;
      }
    }

    if (best >= 0) {
      to_row[bestp].push_back(make_pair(i, best));
      if (to_row[bestp].size() > 1) {
        conflicts.push_back(bestp);
      }
    }
    auto t3 = Clock::now();

    timeSetup += t1 - t0;
    timeATR += t2 - t1;
    timePivotComp += t3 - t2;
  }
  cout << endl << "Setup Stage: " << timeSetup.count() << "ms" << endl;
  cout << "ATR Stage: " << timeATR.count() << "ms" << endl;
  cout << "Pivot Stage TOTAL: " << timePivotComp.count() << "ms" << endl;

  cout << "Took: " << step_counter << " steps." << endl;
}

Mat<zz_pX> first_step(Vec<zz_p> &x_coords, Mat<zz_p> &y_coords,
                      Vec<zz_pX> &a_list, Vec<zz_p> &x_indics,
                      LatticeParams params) {
  long one_count = std::count(x_indics.begin(), x_indics.end(), zz_p(1));
  Vec<zz_pX> lagr_polys;
  if (one_count == x_coords.length()) {
    lagr_polys = a_list;
  } else {
    lagr_polys = create_interpols(y_coords, x_indics, params);
  }

  auto mb_start = std::chrono::high_resolution_clock::now();
  auto M_D = make_basis(params, a_list, x_coords, x_indics);
  auto mb_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> mb_duration = mb_end - mb_start;
  printf("Make basis took %f seconds\n", mb_duration.count());

  auto wpf_start = std::chrono::high_resolution_clock::now();
  weak_popov_form(M_D);
  auto wpf_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> wpf_duration = wpf_end - wpf_start;
  printf("Weak Popov Form took %f seconds\n", wpf_duration.count());

  // pretty_print_matrix(M_D);
  return M_D;
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

  auto lgb_start = std::chrono::high_resolution_clock::now();
  // Make the lagrange basis
  LatticeParams params = lagrange_basis(x_coords, c, ell);
  auto lgb_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> lgb_duration = lgb_end - lgb_start;
  printf("Lagrange basis took %f seconds\n", lgb_duration.count());

  // Make a giant vector (a_list) based on x and y coordinates
  Vec<zz_p> x_indics;
  for (size_t i = 0; i < x_coords.length(); i++) {
    x_indics.append(zz_p(1));
  }
  auto start = chrono::high_resolution_clock::now();
  Vec<zz_pX> a_list = create_interpols(y_coords, x_indics, params);
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> dur = end - start;
  cout << "Interpols took " << dur.count() << "s" << endl;

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
    Mat<zz_pX> A = first_step(x_coords, y_coords, a_list, x_indics, params);

    start = chrono::high_resolution_clock::now();
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
      end = chrono::high_resolution_clock::now();
      dur = end - start;
      cout << "Extracting dealer took " << dur.count() << "s" << endl;

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
      continue;
    } else {
      // Just ignore this for now. This is the "unhappy path" which happens
      // extremely rarely and costs a fair amount of time to go down.
      // TODO: Calculate the exact probability of hitting this path.
      cout << "Unhappy path :(" << endl;
      clfs = false;
      continue;
    }
  }

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

optional<json> create_json() {
  try {
    return json{};
  } catch (exception &e) {
    cout << "Failed to initialize json object: " << e.what() << endl;
    return nullopt;
  }
}

int main(int argc, char *argv[]) {
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
  zz_p::init(100003);

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

  cout << "Parallelizing matrix operations over " << config.num_threads
       << " thread";
  if (config.num_threads > 1) {
    cout << "s";
  }
  cout << "." << endl;
  NTL::SetNumThreads(config.num_threads);

  // We iteratively decode until no solution is found, at which point we end.
  bool found_solution = true;

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
      // The instance file contains a list of the correct polynomials, i.e. all
      // those that are at least t-present in the full instance.
      // We deduce the sent "value" of each dealer, i.e. the concatenation of
      // the constant terms of each polynomial.

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
