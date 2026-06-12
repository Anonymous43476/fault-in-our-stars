
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>
#include <NTL/lzz_pXFactoring.h>
#include <NTL/matrix.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace NTL;

void dump_ntl_matrix_binary(const Mat<zz_pX> &mat, size_t max_degree,
                            const std::string &filename) {
  ofstream out(filename, ios::binary);
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

  for (long i = 0; i < rows; i++) {
    for (long j = 0; j < cols; j++) {
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

std::string pretty_print(const zz_pX &poly) {
  if (IsZero(poly)) return "0";

  std::ostringstream oss;

  bool printed_something = false;
  for (long i = deg(poly); i >= 0; i--) {
    zz_p coeff = poly[i];
    if (IsZero(coeff)) continue;

    stringstream ss;
    ss << coeff;
    string coeff_str = ss.str();

    if (printed_something) {
      oss << " + ";
    }

    if (i > 0) {
      if (coeff_str != "1") {
        oss << coeff_str;
      }
      oss << "x"
          << "^" << i;
    } else {
      oss << coeff_str;
    }

    printed_something = true;
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

bool point_in(zz_p point, Vec<zz_p> vec) {
  auto it = find(vec.begin(), vec.end(), point);

  if (it != vec.end()) {
    return true;
  }

  return false;
}

Vec<zz_pX> zero_vec(long len) {
  Vec<zz_pX> v;
  for (long i = 0; i < len; i++) {
    zz_pX p;
    v.append(p);
  }
  return v;
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
