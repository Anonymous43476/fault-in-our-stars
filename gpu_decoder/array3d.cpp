#include "array3d.h"

#include <NTL/lzz_pX.h>
#include <NTL/matrix.h>
#include <NTL/sp_arith.h>

#include <vector>

#include "utils.h"

using namespace std;

Array3D<uint32_t> mat_to_array3d(const Mat<zz_pX> &M) {
  long x = M.NumRows();
  long y = M.NumCols();

  // Find maximum degree across all polynomials
  long max_deg = -1;
  for (long i = 0; i < x; ++i) {
    for (long j = 0; j < y; ++j) {
      long d = deg(M[i][j]);
      if (d > max_deg) max_deg = d;
    }
  }

  long z = (max_deg >= 0) ? (max_deg + 1) : 1;

  Array3D<uint32_t> result(x, y, z);

  // Fill with coefficients (zero-padded automatically via clear)
  for (long i = 0; i < x; ++i) {
    for (long j = 0; j < y; ++j) {
      const zz_pX &poly = M[i][j];
      long d = deg(poly);

      for (long k = 0; k < z; ++k) {
        if (k <= d) {
          result(i, j, k) = rep(coeff(poly, k));
        } else {
          result(i, j, k) = 0;  // set to 0 in zz_p
        }
      }
    }
  }

  return result;  // move-constructed (no copy)
}

Mat<zz_pX> array3d_to_mat(const Array3D<uint32_t> &A) {
  long x = A.dim_x();
  long y = A.dim_y();
  long z = A.dim_z();

  Mat<zz_pX> M;
  M.SetDims(x, y);

  for (long i = 0; i < x; ++i) {
    for (long j = 0; j < y; ++j) {
      zz_pX poly;
      clear(poly);  // ensure starting from 0

      // rebuild polynomial from coefficients
      for (long k = 0; k < z; ++k) {
        const zz_p c(A(i, j, k));
        if (!IsZero(c)) {
          SetCoeff(poly, k, c);
        }
      }

      // optional: normalize representation (removes trailing zeros internally)
      M[i][j] = std::move(poly);
    }
  }

  return M;  // NRVO / move
}

void print_array_degrees(const Array3D<uint32_t> &arr) {
  long x = arr.dim_x();
  long y = arr.dim_y();
  long z = arr.dim_z();

  vector<vector<long>> rows;
  for (int i = 0; i < x; i++) {
    vector<long> row;

    for (int j = 0; j < y; j++) {
      bool found_one = false;

      for (long k = z - 1; k >= 0; k--) {
        uint32_t val = arr(i, j, k);
        if (val > 0) {
          found_one = true;
          row.push_back(k);
          break;
        }
      }

      if (!found_one) {
        row.push_back(-1);
      }
    }

    rows.push_back(row);
  }

  print_table(rows);
}
