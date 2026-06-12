#pragma once

#include <NTL/lzz_pX.h>

using namespace std;
using namespace NTL;

void dump_ntl_matrix_binary(const Mat<zz_pX> &mat, size_t max_degree,
                            const std::string &filename);

string pretty_print(const zz_pX &poly);

void pretty_print_matrix(const Mat<zz_p> &mat);

void pretty_print_matrix(const Mat<zz_pX> &mat);

bool point_in(zz_p point, Vec<zz_p> vec);

Vec<zz_pX> zero_vec(long len);

void add_to_vec(Vec<zz_pX> &target, const Vec<zz_pX> &to_add);

bool a_divides_b(const zz_pX &a, const zz_pX &b);

size_t get_polynomial_size(const zz_pX &poly);

size_t get_matrix_size(const Mat<zz_pX> &matrix);

Vec<zz_pX> mul_row(Vec<zz_pX> &row, zz_pX &val);

void add_to_row(Mat<zz_pX> &matrix, long target, Vec<zz_pX> &val);

void add_multiple_of_row(Mat<zz_pX> &matrix, long target, long source);
