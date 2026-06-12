#pragma once

#include <NTL/lzz_pX.h>

#include <cassert>
#include <cstddef>

using namespace std;
using namespace NTL;

template <typename T>
class Array3D {
 private:
  long x_, y_, z_;
  T *data_;

 public:
  Array3D(long x, long y, long z)
      : x_(x), y_(y), z_(z), data_(new T[x * y * z]) {}

  ~Array3D() { delete[] data_; }

  // disable copy (to avoid accidental expensive copies)
  Array3D(const Array3D &) = delete;
  Array3D &operator=(const Array3D &) = delete;

  // enable move
  Array3D(Array3D &&other) noexcept
      : x_(other.x_), y_(other.y_), z_(other.z_), data_(other.data_) {
    other.data_ = nullptr;
  }

  Array3D &operator=(Array3D &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      x_ = other.x_;
      y_ = other.y_;
      z_ = other.z_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  // access operator
  inline T &operator()(size_t i, size_t j, size_t k) {
    assert(i < x_ && j < y_ && k < z_);
    return data_[i * (y_ * z_) + j * z_ + k];
  }

  inline const T &operator()(size_t i, size_t j, size_t k) const {
    assert(i < x_ && j < y_ && k < z_);
    return data_[i * (y_ * z_) + j * z_ + k];
  }

  // raw access if needed
  T *data() { return data_; }
  const T *data() const { return data_; }

  long dim_x() const { return x_; }
  long dim_y() const { return y_; }
  long dim_z() const { return z_; }
};

Array3D<uint32_t> mat_to_array3d(const Mat<zz_pX> &M);

Mat<zz_pX> array3d_to_mat(const Array3D<uint32_t> &A);

void print_array_degrees(const Array3D<uint32_t> &arr);
