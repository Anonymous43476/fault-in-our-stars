#pragma once

#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

template <typename T>
const T pop_set(std::set<T>& s) {
  if (s.empty()) {
    throw std::out_of_range("Set is empty");
  }

  T smallest = *s.begin();
  s.erase(s.begin());
  return smallest;
}

void pretty_print_vec(const vector<unsigned long> vec);

void pretty_print_pair_vec(const vector<vector<pair<long, long>>> vec);

template <typename T>
std::string to_string_custom(const T& val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

template <typename T>
void print_table(const std::vector<std::vector<T>>& table) {
  if (table.empty()) return;

  size_t cols = table[0].size();
  std::vector<size_t> col_widths(cols, 0);

  // Compute max width per column
  for (const auto& row : table) {
    for (size_t j = 0; j < cols; ++j) {
      std::string s = to_string_custom(row[j]);
      col_widths[j] = std::max(col_widths[j], s.size());
    }
  }

  // Print rows
  for (const auto& row : table) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << std::setw(col_widths[j] + 2) << to_string_custom(row[j]);
    }
    std::cout << "\n";
  }
}
