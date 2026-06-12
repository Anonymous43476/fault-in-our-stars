#include <iostream>
#include <vector>

using namespace std;

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
