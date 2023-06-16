#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

/**
 * @brief 
 * 
 */
class LDLTMgr {
private:
  std::pair<int, int> p;
  std::vector<double> v;
  int _n;
  std::vector<std::vector<double>> _T;

public:
  /**
   * @brief Construct a new LDLTMgr object
   * 
   * @param N 
   */
  LDLTMgr(int N) {
    p = std::make_pair(0, 0);
    v = std::vector<double>(N, 0);
    _n = N;
    _T = std::vector<std::vector<double>>(N, std::vector<double>(N, 0));
  }

  /**
   * @brief 
   * 
   * @param A 
   * @return true 
   * @return false 
   */
  bool factorize(std::vector<std::vector<double>> A) {
    return factor([&A](int i, int j) { return A[i][j]; });
  }

  /**
   * @brief 
   * 
   * @param get_elem 
   * @return true 
   * @return false 
   */
  bool factor(std::function<double(int, int)> get_elem) {
    int start = 0;
    p = std::make_pair(0, 0);
    for (int i = 0; i < _n; i++) {
      double d = get_elem(i, start);
      for (int j = start; j < i; j++) {
        _T[j][i] = d;
        _T[i][j] = d / _T[j][j];
        int s = j + 1;
        d = get_elem(i, s) - std::inner_product(_T[i].begin() + start,
                                                _T[i].begin() + s,
                                                _T[j].begin() + start, 0.0);
      }
      _T[i][i] = d;
      if (d <= 0.0) {
        p = std::make_pair(start, i + 1);
        break;
      }
    }
    return is_spd();
  }

  /**
   * @brief 
   * 
   * @return true 
   * @return false 
   */
  bool is_spd() { return p.second == 0; }

  /**
   * @brief 
   * 
   * @return double 
   */
  double witness() {
    if (is_spd()) {
      throw std::runtime_error("Matrix is SPD");
    }
    int start = p.first, n = p.second;
    int m = n - 1;
    v[m] = 1.0;
    for (int i = m; i > start; i--) {
      v[i - 1] = -std::inner_product(_T[i].begin() + i, _T[n].begin() + i,
                                     v.begin() + i, 0.0);
    }
    return -_T[m][m];
  }

  /**
   * @brief 
   * 
   * @param A 
   * @return double 
   */
  double sym_quad(std::vector<std::vector<double>> A) {
    int s = p.first, n = p.second;
    std::vector<double> v =
        std::vector<double>(this->v.begin() + s, this->v.begin() + n);
    double result = 0.0;
    for (int i = s; i < n; i++) {
      double temp = 0.0;
      for (int j = s; j < n; j++) {
        temp += A[i][j] * v[j - s];
      }
      result += v[i - s] * temp;
    }
    return result;
  }
};
