#include <cmath>
#include <iostream>
#include <utility>  // pair
#include <vector>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;
typedef pair<Vec, double> Cut;

class LowpassOracle {
  private:
    bool more_alt = true;
    int i_Anr = 0;
    int i_As = 0;
    int i_Ap = 0;
    int count = 0;
    Mat Ap;
    Mat As;
    Mat Anr;
    double Lpsq;
    double Upsq;

  public:
    LowpassOracle(Mat Ap, Mat As, Mat Anr, double Lpsq, double Upsq) {
        this->Ap = Ap;
        this->As = As;
        this->Anr = Anr;
        this->Lpsq = Lpsq;
        this->Upsq = Upsq;
    }

    pair<Vec, pair<double, double>> assess_optim(Vec x, double Spsq) {
        int n = x.size();
        more_alt = true;

        // case 2, passband constraints
        int N = Ap.size();
        int n_Ap = Ap[0].size();
        for (int k = i_Ap; k < N; k++) {
            double v = 0;
            for (int i = 0; i < n_Ap; i++) {
                v += Ap[k][i] * x[i];
            }
            if (v > Upsq) {
                Vec g = Ap[k];
                pair<double, double> f = make_pair(v - Upsq, v - Lpsq);
                i_Ap = k + 1;
                return make_pair(g, f);
            }
            if (v < Lpsq) {
                Vec g = Ap[k];
                pair<double, double> f = make_pair(-v + Lpsq, -v + Upsq);
                i_Ap = k + 1;
                return make_pair(g, f);
            }
        }

        // case 3, stopband constraint
        N = As.size();
        int n_As = As[0].size();
        double fmax = -INFINITY;
        int imax = 0;
        for (int k = i_As; k < N; k++) {
            double v = 0;
            for (int i = 0; i < n_As; i++) {
                v += As[k][i] * x[i];
            }
            if (v > Spsq) {
                Vec g = As[k];
                pair<double, double> f = make_pair(v - Spsq, v);
                i_As = k + 1;
                return make_pair(g, f);
            }
            if (v < 0) {
                Vec g = As[k];
                pair<double, double> f = make_pair(-v, -v + Spsq);
                i_As = k + 1;
                return make_pair(g, f);
            }
            if (v > fmax) {
                fmax = v;
                imax = k;
            }
        }

        // case 4, nonnegative-real constraint on other frequencies
        // N = Anr.size();
        // int n_Anr = Anr[0].size();
        // for (int k = i_Anr; k < N; k++) {
        //     double v = 0;
        //     for (int i = 0; i < n_Anr; i++) {
        //         v += Anr[k][i] * x[i];
        //     }
        //     if (v < 0) {
        //         double f = -v;
        //         Vec g = Anr[k];
        //         i_Anr = k + 1;
        //         return make_pair(g, f);
        //     }
        // }

        // more_alt = false;

        // // case 1 (unlikely)
        // if (x[0] < 0) {
        //     Vec g(n, 0.0);
        //     g[0] = -1.0;
        //     double f = -x[0];
        //     return make_pair(g, f);
        // }

        // Begin objective function
        Spsq = fmax;
        pair<double, double> f = make_pair(0.0, fmax);
        Vec g = As[imax];
        return make_pair(g, f);
    }
};

int main() {
    // Example usage
    Mat Ap = {{1, 2, 3}, {4, 5, 6}};
    Mat As = {{7, 8, 9}, {10, 11, 12}};
    Mat Anr = {{13, 14, 15}, {16, 17, 18}};
    double Lpsq = 0.5;
    double Upsq = 1.5;

    LowpassOracle oracle(Ap, As, Anr, Lpsq, Upsq);

    Vec x = {0.1, 0.2, 0.3};
    double Spsq = 0.8;

    pair<Vec, pair<double, double>> result = oracle.assess_optim(x, Spsq);

    cout << "g: ";
    for (double val : result.first) {
        cout << val << " ";
    }
    cout << endl;

    cout << "f: " << result.second.first << " " << result.second.second << endl;

    return 0;
}
