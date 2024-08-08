#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <fstream>  // Librería para manejar archivos

using namespace std;

vector<double> generate_normal_distribution(double mu, double sigma, int N) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(mu, sigma);

    vector<double> dseta(N);
    for (int i = 0; i < N; ++i) {
        dseta[i] = dist(gen);
    }
    return dseta;
}

void save_matrix_to_file(const vector<vector<double>>& matrix, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "No se pudo abrir el archivo " << filename << endl;
        return;
    }

    for (const auto& row : matrix) {
        for (const auto& val : row) {
            file << val << " ";
        }
        file << endl;
    }

    file.close();
}

vector<vector<double>> _evolve_cooperation(vector<double> x, vector<double> a, double mu = 1, double sigma = 0.1, int steps = 1000) {
    assert(x.size() == a.size() && "Número de agentes inconsistente");

    int N = x.size();
    vector<vector<double>> X(steps + 1, vector<double>(N));
    X[0] = x;

    for (int i = 1; i <= steps; ++i) {
        vector<double> dseta = generate_normal_distribution(mu, sigma, N);
        double mean_a_x_dseta = inner_product(a.begin(), a.end(), dseta.begin(), 0.0) / N;

        for (int j = 0; j < N; ++j) {
            x[j] = x[j] * dseta[j] * (1 - a[j]) + mean_a_x_dseta;
            X[i][j] = x[j];
        }
    }

    return X;
}

vector<vector<double>> _evolve_defection(vector<double> x, vector<double> a, double mu = 1, double sigma = 0.1, int steps = 1000) {
    assert(x.size() == a.size() && "Número de agentes inconsistente");

    int N = x.size();
    vector<vector<double>> X_coop(steps + 1, vector<double>(N));
    vector<vector<double>> X_def(steps + 1, vector<double>(N));
    X_coop[0] = x;
    X_def[0] = x;

    for (int i = 1; i <= steps; ++i) {
        vector<double> dseta = generate_normal_distribution(mu, sigma, N);

        for (int j = 0; j < N; ++j) {
            x[j] = x[j] * dseta[j] * (1 - a[j]) + inner_product(a.begin(), a.end(), dseta.begin(), 0.0) / N;
            X_coop[i][j] = x[j];
            X_def[i][j] = X_def[i][j] * dseta[j];
        }
    }

    return X_coop;
}

vector<vector<double>> _evolve_network_cooperation(vector<vector<int>> Adj, vector<double> x, vector<double> a, double mu = 1, double sigma = 0.1, int steps = 1000) {
    assert(x.size() == a.size() && "Número de agentes inconsistente");
    int N = x.size();
    assert(Adj.size() == N && Adj[0].size() == N && "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes");

    vector<vector<double>> X(steps + 1, vector<double>(N));
    X[0] = x;

    for (int i = 1; i <= steps; ++i) {
        vector<double> dseta = generate_normal_distribution(mu, sigma, N);
        vector<double> x_aux(N);

        for (int j = 0; j < N; ++j) {
            x_aux[j] = x[j] * dseta[j] * a[j];
        }

        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            double adj_sum = 0.0;

            for (int k = 0; k < N; ++k) {
                sum += Adj[j][k] * x_aux[k];
                adj_sum += Adj[j][k];
            }

            double mean = adj_sum != 0 ? sum / adj_sum : 0.0;
            x[j] = x[j] * dseta[j] * (1 - a[j]) + mean;
            X[i][j] = x[j];
        }
    }

    return X;
}

vector<vector<double>> _evolve_network_defection(vector<vector<int>> Adj, vector<double> x, vector<double> a, double mu = 1, double sigma = 0.1, int steps = 1000) {
    assert(x.size() == a.size() && "Número de agentes inconsistente");
    int N = x.size();
    assert(Adj.size() == N && Adj[0].size() == N && "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes");

    vector<vector<double>> X_coop(steps + 1, vector<double>(N));
    vector<vector<double>> X_def(steps + 1, vector<double>(N));
    X_coop[0] = x;
    X_def[0] = x;

    for (int i = 1; i <= steps; ++i) {
        vector<double> dseta = generate_normal_distribution(mu, sigma, N);
        vector<double> x_aux(N);

        for (int j = 0; j < N; ++j) {
            x_aux[j] = x[j] * dseta[j] * a[j];
        }

        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            double adj_sum = 0.0;

            for (int k = 0; k < N; ++k) {
                sum += Adj[j][k] * x_aux[k];
                adj_sum += Adj[j][k];
            }

            double mean = adj_sum != 0 ? sum / adj_sum : 0.0;
            x[j] = x[j] * dseta[j] * (1 - a[j]) + mean;
            X_def[i][j] = X_def[i][j] * dseta[j];
            X_coop[i][j] = x[j];
        }
    }

    return X_coop;
}

vector<vector<double>> evolve(vector<double> x, vector<double> a, double mu = 1, double sigma = 0.1, bool defection = true, int steps = 1000, vector<vector<int>> Adj = {}) {
    bool network = !Adj.empty();

    if (defection && network) {
        return _evolve_network_defection(Adj, x, a, mu, sigma, steps);
    }
    if (defection && !network) {
        return _evolve_defection(x, a, mu, sigma, steps);
    }
    if (!defection && network) {
        return _evolve_network_cooperation(Adj, x, a, mu, sigma, steps);
    }
    if (!defection && !network) {
        return _evolve_cooperation(x, a, mu, sigma, steps);
    }

    throw invalid_argument("Invalid parameters entered");
}

vector<vector<double>> get_growth(vector<vector<double>> X) {
    int steps = X.size();
    int N = X[0].size();

    vector<vector<double>> Gamma(steps, vector<double>(N));
    vector<double> x_ini = X[0];

    for (int i = 1; i < steps; ++i) {
        for (int j = 0; j < N; ++j) {
            Gamma[i][j] = log(X[i][j] / x_ini[j]) / i;
        }
    }

    return Gamma;
}

int main() {
    vector<vector<int>> Adj = {
        {0,1,1,1,1,1,1},
        {1,0,1,0,0,0,0},
        {1,1,0,1,0,0,0},
        {1,0,1,0,0,0,0},
        {1,0,0,0,0,1,0},
        {1,0,0,0,1,0,0},
        {1,0,0,0,0,0,0}
    };

    int N = 7;
    double x0 = 1;
    double a_i = 0.2;
    vector<double> x_ini(N, x0);
    vector<double> a(N, a_i);
    int steps = 1000;
    double mu = 1;
    double sigma = 0.1;

    vector<vector<double>> X = evolve(x_ini, a, mu, sigma, false, steps, Adj);

    // Guardar la matriz X en un archivo
    save_matrix_to_file(X, "X.txt");

    
    return 0;
}
