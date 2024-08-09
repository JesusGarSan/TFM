#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <fstream>
#include "evolution.cpp" // Incluye las funciones que definiste anteriormente

using namespace std;

struct SimulationResults {
    vector<vector<double>> X_coop;
    vector<vector<double>> X_def;
    vector<vector<double>> Gammas_coop;
    vector<vector<double>> Gammas_def;
};


SimulationResults simulate(const vector<double>& x_ini, const vector<double>& a, double mu = 1, double sigma = 0.1, bool defection = true, int steps = 10000, int M = 1, bool plot = false, bool verbose = false, const vector<vector<int>>& Adj = {}) {
    assert(x_ini.size() == a.size() && "Número de agentes inconsistente");
    int N = x_ini.size();
    vector<vector<double>> X_coop(N, vector<double>(M, 0));
    vector<vector<double>> X_def(N, vector<double>(M, 0));
    vector<vector<double>> Gammas_coop(N, vector<double>(M, 0));
    vector<vector<double>> Gammas_def(N, vector<double>(M, 0));

    for (int m = 0; m < M; ++m) {
        if (verbose) {
            cout << "Running simulation " << m + 1 << "..." << endl;
        }

        vector<vector<double>> x_coop, x_def;

        EvolutionResult result = evolve(x_ini, a, mu, sigma, true, steps, Adj);
        x_coop = result.X_coop;
        x_def = result.X_def;

        for (int i = 0; i < N; ++i) {
            X_coop[i][m] = x_coop.back()[i];
            Gammas_coop[i][m] = get_growth(x_coop).back()[i];

            X_def[i][m] = x_def.back()[i];
            Gammas_def[i][m] = get_growth(x_def).back()[i];
        }
    }

    SimulationResults results = {X_coop, X_def, Gammas_coop, Gammas_def};
    return results;
}




void fig_1_data(const vector<int>& N_array, const vector<double>& a_1_array, int M = 10, int steps = 10000, double x_ini = 1.0, double a_i = 0.5, double mu = 1.0, double sigma = 0.1, bool save = false, bool verbose = false) {
    for (const auto& N : N_array) {
        for (const auto& a_1 : a_1_array) {
            if (verbose) {
                cout << "Running for " << N << " agents & share parameter " << a_1 << "..." << endl;
            }

            vector<double> x(N, x_ini);
            vector<double> a(N, a_i);
            a[0] = a_1;

            SimulationResults results = simulate(x, a, mu, sigma, true, steps, M);

            vector<double> Gammas_rel(M);
            for (int m = 0; m < M; ++m) {
                Gammas_rel[m] = (results.Gammas_coop[0][m] - results.Gammas_def[0][m]) * 100;
            }

            double gamma = accumulate(results.Gammas_coop[0].begin(), results.Gammas_coop[0].end(), 0.0) / M * 100;
            double gamma_def = accumulate(results.Gammas_def[0].begin(), results.Gammas_def[0].end(), 0.0) / M * 100;
            double gamma_rel = accumulate(Gammas_rel.begin(), Gammas_rel.end(), 0.0) / M;
            double error = sqrt(inner_product(Gammas_rel.begin(), Gammas_rel.end(), Gammas_rel.begin(), 0.0) / M) / sqrt(M);
            double rel_error = error / gamma_rel;

            if (save) {
                ofstream file("./data/relative_long_term_growth_rate.csv", ios::app);
                if (file.is_open()) {
                    file << N << "," << x_ini << "," << a_i << "," << a_1 << "," << mu << "," << sigma << "," << steps << "," << M << "," 
                         << gamma << "," << gamma_def << "," << gamma_rel << "," << error << "," << rel_error << endl;
                    file.close();
                } else {
                    cerr << "No se pudo abrir el archivo para guardar los datos." << endl;
                }
            }
        }
    }
}


int main() {
    vector<int> N = {2, 3, 4, 6, 10};
    // Esto hay que cambiarlo, es memoria dinámica.
    vector<double> a;
    for (double i = 0.02; i < 1.5; i += 0.02) {
        a.push_back(i);
    }
    int M = 100;                              // Número de simulaciones
    int steps = 10000;                        // Número de pasos
    double x_ini = 1.0;                       // Valor inicial de los agentes
    double a_i = 0.5;                         // Valor del parámetro de compartición
    double mu = 1.0;                          // Media de la distribución normal
    double sigma = 0.1;                       // Desviación estándar de la distribución normal
    bool save = true;                         // Guardar los resultados
    bool verbose = true;                      // Modo verboso

    // Llamada completa a la función fig_1_data
    fig_1_data(N, a, M, steps, x_ini, a_i, mu, sigma, save, verbose);
    return 0;
}
