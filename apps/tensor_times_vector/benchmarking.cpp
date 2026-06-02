#include "taco.h"
#include <chrono>
#include <iostream>
#include <random>

using namespace taco;

int main() {
    // Define formats
    Format csr({Dense, Sparse});   // CSR matrix
    Format dv({Dense});            // dense vector

    // Read matrix
    Tensor<double> A = read("pwtk.mtx", csr);

    int nrow = A.getDimension(0);
    int ncol = A.getDimension(1);

    // Create tensors
    Tensor<double> x({ncol}, dv);
    Tensor<double> z({nrow}, dv);
    Tensor<double> y({nrow}, dv);

    // Fill x and z
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int k = 0; k < ncol; k++) {
        x.insert({k}, dist(gen));
    }
    x.pack();

    for (int k = 0; k < nrow; k++) {
        z.insert({k}, dist(gen));
    }
    z.pack();

    // Define index variables
    IndexVar i, j;

    // Define SpMV: y[i] = A[i,j] * x[j] + z[i]
    y(i) = A(i, j) * x(j) + z(i);

    // Generate and compile code before timing
    y.compile();

    // Time only the actual computation
    auto start = std::chrono::high_resolution_clock::now();

    y.compute();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}