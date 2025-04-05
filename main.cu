#include <fmt/core.h>

#include <stdexcept>  // For std::runtime_error
#include <utility>

#include "external/Qui1Framework/include/common/error_check.cuh"
#include "external/Qui1Framework/include/matrix/qui1_device_matrix.cuh"
#include "external/Qui1Framework/include/matrix/qui1_matrix_helper.cuh"
#include "external/Qui1Framework/include/wrapper/solver/qui1_cusolver_wrapper.cuh"
#include "external/Qui1Framework/include/common/device_info_print.cuh"

using data_type = float;

int main(int argc, char** argv) {
    qui1::common::print_device_info();
    {
        const size_t N = 32;
        fmt::print("Matrix size N = {}\n", N);
        // 1. Create matrix and fill with random data
        qui1::DeviceMatrix<data_type> A(N, N);
        qui1::MatrixHelper::fillWithRandom(A);
        // 2. Get a view (using right-value reference)
        auto&& A_view = A.getView(4, 4, 1, 1);
        qui1::MatrixHelper::printMatrix(A_view);
        // 3. Prepare for LU decomposition
        qui1::CusolverWrapper solver;
        int* devIpiv = nullptr;
        const int min_mn = static_cast<int>(std::min(A.getRows(), A.getCols()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&devIpiv), sizeof(int) * min_mn));
        // 4. Perform LU decomposition
        solver.getrf(A_view, devIpiv);
        qui1::MatrixHelper::printMatrix(A_view);
        // 5. Clean up
        if (devIpiv) {
            CUDA_CHECK(cudaFree(devIpiv));
        }
    }
    {

    }
    return 0;
}
