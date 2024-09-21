#ifndef PBAT_MATH_LINALG_PRECONDITIONERS_H
#define PBAT_MATH_LINALG_PRECONDITIONERS_H

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Include TBB headers for parallelization
#include <tbb/tbb.h>
#include <tbb/mutex.h>

namespace pbat {
namespace math {
namespace linalg {

class Preconditioners
{
public:
    // Diagonal (Jacobi) preconditioner
    static Eigen::VectorXf Diagonal(const Eigen::MatrixXf& A)
    {
        return A.diagonal();
    }

    // Inverse of the diagonal preconditioner
    static Eigen::VectorXf InverseDiagonal(const Eigen::MatrixXf& A)
    {
        return A.diagonal().cwiseInverse();
    }

    // Gauss-Seidel preconditioner (Lower triangular part)
    static Eigen::MatrixXf GaussSeidel(const Eigen::MatrixXf& A)
    {
        return A.triangularView<Eigen::Lower>();
    }

    // Successive Over-Relaxation (SOR) preconditioner
    static Eigen::MatrixXf SOR(const Eigen::MatrixXf& A, float omega)
    {
        Eigen::MatrixXf M = (1.0f / omega) * A.diagonal().asDiagonal() + A.triangularView<Eigen::StrictlyLower>();
        return M;
    }

    // Incomplete LU (ILU) preconditioner
    static Eigen::SparseMatrix<float> ILU(const Eigen::SparseMatrix<float>& A)
    {
        // Use Eigen's built-in incomplete LU factorization
        Eigen::IncompleteLU<Eigen::SparseMatrix<float>> ilu;
        ilu.compute(A);

        // Extract L and U factors
        Eigen::SparseMatrix<float> L = ilu.matrixL();
        Eigen::SparseMatrix<float> U = ilu.matrixU();

        // Return combined preconditioner
        return L * U;
    }

    // Symmetric Successive Over-Relaxation (SSOR) preconditioner
    static Eigen::MatrixXf SSOR(const Eigen::MatrixXf& A, float omega)
    {
        Eigen::MatrixXf D_inv = A.diagonal().cwiseInverse().asDiagonal();
        Eigen::MatrixXf L = A.triangularView<Eigen::StrictlyLower>();
        Eigen::MatrixXf U = A.triangularView<Eigen::StrictlyUpper>();
        Eigen::MatrixXf M = (1.0f / omega) * D_inv + L * D_inv * U;
        return M;
    }

    // Multilevel Additive Schwarz (MAS) preconditioner
    static void MAS(const Eigen::MatrixXd& A,
                    const Eigen::VectorXd& r,
                    Eigen::VectorXd& z,
                    int num_subdomains_per_dim,
                    int overlap)
    {
        // Assuming A is N*N x N*N matrix for a 2D grid
        int N = static_cast<int>(std::sqrt(A.rows()));
        if (N * N != A.rows()) {
            throw std::runtime_error("Matrix A size is not a perfect square.");
        }

        // Generate subdomains
        std::vector<Subdomain> subdomains = create_subdomains(N, num_subdomains_per_dim, overlap);

        // Initialize z to zero
        z.setZero();

        // Mutex for thread-safe accumulation
        tbb::mutex z_mutex;

        // Use TBB parallel_for to process subdomains in parallel
        tbb::parallel_for(tbb::blocked_range<size_t>(0, subdomains.size()),
                          [&](const tbb::blocked_range<size_t>& range) {
            for (size_t idx = range.begin(); idx != range.end(); ++idx) {
                const Subdomain& sd = subdomains[idx];
                int sub_size = static_cast<int>(sd.indices.size());
                if (sub_size == 0) continue;

                // Extract A_sub and r_sub
                Eigen::MatrixXd A_sub(sub_size, sub_size);
                Eigen::VectorXd r_sub(sub_size);

                for (int i = 0; i < sub_size; ++i) {
                    r_sub(i) = r(sd.indices[i]);
                    for (int j = 0; j < sub_size; ++j) {
                        A_sub(i, j) = A(sd.indices[i], sd.indices[j]);
                    }
                }

                // Solve A_sub * z_sub = r_sub
                Eigen::VectorXd z_sub = A_sub.ldlt().solve(r_sub);

                // Accumulate z_sub into z
                // Use mutex to protect shared resource
                {
                    tbb::mutex::scoped_lock lock(z_mutex);
                    for (int i = 0; i < sub_size; ++i) {
                        z(sd.indices[i]) += z_sub(i);
                    }
                }
            }
        });
    }

private:
    // Struct to represent a subdomain
    struct Subdomain {
        int start_row;
        int end_row;
        int start_col;
        int end_col;
        std::vector<int> indices;
    };

    // Function to generate overlapping subdomains
    static std::vector<Subdomain> create_subdomains(int N, int num_subdomains_per_dim, int overlap)
    {
        std::vector<Subdomain> subdomains;
        int step = N / num_subdomains_per_dim;
        if (step == 0) step = 1; // Ensure step is at least 1 to avoid division by zero

        for (int i = 0; i < num_subdomains_per_dim; ++i) {
            for (int j = 0; j < num_subdomains_per_dim; ++j) {
                Subdomain sd;
                sd.start_row = std::max(i * step - overlap, 0);
                sd.end_row = std::min((i + 1) * step + overlap, N);
                sd.start_col = std::max(j * step - overlap, 0);
                sd.end_col = std::min((j + 1) * step + overlap, N);

                // Populate global indices
                for (int row = sd.start_row; row < sd.end_row; ++row) {
                    for (int col = sd.start_col; col < sd.end_col; ++col) {
                        int idx = row * N + col;
                        sd.indices.push_back(idx);
                    }
                }

                subdomains.push_back(sd);
            }
        }

        return subdomains;
    }
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PRECONDITIONERS_H
