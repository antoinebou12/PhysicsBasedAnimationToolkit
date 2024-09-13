#include "PGS.h"
#include <Eigen/Sparse>

namespace pbat {
namespace math {
namespace linalg {

PGSSolver::PGS() = default;

bool PGS::Solve(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    Eigen::VectorXd& x,
    int maxIterations,
    double tolerance)
{
    const int n = A.rows();
    x = Eigen::VectorXd::Zero(n); // Initialize the solution vector to zero

    for (int k = 0; k < maxIterations; ++k) {
        Eigen::VectorXd x_old = x;

        // Iterating over each variable
        for (int i = 0; i < n; ++i) {
            double Aii = A.coeff(i, i);
            if (Aii == 0) continue; // Skip if the diagonal element is zero

            double sum = 0.0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if (it.row() != i) {
                    sum += it.value() * x[it.row()];
                }
            }

            // Update x[i] using the PGS formula
            x[i] = (b[i] - sum) / Aii;

            // Apply projection (for LCP problems, project x[i] >= 0)
            if (x[i] < 0) {
                x[i] = 0;
            }
        }

        // Check convergence
        if ((x - x_old).norm() < tolerance) {
            return true; // Converged
        }
    }

    return false; // Did not converge within the given iterations
}

} // namespace linalg
} // namespace math
} // namespace pbat
