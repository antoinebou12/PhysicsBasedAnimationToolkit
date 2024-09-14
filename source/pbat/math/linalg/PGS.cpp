#include "PGSSolver.h"
#include <Eigen/Sparse>

namespace pbat {
namespace math {
namespace linalg {

PGSSolver::PGS() = default;

double PGS::ApplyProjection(double xi, double lower, double upper) const {
    // Apply bounds projection, ensuring that xi is within the specified range [lower, upper].
    if (xi < lower) return lower;
    if (xi > upper) return upper;
    return xi;
}

bool PGSSolver::Solve(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    Eigen::VectorXd& x,
    const Eigen::VectorXd& lower_bound,
    const Eigen::VectorXd& upper_bound,
    int maxIterations,
    double tolerance,
    double omega)
{
    const int n = A.rows();
    x = Eigen::VectorXd::Zero(n); // Initialize the solution vector to zero
    Eigen::VectorXd residual(n);

    bool has_lower_bound = (lower_bound.size() == n);
    bool has_upper_bound = (upper_bound.size() == n);

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

            // Calculate new value for x[i] using the PGS update formula
            double xi_new = (b[i] - sum) / Aii;
            xi_new = omega * xi_new + (1.0 - omega) * x[i]; // Over-relaxation

            // Apply projection to satisfy constraints (if any)
            if (has_lower_bound && has_upper_bound) {
                x[i] = ApplyProjection(xi_new, lower_bound[i], upper_bound[i]);
            } else if (has_lower_bound) {
                x[i] = ApplyProjection(xi_new, lower_bound[i], std::numeric_limits<double>::infinity());
            } else if (has_upper_bound) {
                x[i] = ApplyProjection(xi_new, -std::numeric_limits<double>::infinity(), upper_bound[i]);
            } else {
                x[i] = std::max(0.0, xi_new); // Default projection for LCP
            }
        }

        // Compute the residual and check for convergence
        residual = A * x - b;
        if (residual.norm() < tolerance) {
            return true; // Converged
        }

        // If the norm difference between iterations is small, terminate early
        if ((x - x_old).norm() < tolerance) {
            return true; // Converged
        }
    }

    return false; // Did not converge within the given iterations
}

} // namespace linalg
} // namespace math
} // namespace pbat
