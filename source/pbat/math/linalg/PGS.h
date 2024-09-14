#ifndef PBAT_MATH_LINALG_PGS_H
#define PBAT_MATH_LINALG_PGS_H

#include "pbat/Aliases.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat {
namespace math {
namespace linalg {

/**
 * @brief Projected Gauss-Seidel (PGS) solver for solving linear complementary problems (LCPs)
 *        and systems with optional upper and lower bound constraints.
 */
class PGS
{
  public:
    /**
     * @brief Default constructor for PGSSolver.
     */
    PBAT_API PGS();

    /**
     * @brief Solves the system using the PGS method with optional constraints.
     *
     * @param A The system matrix (sparse).
     * @param b The right-hand side vector.
     * @param x The solution vector.
     * @param lower_bound Optional lower bound constraints (default: no lower bound).
     * @param upper_bound Optional upper bound constraints (default: no upper bound).
     * @param maxIterations The maximum number of iterations.
     * @param tolerance The tolerance level for convergence.
     * @param omega Over-relaxation parameter (default: 1.0, no relaxation).
     * @return true If the solution converged.
     * @return false If the solution did not converge within the maximum number of iterations.
     */
    PBAT_API bool Solve(
        const Eigen::SparseMatrix<double>& A,
        const Eigen::VectorXd& b,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& lower_bound = Eigen::VectorXd(),
        const Eigen::VectorXd& upper_bound = Eigen::VectorXd(),
        int maxIterations = 1000,
        double tolerance = 1e-5,
        double omega = 1.0);

  private:
    // Helper function to apply projection with bounds.
    inline double ApplyProjection(double xi, double lower, double upper) const;
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGS_SOLVER_H
