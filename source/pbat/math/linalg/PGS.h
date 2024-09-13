#ifndef PBAT_MATH_LINALG_PGS_SOLVER_H
#define PBAT_MATH_LINALG_PGS_SOLVER_H

#include "pbat/Aliases.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat {
namespace math {
namespace linalg {

/**
 * @brief Projected Gauss-Seidel (PGS) solver for solving linear complementary problems (LCPs).
 */
class PGSSolver
{
  public:
    /**
     * @brief Default constructor for PGSSolver.
     */
    PBAT_API PGS();

    /**
     * @brief Solves the system using the PGS method.
     * 
     * @param A The system matrix (sparse).
     * @param b The right-hand side vector.
     * @param x The solution vector.
     * @param maxIterations The maximum number of iterations.
     * @param tolerance The tolerance level for convergence.
     * @return true If the solution converged.
     * @return false If the solution did not converge within the maximum number of iterations.
     */
    PBAT_API bool Solve(
        const Eigen::SparseMatrix<double>& A,
        const Eigen::VectorXd& b,
        Eigen::VectorXd& x,
        int maxIterations = 1000,
        double tolerance = 1e-5);

  private:
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGS_SOLVER_H
