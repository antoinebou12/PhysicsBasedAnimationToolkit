#ifndef PBAT_MATH_LINALG_MULTIGRID_SOLVER_H
#define PBAT_MATH_LINALG_MULTIGRID_SOLVER_H

#include <Eigen/Dense>
#include <vector>

namespace pbat {
namespace math {
namespace linalg {

class MultigridSolver
{
public:
    MultigridSolver(int levels = 2, int maxIter = 5);
    ~MultigridSolver();

    void Solve(const Eigen::MatrixXf& A,
               const Eigen::VectorXf& b,
               Eigen::VectorXf& x);

private:
    int m_levels;
    int m_maxIter;

    void VCycle(int level,
                const std::vector<Eigen::MatrixXf>& A_levels,
                const std::vector<Eigen::MatrixXf>& P_levels,
                Eigen::VectorXf& x,
                const Eigen::VectorXf& b);
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MULTIGRID_SOLVER_H
