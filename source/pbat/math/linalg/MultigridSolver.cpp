#include "MultigridSolver.h"

namespace pbat {
namespace math {
namespace linalg {

MultigridSolver::MultigridSolver(int levels, int maxIter)
    : m_levels(levels), m_maxIter(maxIter)
{
}

MultigridSolver::~MultigridSolver()
{
}

void MultigridSolver::Solve(const Eigen::MatrixXf& A,
                            const Eigen::VectorXf& b,
                            Eigen::VectorXf& x)
{
    // Build multigrid levels (simplified example)
    std::vector<Eigen::MatrixXf> A_levels;
    std::vector<Eigen::MatrixXf> P_levels; // Prolongation operators

    A_levels.push_back(A);

    for (int l = 1; l < m_levels; ++l)
    {
        int size = A_levels[l - 1].rows() / 2;
        Eigen::MatrixXf A_coarse = A_levels[l - 1].block(0, 0, size, size);
        A_levels.push_back(A_coarse);

        // Simplified prolongation operator
        Eigen::MatrixXf P = Eigen::MatrixXf::Zero(size * 2, size);
        for (int i = 0; i < size; ++i)
        {
            P(2 * i, i) = 1.0f;
        }
        P_levels.push_back(P);
    }

    // Start V-Cycle
    VCycle(0, A_levels, P_levels, x, b);
}

void MultigridSolver::VCycle(int level,
                             const std::vector<Eigen::MatrixXf>& A_levels,
                             const std::vector<Eigen::MatrixXf>& P_levels,
                             Eigen::VectorXf& x,
                             const Eigen::VectorXf& b)
{
    // Pre-smoothing using a few iterations of Jacobi
    for (int iter = 0; iter < m_maxIter; ++iter)
    {
        Eigen::VectorXf x_new = x;
        for (int i = 0; i < A_levels[level].rows(); ++i)
        {
            float sigma = A_levels[level].row(i).dot(x) - A_levels[level](i, i) * x(i);
            x_new(i) = (b(i) - sigma) / A_levels[level](i, i);
        }
        x = x_new;
    }

    // Compute residual
    Eigen::VectorXf r = b - A_levels[level] * x;

    if (level + 1 < m_levels)
    {
        // Restrict residual to coarse grid
        Eigen::VectorXf r_coarse = P_levels[level].transpose() * r;

        // Initialize error vector on coarse grid
        Eigen::VectorXf e_coarse = Eigen::VectorXf::Zero(r_coarse.size());

        // Recursive call to VCycle
        VCycle(level + 1, A_levels, P_levels, e_coarse, r_coarse);

        // Prolongate error and correct
        x += P_levels[level] * e_coarse;
    }
    else
    {
        // Solve on the coarsest grid
        x += A_levels[level].fullPivLu().solve(r);
    }

    // Post-smoothing
    for (int iter = 0; iter < m_maxIter; ++iter)
    {
        Eigen::VectorXf x_new = x;
        for (int i = 0; i < A_levels[level].rows(); ++i)
        {
            float sigma = A_levels[level].row(i).dot(x) - A_levels[level](i, i) * x(i);
            x_new(i) = (b(i) - sigma) / A_levels[level](i, i);
        }
        x = x_new;
    }
}

} // namespace linalg
} // namespace math
} // namespace pbat
