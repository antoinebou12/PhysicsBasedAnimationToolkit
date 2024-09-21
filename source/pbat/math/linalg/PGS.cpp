#include "PGS.h"
#include <algorithm>

namespace pbat {
namespace math {
namespace linalg {

PGS::PGS(int maxIter) : m_maxIter(maxIter)
{
}

PGS::~PGS()
{
}

void PGS::SetMaxIterations(int maxIter)
{
    m_maxIter = maxIter;
}

int PGS::GetMaxIterations() const
{
    return m_maxIter;
}

void PGS::Solve(const Eigen::MatrixXf& A,
                const Eigen::VectorXf& b,
                Eigen::VectorXf& x,
                const Eigen::VectorXf& lowerBounds,
                const Eigen::VectorXf& upperBounds,
                const Eigen::MatrixXf& preconditioner) const
{
    Eigen::MatrixXf M;
    if (preconditioner.size() == 0)
    {
        // No preconditioner provided; use the identity matrix
        M = Eigen::MatrixXf::Identity(A.rows(), A.cols());
    }
    else
    {
        M = preconditioner;
    }

    Eigen::MatrixXf M_inv = M.inverse();

    for (int iter = 0; iter < m_maxIter; ++iter)
    {
        for (int i = 0; i < A.rows(); ++i)
        {
            float sigma = 0.0f;
            for (int j = 0; j < A.cols(); ++j)
            {
                if (i != j)
                {
                    sigma += A(i, j) * x(j);
                }
            }

            float xi = M_inv.row(i).dot(b - sigma * Eigen::VectorXf::Unit(A.rows(), i));

            // Apply the projection onto the box constraints
            x(i) = std::max(lowerBounds(i), std::min(upperBounds(i), xi));
        }
    }
}

} // namespace linalg
} // namespace math
} // namespace pbat
