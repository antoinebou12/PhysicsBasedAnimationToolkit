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
                const Eigen::VectorXf& preconditioner) const
{
    Eigen::VectorXf M_inv_diag;
    if (preconditioner.size() == 0)
    {
        // No preconditioner provided; use inverse of the diagonal of A
        M_inv_diag = A.diagonal().cwiseInverse();
    }
    else
    {
        M_inv_diag = preconditioner;
    }

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

            float xi = M_inv_diag(i) * (b(i) - sigma);

            // Apply the projection onto the box constraints
            x(i) = std::max(lowerBounds(i), std::min(upperBounds(i), xi));
        }
    }
}

} // namespace linalg
} // namespace math
} // namespace pbat
