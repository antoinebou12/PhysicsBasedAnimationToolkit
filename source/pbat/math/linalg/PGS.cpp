#include "PGS.h"
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace pbat {
namespace math {
namespace linalg {

PGS::PGS(int maxIter) : m_maxIter(maxIter)
{
}

void PGS::Solve(Eigen::MatrixXf const& A, Eigen::VectorXf const& b, Eigen::VectorXf& x, Eigen::VectorXf const& lowerBounds, Eigen::VectorXf const& upperBounds) const
{
    for (int iter = 0; iter < m_maxIter; ++iter)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(A.rows())), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                float sigma = 0.0f;
                for (int j = 0; j < static_cast<int>(A.cols()); ++j)  // Cast A.cols() to int
                {
                    if (i != j)
                    {
                        sigma += A(i, j) * x(j);
                    }
                }

                x(i) = (b(i) - sigma) / A(i, i);

                // Apply the projection onto the box constraints
                x(i) = std::max(lowerBounds(i), std::min(upperBounds(i), x(i)));
            }
        });
    }
}

PGS::~PGS()
{
    // Destructor logic (if needed)
}



} // namespace linalg
} // namespace math
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[math][linalg] PGS")
{
    using namespace pbat::math::linalg;

    // Arrange
    PGS solver(100); // Set maximum iterations to 100
    auto constexpr n = 3;
    Eigen::MatrixXf A = Eigen::MatrixXf::Identity(n, n) * 4.0f; // Simple diagonal matrix
    Eigen::VectorXf b = Eigen::VectorXf::Constant(n, 2.0f);     // Constant vector
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);               // Initial guess (zero vector)
    
    // Define box constraints (lower and upper bounds)
    Eigen::VectorXf lowerBounds = Eigen::VectorXf::Constant(n, 0.0f); // Non-negative solution
    Eigen::VectorXf upperBounds = Eigen::VectorXf::Constant(n, 5.0f); // Upper bound is 5.0

    SUBCASE("Solves simple SPD system with box constraints")
    {
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expected result: x = b / A diagonal = 2 / 4 = 0.5 for all elements, within bounds
        CHECK(x.isApprox(Eigen::VectorXf::Constant(n, 0.5f), 1e-5f));
    }

    SUBCASE("Respects upper bounds in solution")
    {
        // Change the right-hand side to test upper bound projection
        b = Eigen::VectorXf::Constant(n, 30.0f);
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expect x to be capped at the upper bound (5.0) for all elements
        CHECK(x.isApprox(upperBounds, 1e-5f));
    }

    SUBCASE("Respects lower bounds in solution")
    {
        // Change the right-hand side to test lower bound projection
        b = Eigen::VectorXf::Constant(n, -30.0f);
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expect x to be capped at the lower bound (0.0) for all elements
        CHECK(x.isApprox(lowerBounds, 1e-5f));
    }
}
