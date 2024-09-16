#include "PGS.h"
#include <pybind11/eigen.h>
#include <pbat/math/linalg/PGS.h>
#include <pbat/profiling/Profiling.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindPGS([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    std::string const className = "PGS";
    using PGSType = pbat::math::linalg::PGS;

    pyb::class_<PGSType>(m, className.data())
        .def(pyb::init<int>(), pyb::arg("maxIter") = 100) // Constructor with default maxIter
        .def(
            "solve",
            [](PGSType& solver,  // Pass solver to capture the method invocation
               Eigen::Ref<const Eigen::MatrixXf> A, 
               Eigen::Ref<const Eigen::VectorXf> b,
               Eigen::Ref<Eigen::VectorXf> x,
               Eigen::Ref<const Eigen::VectorXf> lowerBounds,
               Eigen::Ref<const Eigen::VectorXf> upperBounds) {
                   pbat::profiling::Profile("math.linalg.PGS.Solve", [&]() {
                       solver.Solve(A, b, x, lowerBounds, upperBounds); // Use solver object to call method
                   });
               },
            pyb::arg("A"),
            pyb::arg("b"),
            pyb::arg("x"),
            pyb::arg("lower_bounds"),
            pyb::arg("upper_bounds"),
            "Solves a linear system using Projected Gauss-Seidel method with box constraints.")
        .def(
            "set_max_iterations",
            [](PGSType& solver, int maxIter) { // Use solver instead of '.' to call method
                solver.SetMaxIterations(maxIter);
            },
            pyb::arg("max_iter"),
            "Sets the maximum number of iterations for the solver.")
        .def(
            "get_max_iterations",
            [](PGSType& solver) {  // Use solver instead of '.' to call method
                return solver.GetMaxIterations();
            },
            "Returns the maximum number of iterations for the solver.");
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat
