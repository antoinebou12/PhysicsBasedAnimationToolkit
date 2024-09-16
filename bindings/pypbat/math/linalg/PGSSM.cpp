#include "PGSSM.h"
#include <pybind11/eigen.h>
#include <pbat/math/linalg/PGSSM.h>
#include <pbat/profiling/Profiling.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindPGSSM([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    std::string const className = "PGSSM";
    using PGSSMType = pbat::math::linalg::PGSSM;

    pyb::class_<PGSSMType>(m, className.data())
        // Constructor: PGSSM(maxIter=100, subIter=10)
        .def(pyb::init<int, int>(), 
             pyb::arg("maxIter") = 100, pyb::arg("subIter") = 10, 
             "Constructor that initializes PGSSM with max iterations and subspace iterations.")

        // Solve method
        .def("solve", 
             [](PGSSMType& solver, 
                Eigen::Ref<const Eigen::MatrixXf> A, 
                Eigen::Ref<const Eigen::VectorXf> b,
                Eigen::Ref<Eigen::VectorXf> x,
                Eigen::Ref<const Eigen::VectorXf> lowerBounds,
                Eigen::Ref<const Eigen::VectorXf> upperBounds) {
                pbat::profiling::Profile("math.linalg.PGSSM.Solve", [&]() {
                    solver.Solve(A, b, x, lowerBounds, upperBounds);
                });
             },
             pyb::arg("A"), 
             pyb::arg("b"), 
             pyb::arg("x"), 
             pyb::arg("lower_bounds"), 
             pyb::arg("upper_bounds"), 
             R"pbdoc(
                Solves a linear system using the Projected Gauss-Seidel Subspace Minimization method with box constraints.
                
                Parameters:
                A: Matrix representing the system (MatrixXf).
                b: Vector representing the right-hand side of the equation (VectorXf).
                x: Vector where the solution will be stored (VectorXf).
                lower_bounds: Vector representing the lower bounds for each variable (VectorXf).
                upper_bounds: Vector representing the upper bounds for each variable (VectorXf).
             )pbdoc")

        // Set max iterations
        .def("set_max_iterations", 
             [](PGSSMType& solver, int maxIter) {
                if (maxIter <= 0) throw std::invalid_argument("max_iter must be greater than 0");
                solver.SetMaxIterations(maxIter);
             }, 
             pyb::arg("max_iter"), 
             "Sets the maximum number of iterations for the solver.")

        // Get max iterations
        .def("get_max_iterations", 
             [](PGSSMType& solver) {
                return solver.GetMaxIterations();
             }, 
             "Returns the maximum number of iterations for the solver.")

        // Set sub-iterations
        .def("set_sub_iterations", 
             [](PGSSMType& solver, int subIter) {
                if (subIter <= 0) throw std::invalid_argument("sub_iter must be greater than 0");
                solver.SetSubIterations(subIter);
             }, 
             pyb::arg("sub_iter"), 
             "Sets the number of sub-iterations for the subspace minimization.")

        // Get sub-iterations
        .def("get_sub_iterations", 
             [](PGSSMType& solver) {
                return solver.GetSubIterations();
             }, 
             "Returns the number of sub-iterations for the subspace minimization.")

        // Add property-style access for max_iter and sub_iter
        .def_property("max_iterations", 
            [](PGSSMType& solver) { return solver.GetMaxIterations(); }, 
            [](PGSSMType& solver, int maxIter) {
                if (maxIter <= 0) throw std::invalid_argument("max_iter must be greater than 0");
                solver.SetMaxIterations(maxIter);
            },
            "Property to get and set the maximum number of iterations.")

        .def_property("sub_iterations", 
            [](PGSSMType& solver) { return solver.GetSubIterations(); }, 
            [](PGSSMType& solver, int subIter) {
                if (subIter <= 0) throw std::invalid_argument("sub_iter must be greater than 0");
                solver.SetSubIterations(subIter);
            },
            "Property to get and set the number of sub-iterations.");
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat
