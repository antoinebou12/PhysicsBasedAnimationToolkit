#include "Kernel.h"
#include <pybind11/eigen.h> // For Eigen types
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pbat {
namespace py {
namespace fluid {

void Bind(pybind11::module& m)
{
    py::class_<SPH::Kernel>(m, "Kernel")
        .def(py::init<float>(), py::arg("h") = 0.0f, "Constructs a Kernel object with smoothing length h.")
        .def("poly6", &SPH::Kernel::poly6, "The poly6 kernel function.",
             py::arg("pi"), py::arg("pj"))
        .def("spiky", &SPH::Kernel::spiky, "The spiky kernel gradient function.",
             py::arg("pi"), py::arg("pj"))
        .def("scorr", &SPH::Kernel::scorr, "Computes the correction factor for density fluctuation.")
        .def("__repr__", [](const SPH::Kernel& k) {
            return "<SPH.Kernel with smoothing length>";
        });
}
} // namespace fluid
} // namespace py
} // namespace pbat
