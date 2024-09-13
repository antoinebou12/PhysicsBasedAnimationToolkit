#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // For Eigen types

#include "Kernel.h"

namespace py = pybind11;

PYBIND11_MODULE(sph_kernel, m)
{
    py::class_<SPH::Kernel>(m, "Kernel")
        .def(py::init<float>(), py::arg("h") = 0.0f, "Constructs a Kernel object with smoothing length h.")
        .def("poly6", &SPH::Kernel::poly6, "The poly6 kernel function.",
             py::arg("pi"), py::arg("pj"))
        .def("spiky", &SPH::Kernel::spiky, "The spiky kernel gradient function.",
             py::arg("pi"), py::arg("pj"))
        .def("scorr", &SPH::Kernel::scorr, "Computes the correction factor for density fluctuation.");
}
