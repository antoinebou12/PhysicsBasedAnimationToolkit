#include "ElasticEnergy.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/lod/ElasticEnergy.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindElasticEnergy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::lod::CageQuadrature;
    using pbat::sim::vbd::lod::ElasticEnergy;
    pyb::class_<ElasticEnergy>(m, "ElasticEnergy")
        .def(
            pyb::init([](Data const& problem, CageQuadrature const& CQ) {
                return ElasticEnergy(problem, CQ);
            }),
            pyb::arg("problem"),
            pyb::arg("cage_quadrature"))
        .def_readwrite("mug", &ElasticEnergy::mug)
        .def_readwrite("lambdag", &ElasticEnergy::lambdag);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat