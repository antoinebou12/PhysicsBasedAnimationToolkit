#include "Bvh.h"

#include <pbat/gpu/geometry/Bvh.h>
#include <pbat/gpu/geometry/Primitives.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBvh(pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    using namespace pbat::gpu::geometry;
    pyb::class_<Bvh>(m, "LinearBvh")
        .def(
            pyb::init([](std::size_t nPrimitives, std::size_t nOverlaps) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Bvh.Construct", [&]() {
                    Bvh bvh(nPrimitives, nOverlaps);
                    return bvh;
                });
            }),
            pyb::arg("max_boxes"),
            pyb::arg("max_overlaps"),
            "Allocate BVH on GPU for max_boxes primitives, which can detect a maximum of "
            "max_overlaps box overlaps.")
        .def(
            "build",
            [](Bvh& bvh,
               Points const& P,
               Simplices const& S,
               Eigen::Vector<GpuScalar, 3> const& min,
               Eigen::Vector<GpuScalar, 3> const& max,
               GpuScalar expansion) {
                pbat::profiling::Profile("pbat.gpu.geometry.Bvh.Build", [&]() {
                    bvh.Build(P, S, min, max, expansion);
                });
            },
            pyb::arg("P"),
            pyb::arg("S"),
            pyb::arg("min"),
            pyb::arg("max"),
            pyb::arg("expansion") = GpuScalar{0},
            "Constructs, on the GPU, a bounding volume hierarchy of axis-aligned boxes over the "
            "simplex set S with vertex positions P. (min,max) denote the extremeties of an "
            "axis-aligned bounding box embedding (P,S). expansion inflates the BVH nodes' bounding "
            "boxes as set by the user.")
        .def(
            "detect_self_overlaps",
            [](Bvh& bvh, Simplices const& S) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Bvh.DetectSelfOverlaps", [&]() {
                    return bvh.DetectSelfOverlaps(S);
                });
            },
            pyb::arg("S"),
            "Detect self-overlaps (si,sj) between bounding boxes of simplices (si,sj) of S into a "
            "|#overlaps|x2 array, where si < sj. S must index into points P and was used in the "
            "most recent call to build.");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat