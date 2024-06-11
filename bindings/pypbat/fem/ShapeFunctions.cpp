#include "ShapeFunctions.h"

#include "For.h"
#include "Mesh.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>
#include <string>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kMaxQuadratureOrder = 6u;
        auto const throw_bad_quad_order    = [](int qorder) {
            std::string const what = fmt::format(
                "Invalid quadrature order={}, supported orders are [1,{}]",
                qorder,
                kMaxQuadratureOrder);
            throw std::invalid_argument(what);
        };
        std::string const meshTypeName = MeshTypeName<MeshType>();

        std::string const integratedShapeFunctionsName =
            "integrated_shape_functions_" + meshTypeName;
        m.def(
            integratedShapeFunctionsName.data(),
            [&](MeshType const& mesh,
                Eigen::Ref<MatrixX const> const& detJe,
                int qorder) -> MatrixX {
                MatrixX R;
                pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
                    if (qorder == QuadratureOrder)
                    {
                        R = pbat::fem::IntegratedShapeFunctions<QuadratureOrder, MeshType>(
                            mesh,
                            detJe);
                    }
                });
                if (R.size() == 0)
                    throw_bad_quad_order(qorder);
                return R;
            },
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("quadrature_order"));
        std::string const shapeFunctionGradientsName = "shape_function_gradients_" + meshTypeName;
        m.def(
            shapeFunctionGradientsName.data(),
            [&](MeshType const& mesh, int qorder) -> MatrixX {
                MatrixX R;
                pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
                    if (qorder == QuadratureOrder)
                    {
                        R = pbat::fem::ShapeFunctionGradients<QuadratureOrder>(mesh);
                    }
                });
                if (R.size() == 0)
                    throw_bad_quad_order(qorder);
                return R;
            },
            pyb::arg("mesh"),
            pyb::arg("quadrature_order"));
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
