#ifndef PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
#define PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH

#include "BvhQueryImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Morton.cuh"
#include "pbat/gpu/common/Queue.cuh"
#include "pbat/gpu/common/Stack.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>

namespace pbat {
namespace gpu {
namespace geometry {
namespace BvhQueryImplKernels {

struct FComputeAabb
{
    __device__ void operator()(int s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            b[d][s] = x[d][inds[0][s]];
            e[d][s] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][s] = min(b[d][s], x[d][inds[m][s]]);
                e[d][s] = max(e[d][s], x[d][inds[m][s]]);
            }
            b[d][s] -= r;
            e[d][s] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = common::MortonCodeType;

    __device__ void operator()(int s)
    {
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{};
        for (auto d = 0; d < 3; ++d)
        {
            auto cd = GpuScalar{0.5} * (b[d][s] + e[d][s]);
            c[d]    = (cd - sb[d]) / sbe[d];
        }
        morton[s] = common::Morton3D(c);
    }

    std::array<GpuScalar, 3> sb;
    std::array<GpuScalar, 3> sbe;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    MortonCodeType* morton;
};

struct FDetectOverlaps
{
    using OverlapType = typename BvhQueryImpl::OverlapType;

    __device__ bool AreSimplicesTopologicallyAdjacent(GpuIndex si, GpuIndex sj) const
    {
        auto count{0};
        for (auto i = 0; i < queryInds.size(); ++i)
            for (auto j = 0; j < inds.size(); ++j)
                count += (queryInds[i][si] == inds[j][sj]);
        return count > 0;
    }

    __device__ bool AreBoxesOverlapping(GpuIndex i, GpuIndex j) const
    {
        // clang-format off
        return (queryE[0][i] >= b[0][j]) and (queryB[0][i] <= e[0][j]) and
               (queryE[1][i] >= b[1][j]) and (queryB[1][i] <= e[1][j]) and
               (queryE[2][i] >= b[2][j]) and (queryB[2][i] <= e[2][j]);
        // clang-format on
    }

    __device__ void operator()(auto query)
    {
        // Traverse nodes depth-first starting from the root.
        common::Stack<GpuIndex, 64> stack{};
        stack.Push(0);
        do
        {
            assert(not stack.IsFull());
            GpuIndex const node = stack.Pop();
            // Check each child node for overlap.
            GpuIndex const lc            = child[0][node];
            GpuIndex const rc            = child[1][node];
            bool const bLeftBoxOverlaps  = AreBoxesOverlapping(query, lc);
            bool const bRightBoxOverlaps = AreBoxesOverlapping(query, rc);

            // Query overlaps another leaf node -> report collision if topologically separate
            // simplices
            bool const bIsLeftLeaf = lc >= leafBegin;
            if (bLeftBoxOverlaps and bIsLeftLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[lc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(
                        si,
                        sj) /* and AreSimplicesOverlapping(si, sj) */
                    and not overlaps.Append({si, sj}))
                    break;
            }
            bool const bIsRightLeaf = rc >= leafBegin;
            if (bRightBoxOverlaps and bIsRightLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[rc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(
                        si,
                        sj) /* and AreSimplicesOverlapping(si, sj) */
                    and not overlaps.Append({si, sj}))
                    break;
            }

            // Query overlaps an internal node -> traverse.
            bool const bTraverseLeft  = bLeftBoxOverlaps and not bIsLeftLeaf;
            bool const bTraverseRight = bRightBoxOverlaps and not bIsRightLeaf;
            if (bTraverseLeft)
                stack.Push(lc);
            if (bTraverseRight)
                stack.Push(rc);
        } while (not stack.IsEmpty());
    }

    std::array<GpuScalar const*, 3> x;

    GpuIndex* querySimplex;
    std::array<GpuIndex const*, 4> queryInds;
    std::array<GpuScalar*, 3> queryB;
    std::array<GpuScalar*, 3> queryE;

    GpuIndex const* simplex;
    std::array<GpuIndex const*, 4> inds;
    std::array<GpuScalar const*, 3> b;
    std::array<GpuScalar const*, 3> e;
    std::array<GpuIndex const*, 2> child;
    GpuIndex leafBegin;

    common::DeviceSynchronizedList<OverlapType> overlaps;
};

struct FContactPairs
{
    using OverlapType              = typename BvhQueryImpl::OverlapType;
    using NearestNeighbourPairType = typename BvhQueryImpl::NearestNeighbourPairType;
    using Vector3                  = pbat::gpu::math::linalg::Matrix<GpuScalar, 3>;

    __device__ Vector3 Position(auto v) const
    {
        Vector3 P{};
        P(0) = x[0][v];
        P(1) = x[1][v];
        P(2) = x[2][v];
        return P;
    }
    __device__ Vector3 Lower(auto s) const
    {
        Vector3 P{};
        P(0) = b[0][s];
        P(1) = b[1][s];
        P(2) = b[2][s];
        return P;
    }
    __device__ Vector3 Upper(auto s) const
    {
        Vector3 P{};
        P(0) = e[0][s];
        P(1) = e[1][s];
        P(2) = e[2][s];
        return P;
    }
    __device__ GpuScalar MinDistance(Vector3 const& X, Vector3 const& L, Vector3 const& U) const
    {
        using namespace pbat::gpu::math::linalg;
        Vector3 const DX = Min(U, Max(L, X)) - X;
        return SquaredNorm(DX);
    }
    __device__ GpuScalar MinMaxDistance(Vector3 const& X, Vector3 const& L, Vector3 const& U) const
    {
        using namespace pbat::gpu::math::linalg;
        Vector3 const DXL = Squared(L - X);
        Vector3 const DXU = Squared(U - X);
        Vector3 const rm  = Min(DXL, DXU);
        Vector3 const rM  = Max(DXL, DXU);
        std::array<GpuScalar, 3> const d{
            rm(0) + rM(1) + rM(2),
            rM(0) + rm(1) + rM(2),
            rM(0) + rM(1) + rm(2),
        };
        return min(d[0], min(d[1], d[2]));
    }
    __device__ GpuScalar Distance(Vector3 const& P, GpuIndex s) const
    {
        using namespace pbat::gpu::math::linalg;
        Matrix<GpuScalar, 3, 3> ABC;
        auto A = ABC.Col(0);
        auto B = ABC.Col(1);
        auto C = ABC.Col(2);
        A      = Position(targetInds[0][s]);
        B      = Position(targetInds[1][s]);
        C      = Position(targetInds[2][s]);
        Matrix<GpuScalar, 3> uvw;
        uvw.SetZero();

        /**
         * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.5
         */

        // Check if P in vertex region outside A
        Vector3 const AB                     = B - A;
        Vector3 const AC                     = C - A;
        Vector3 const AP                     = P - A;
        GpuScalar const d1                   = Dot(AB, AP);
        GpuScalar const d2                   = Dot(AC, AP);
        bool const bIsInVertexRegionOutsideA = d1 <= GpuScalar{0} and d2 <= GpuScalar{0};
        if (bIsInVertexRegionOutsideA)
        {
            uvw(0) = GpuScalar{1}; // barycentric coordinates (1,0,0)
        }

        // Check if P in vertex region outside B
        Vector3 const BP                     = P - B;
        GpuScalar const d3                   = Dot(AB, BP);
        GpuScalar const d4                   = Dot(AC, BP);
        bool const bIsInVertexRegionOutsideB = d3 >= GpuScalar{0} and d4 <= d3;
        if (bIsInVertexRegionOutsideB)
        {
            uvw(1) = GpuScalar{1}; // barycentric coordinates (0,1,0)
        }

        // Check if P in edge region of AB, if so return projection of P onto AB
        GpuScalar const vc = d1 * d4 - d3 * d2;
        bool const bIsInEdgeRegionOfAB =
            vc <= GpuScalar{0} and d1 >= GpuScalar{0} and d3 <= GpuScalar{0};
        if (bIsInEdgeRegionOfAB)
        {
            GpuScalar const v = d1 / (d1 - d3);
            uvw(0)            = GpuScalar{1} - v;
            uvw(1)            = v; // barycentric coordinates (1-v,v,0)
        }

        // Check if P in vertex region outside C
        Vector3 const CP                     = P - C;
        GpuScalar const d5                   = Dot(AB, CP);
        GpuScalar const d6                   = Dot(AC, CP);
        bool const bIsInVertexRegionOutsideC = d6 >= GpuScalar{0} and d5 <= d6;
        if (bIsInVertexRegionOutsideC)
        {
            uvw(2) = GpuScalar{1}; // barycentric coordinates (0,0,1)
        }

        // Check if P in edge region of AC, if so return projection of P onto AC
        GpuScalar const vb = d5 * d2 - d1 * d6;
        bool const bIsInEdgeRegionOfAC =
            (vb <= GpuScalar{0} and d2 >= GpuScalar{0} and d6 <= GpuScalar{0});
        if (bIsInEdgeRegionOfAC)
        {
            GpuScalar const w = d2 / (d2 - d6);
            uvw(0)            = GpuScalar{1} - w; // barycentric coordinates (1-w,0,w)
            uvw(2)            = w;
        }
        // Check if P in edge region of BC, if so return projection of P onto BC
        GpuScalar const va = d3 * d6 - d5 * d4;
        bool const bIsInEdgeRegionOfBC =
            va <= GpuScalar{0} and (d4 - d3) >= GpuScalar{0} and (d5 - d6) >= GpuScalar{0};
        if (bIsInEdgeRegionOfBC)
        {
            GpuScalar const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            // barycentric coordinates (0,1-w,w)
            uvw(1) = GpuScalar{1} - w;
            uvw(2) = w;
        }
        // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
        if (not(bIsInVertexRegionOutsideA or bIsInVertexRegionOutsideB or
                bIsInVertexRegionOutsideC or bIsInEdgeRegionOfAB or bIsInEdgeRegionOfAC or
                bIsInEdgeRegionOfBC))
        {
            GpuScalar const denom = GpuScalar{1} / (va + vb + vc);
            uvw(1)                = vb * denom;
            uvw(2)                = vc * denom;
            uvw(0)                = GpuScalar{1} - uvw(1) - uvw(2);
        }
        return SquaredNorm(P - ABC * uvw);
    }

    struct BoxOrSimplex
    {
        GpuIndex node; ///< BVH node index
        GpuScalar d;   ///< Distance between query and node
    };

    template <class FIsLeaf, class FDistance>
    struct BranchAndBound
    {
        __device__ BranchAndBound(
            Vector3 X,
            GpuScalar R,
            FIsLeaf fIsLeafIn,
            FDistance fDistanceIn,
            GpuScalar dzero)
            : stack{},
              nearest{},
              X(X),
              R(R),
              fIsLeaf(std::forward<FIsLeaf>(fIsLeafIn)),
              fDistance(std::forward<FDistance>(fDistanceIn)),
              dzero(dzero)
        {
        }

        __device__ void Push(GpuIndex c, GpuScalar dbox)
        {
            bool const bIsLeaf = fIsLeaf(c);
            if (bIsLeaf)
            {
                GpuScalar d = fDistance(X, c);
                if (d < R)
                {
                    nearest.Clear();
                    nearest.Push(c);
                    R = d;
                }
                else if (d - R <= dzero and not nearest.IsFull())
                {
                    nearest.Push(c);
                }
            }
            else
            {
                stack.Push({c, dbox});
            }
        }

        __device__ BoxOrSimplex Pop()
        {
            BoxOrSimplex bos = stack.Top();
            stack.Pop();
            return bos;
        }

        common::Stack<BoxOrSimplex, 64> stack;
        common::Queue<GpuIndex, 8> nearest;
        Vector3 X;
        GpuScalar R;
        FIsLeaf const fIsLeaf;
        FDistance const fDistance;
        GpuScalar dzero;
    };

    __device__ void operator()(OverlapType const& o)
    {
        // WARNING: Self contacts not supported yet
        if (queryBodies[o.first] == targetBodies[o.second])
            return;

        // Branch and bound needs to know which nodes are leaves, and what the cost function to
        // minimize is.
        auto const fIsLeaf = [this](GpuIndex c) {
            return c >= leafBegin;
        };
        auto const fDistance = [this](Vector3 const& X, GpuIndex c) {
            return Distance(X, simplex[c - leafBegin]);
        };
        using FIsLeaf   = decltype(fIsLeaf);
        using FDistance = decltype(fDistance);

        // Branch and bound over BVH
        GpuIndex const v = queryInds[0][o.first];
        BranchAndBound<FIsLeaf, FDistance> traversal{Position(v), R, fIsLeaf, fDistance, dzero};
        traversal.Push(0, MinDistance(traversal.X, Lower(0), Upper(0)));
        do
        {
            assert(not traversal.stack.IsFull());
            BoxOrSimplex const bos = traversal.Pop();
            if (bos.d > traversal.R)
                continue;

            GpuIndex const lc = child[0][bos.node];
            GpuIndex const rc = child[1][bos.node];

            Vector3 const LL = Lower(lc);
            Vector3 const LU = Upper(lc);
            Vector3 const RL = Lower(rc);
            Vector3 const RU = Upper(rc);

            GpuScalar Ldmin    = MinDistance(traversal.X, LL, LU);
            GpuScalar Rdmin    = MinDistance(traversal.X, RL, RU);
            GpuScalar Ldminmax = MinMaxDistance(traversal.X, LL, LU);
            GpuScalar Rdminmax = MinMaxDistance(traversal.X, RL, RU);

            if (Ldmin <= Rdminmax)
                traversal.Push(lc, Ldmin);
            if (Rdmin <= Ldminmax)
                traversal.Push(rc, Rdmin);
        } while (not traversal.stack.IsEmpty());
        // Collect results
        while (not traversal.nearest.IsEmpty())
        {
            GpuIndex leaf = traversal.nearest.Top();
            GpuIndex s    = simplex[leaf - leafBegin];
            neighbours.Append({v, s});
            traversal.nearest.Pop();
        }
    }

    std::array<GpuScalar const*, 3> x;

    GpuIndex const* queryBodies;              ///< Body indices of query simplices
    std::array<GpuIndex const*, 4> queryInds; ///< Vertex indices of query simplices
    GpuScalar R;                              ///< Nearest neighbour query search radius
    GpuScalar dzero; ///< Error tolerance for different distances to be considered the same, i.e. if
                     ///< the distances di,dj to the nearest neighbours i,j are similar, we
                     ///< considered both i and j to be nearest neighbours.

    GpuIndex const* targetBodies;              ///< Body indices of target simplices
    std::array<GpuIndex const*, 4> targetInds; ///< Vertex indices of target simplices
    GpuIndex const* simplex;                   ///< Target simplices
    std::array<GpuScalar const*, 3> b;         ///< Box beginnings of BVH
    std::array<GpuScalar const*, 3> e;         ///< Box endings of BVH
    std::array<GpuIndex const*, 2> child;      ///< BVH children
    GpuIndex leafBegin;                        ///< Index to beginning of BVH's leaves array

    common::DeviceSynchronizedList<NearestNeighbourPairType>
        neighbours; ///< Nearest neighbour pairs found
};

} // namespace BvhQueryImplKernels
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
