#ifndef PBAT_GPU_COMMON_MORTON_CUH
#define PBAT_GPU_COMMON_MORTON_CUH

#include <cstddef>
#include <cuda/std/cmath>
#include <pbat/gpu/Aliases.h>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace common {

using MortonCodeType = cuda::std::uint32_t;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ MortonCodeType ExpandBits(MortonCodeType v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ MortonCodeType Morton3D(std::array<GpuScalar, 3> x)
{
    static_assert(
        std::is_same_v<GpuScalar, float>,
        "Morton code only supported for single precision floating point numbers");
    using namespace cuda::std;
    x[0]              = fminf(fmaxf(x[0] * 1024.0f, 0.0f), 1023.0f);
    x[1]              = fminf(fmaxf(x[1] * 1024.0f, 0.0f), 1023.0f);
    x[2]              = fminf(fmaxf(x[2] * 1024.0f, 0.0f), 1023.0f);
    MortonCodeType xx = ExpandBits(static_cast<MortonCodeType>(x[0]));
    MortonCodeType yy = ExpandBits(static_cast<MortonCodeType>(x[1]));
    MortonCodeType zz = ExpandBits(static_cast<MortonCodeType>(x[2]));
    return xx * 4 + yy * 2 + zz;
}

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_MORTON_CUH