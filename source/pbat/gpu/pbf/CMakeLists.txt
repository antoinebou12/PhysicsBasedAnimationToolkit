target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "HashGrid.cuh"
    "HashGrid.h"
    "Kernel.cuh"
    "Kernel.h"
    "Particle.h"
    "Pbf.h"
    "PbfImpl.cuh"
    "PbfKernels.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Pbf.cu"
    "PbfImpl.cu"
)