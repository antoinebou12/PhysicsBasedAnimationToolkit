target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "HashGrid.h"
    "Kernel.h"
    "Pbf.h"
    "Particle.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "HashGrid.cpp"
    "Kernel.cpp"
    "Pbf.cpp"
)