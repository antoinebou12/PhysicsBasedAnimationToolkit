# Add the Pbf module
target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Pbf.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Pbf.cpp"
)