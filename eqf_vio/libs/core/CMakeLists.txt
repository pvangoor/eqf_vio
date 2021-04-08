set(eqf_violib_core_HEADER_FILES
    include/common.h
    include/SO3.h
    include/SE3.h
    include/SOT3.h
)

set(eqf_violib_core_SOURCE_FILES
    src/common.cpp
    src/SO3.cpp
    src/SE3.cpp
    src/SOT3.cpp
)

add_library(eqf_violib_core
${eqf_violib_core_HEADER_FILES}
${eqf_violib_core_SOURCE_FILES}
)

target_include_directories(eqf_violib_core
    PRIVATE include
    PRIVATE ${EIGEN_INCLUDE_DIRS}
    PRIVATE ${EIGEN_INCLUDE_DIR}/unsupported
)

target_link_libraries(eqf_violib_core
    yaml-cpp
)
