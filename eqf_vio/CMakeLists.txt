cmake_minimum_required(VERSION 3.0)
project(eqf_vio VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)

# Set off if you want to build the project in pure cmake


include(CMakeDependentOption)

# Import external dependencies

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)


# Options and settings
option( BUILD_WITH_ROS "Build for use with ROS" ON)
cmake_dependent_option( BUILD_TESTS "Build Tests" ON
"NOT BUILD_WITH_ROS" OFF)
cmake_dependent_option( BUILD_VISUALISATION "Build Visualisation Tool" OFF
"NOT BUILD_WITH_ROS" OFF)
cmake_dependent_option( BUILD_GIFT "Build EqF_VIO with GIFT for feature tracking" ON
"NOT BUILD_WITH_ROS" OFF)
# cmake_dependent_option( USE_MARCH_NATIVE "Use the flag -march=native" ON
# "NOT BUILD_WITH_ROS" OFF)
option( USE_MARCH_NATIVE "Use the flag -march=native" ON)


set(DEFAULT_CONFIG_FILE "EQF_VIO_config_template.yaml")
get_filename_component(DEFAULT_CONFIG_FILE ${DEFAULT_CONFIG_FILE} ABSOLUTE)
add_compile_definitions(DEFAULT_CONFIG_FILE="${DEFAULT_CONFIG_FILE}")
add_compile_definitions(BUILD_VISUALISATION=$<BOOL:${BUILD_VISUALISATION}>)

option( USE_MARCH_NATIVE "Use the flag -march=native" ON)
if (USE_MARCH_NATIVE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message("march=native was requested but is not supported.")
    endif()
endif()


# Add libs
add_subdirectory(libs)

# Add the include files
set(EQF_VIO_HEADER_FILES
    include/eqf_vio/VIOState.h
    include/eqf_vio/IMUVelocity.h
    include/eqf_vio/VIOGroup.h
    include/eqf_vio/VIOFilter.h
    include/eqf_vio/VIOFilterSettings.h
    include/eqf_vio/CSVReader.h
    include/eqf_vio/VisionMeasurement.h
    include/eqf_vio/EqFMatrices.h
    )
set(EQF_VIO_SOURCE_FILES
    src/VIOState.cpp
    src/VIOGroup.cpp
    src/VIOFilter.cpp
    src/VisionMeasurement.cpp
    src/IMUVelocity.cpp
    src/EqFMatrices.cpp
)

# Add the EqF_VIO library

add_library(eqf_vio_lib
    ${EQF_VIO_SOURCE_FILES}
    ${EQF_VIO_HEADER_FILES}
)

message("This is being included!")
message(${MODULE_INCLUDE_DIRS})

target_include_directories(eqf_vio_lib
    PUBLIC include
    PUBLIC ${MODULE_INCLUDE_DIRS}
    PUBLIC ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(eqf_vio_lib
    ${MODULE_LIBS}
)


# Add the tests

if(BUILD_TESTS)
    enable_testing()
    message("Building the tests.")
    add_subdirectory(test)
endif()

# Add the main executable
if(NOT BUILD_WITH_ROS)
    add_executable(eqf_vio src/main.cpp)
    set(EXECUTABLES_LIST
        eqf_vio
    )

    if (BUILD_GIFT)
        find_package(OpenCV REQUIRED)
        find_package(GIFT REQUIRED)
        add_executable(eqf_vio_gift src/main_gift.cpp)
        list(APPEND EXECUTABLES_LIST eqf_vio_gift)
        target_link_libraries(eqf_vio_gift
            GIFT
            ${OpenCV_LIBS}
        )
        target_include_directories(eqf_vio_gift
            PRIVATE GIFT::GIFT
            PRIVATE ${OpenCV_INCLUDE_DIRS}
        )
    endif()

    foreach(EXECUTABLE_NAME ${EXECUTABLES_LIST})
        target_include_directories(${EXECUTABLE_NAME}
            PRIVATE include
            PRIVATE ${EIGEN_INCLUDE_DIRS}
            PRIVATE ${YAML_CPP_INCLUDE_DIR}
        )
        target_link_libraries(${EXECUTABLE_NAME}
            eqf_vio_lib
            ${MODULE_LIBS}
            yaml-cpp
        )
    endforeach()

endif()

# Build the ROS package
if(BUILD_WITH_ROS)
    find_package(catkin REQUIRED)
    
    catkin_package(
     INCLUDE_DIRS include
     LIBRARIES eqf_vio_lib
    #  CATKIN_DEPENDS other_catkin_pkg
     DEPENDS EIGEN YAML_CPP OPENCV GIFT
    )

endif()