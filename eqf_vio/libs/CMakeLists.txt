
# The order of modules is important!
set(eqf_violib_dirs
    core
)
if (BUILD_VISUALISATION)
    list(APPEND eqf_violib_dirs visualisation)
endif()

set(MODULE_INCLUDE_DIRS "")
set(MODULE_LIBS "")

foreach(module ${eqf_violib_dirs})
    add_subdirectory(${module})
    list(APPEND MODULE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${module}/include")
    list(APPEND MODULE_LIBS "eqf_violib_${module}")
endforeach()

set(MODULE_INCLUDE_DIRS ${MODULE_INCLUDE_DIRS} PARENT_SCOPE)
set(MODULE_LIBS ${MODULE_LIBS} PARENT_SCOPE)
