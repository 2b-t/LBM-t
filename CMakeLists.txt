cmake_minimum_required(VERSION 3.1)
project(lbt)

if (CMAKE_COMPILER_IS_GNUCC)
  option(ENABLE_COVERAGE "Enable coverage reporting for GCC/Clang" False)

  if (ENABLE_COVERAGE)
    message(STATUS "Building with coverage reporting for GCC/Clang.")
    add_compile_options(--coverage -O0)
    link_libraries(--coverage)
    #set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  endif()
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17) # The vast majority of code is written for C++17 but certain tests require C++20
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Libraries: VTK, nlohmann_json, OpenMP (optional)
find_package(VTK REQUIRED)
if(NOT VTK_FOUND)
  message(FATAL_ERROR "VTK library not found!")
else()
  message(STATUS "VTK version: " ${VTK_VERSION})
endif()
if(VTK_VERSION VERSION_LESS "8.90.0")
  include(${VTK_USE_FILE})
endif()

find_package(nlohmann_json REQUIRED)
if (NOT nlohmann_json_FOUND)
  message(FATAL_ERROR "nlohmann-json library not found!")
else()
  message(STATUS "nlohmann-json version: " ${nlohmann_json_VERSION})
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP version " ${OpenMP_VERSION} " found!")
else()
  message(WARNING "OpenMP not found: Multi-threading disabled!")
endif()

add_library(lbt_obj OBJECT
  src/common/disclaimer.cpp
  src/common/openmp_manager.cpp
  src/common/timer.cpp
  src/common/vtk_utilities.cpp 
  src/geometry/vtk_import.cpp
)
target_include_directories(lbt_obj PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(lbt 
  $<TARGET_OBJECTS:lbt_obj>
  src/main.cpp
)
target_include_directories(lbt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set(LBT_LIBRARIES ${VTK_LIBRARIES} nlohmann_json::nlohmann_json)
if(OpenMP_CXX_FOUND)
  set(LBT_LIBRARIES ${LBT_LIBRARIES} OpenMP::OpenMP_CXX)
endif()
target_link_libraries(lbt_obj ${LBT_LIBRARIES})
target_link_libraries(lbt ${LBT_LIBRARIES})


# Testing
enable_testing()

include(GoogleTest)
if (NOT GTest_FOUND)
  set(LBT_TEST_LIBRARIES ${LBT_LIBRARIES} gtest pthread)

  add_library(lbt_test_obj OBJECT
    test/common/disclaimer_test.cpp
    test/common/openmp_manager_test.cpp
    test/common/output_utilities_test.cpp
    test/common/stream_manager_test.cpp
    test/common/timer_test.cpp
    test/common/tuple_utilities_test.cpp
    test/common/type_definitions_test.cpp
    test/common/vtk_utilities_test.cpp
    test/continuums/simple_continuum_test.cpp
    test/continuums/vtk_continuum_test.cpp
    test/geometry/vtk_import_test.cpp
    test/lattices/lattices_test.cpp
    test/materials/materials_test.cpp
    test/math/math_test.cpp
    test/populations/aa_population_test.cpp
    test/populations/ab_population_test.cpp
    test/populations/boundaries/guo_test.cpp
    test/populations/boundaries/normal_test.cpp
    test/populations/boundaries/orientation_test.cpp
    test/populations/indexing/aa_pattern_test.cpp
    test/populations/indexing/indexing_test.cpp
    test/simulation/simulation_test.cpp
    test/test_utilities/test_utilities_test.cpp
    test/units/characteristic_numbers_test.cpp
    test/units/literals_test.cpp
    test/units/units_test.cpp
    test/converter_test.cpp
  )
  target_include_directories(lbt_test_obj PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  add_executable(run_lbt_tests
    $<TARGET_OBJECTS:lbt_obj>
    $<TARGET_OBJECTS:lbt_test_obj>
    test/run_tests.cpp
  )

  target_compile_definitions(run_lbt_tests PUBLIC NDEBUG)

  gtest_discover_tests(run_lbt_tests
    TEST_SUFFIX .noArgs
    TEST_LIST noArgsTests
  )
  set_tests_properties(${noArgsTests} PROPERTIES TIMEOUT 10)

  target_link_libraries(lbt_test_obj ${LBT_TEST_LIBRARIES})
  target_link_libraries(run_lbt_tests ${LBT_TEST_LIBRARIES})
else()
  message(FATAL_ERROR "GoogleTest not found!")
endif()
