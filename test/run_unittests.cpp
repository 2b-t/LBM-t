/**\file     run_unittests.cpp
 * \mainpage Performs all unit tests for LB-t
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#include <gtest/gtest.h>

#include "continuum/characteristic_numbers_unittest.hpp"
#include "continuum/continuum_unittest.hpp"
#include "general/constexpr_math_unittest.hpp"
#include "general/disclaimer_unittest.hpp"
#include "general/literals_unittest.hpp"
#include "general/openmp_manager_unittest.hpp"
#include "general/output_utilities_unittest.hpp"
#include "general/stream_manager_unittest.hpp"
#include "general/timer_unittest.hpp"
#include "general/tuple_utilities_unittest.hpp"
#include "general/units_unittest.hpp"
#include "general/vtk_utilities_unittest.hpp"
#include "geometry/vtk_import_unittest.hpp"
#include "lattice/lattice_unittest.hpp"
#include "population/indexing/aa_pattern_unittest.hpp"
#include "population/indexing/indexing_unittest.hpp"
#include "testing_utilities/testing_utilities_unittest.hpp"
#include "simulation_unittest.hpp"


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
