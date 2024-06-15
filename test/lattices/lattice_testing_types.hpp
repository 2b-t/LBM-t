/**
 * \file     lattice_testing_types.hpp
 * \mainpage Classes and type definitions for generating all the relevant lattices to be tested
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__TEST__LATTICES__LATTICE_TESTING_TYPES
#define LBT__TEST__LATTICES__LATTICE_TESTING_TYPES
#pragma once

#include <tuple>

#include <gtest/gtest.h>

#include "lbt/common/tuple_utilities.hpp"
#include "lbt/lattices/D2Q9.hpp"
#include "lbt/lattices/D3Q15.hpp"
#include "lbt/lattices/D3Q19.hpp"
#include "lbt/lattices/D3Q27.hpp"
#include "../test_utilities/test_utilities.hpp"


namespace lbt {
  namespace test {

    /// The different lattices to be tested
    template <typename T> using LatticeTypes2D = std::tuple<lbt::lattice::D2Q9P10<T>,  lbt::lattice::D2Q9P12<T>>;
    template <typename T> using LatticeTypes3D = std::tuple<lbt::lattice::D3Q15P16<T>, lbt::lattice::D3Q19P20<T>, 
                                                            lbt::lattice::D3Q27P28<T>, lbt::lattice::D3Q27PC<T>>;
    template <typename T> using LatticeTypes = decltype(std::tuple_cat(std::declval<LatticeTypes2D<T>>(), 
                                                                       std::declval<LatticeTypes3D<T>>()));

    /// The different floating data types for the lattices
    using LatticeFloatingTypes = std::tuple<double, float>;

    // Merge data types and lattices data types
    using LatticeTestTypes2D = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes2D, LatticeFloatingTypes>>;
    using LatticeTestTypes3D = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes3D, LatticeFloatingTypes>>;
    using LatticeTestTypes = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes, LatticeFloatingTypes>>;
    
  }
}

#endif // LBT__TEST__LATTICES__LATTICE_TESTING_TYPES
