/**
 * \file     mass_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a mass
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__MASS_LITERALS_TEST
#define LBT__UNITS__MASS_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/mass.hpp"
#include "lbt/units/detail/mass_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using MassLiteralsHelper = UnitLiteralsHelper<lbt::unit::Mass>;

      TEST_P(MassLiteralsHelper, unitConversion) {
        auto const [mass, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(mass.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(MassLiteralsTest, MassLiteralsHelper, ::testing::Values(
          std::make_pair(9.9_t,  9.9e+3L),
          std::make_pair(0.7_kg, 7.0e-1L),
          std::make_pair(3.8_g,  3.8e-3L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__MASS_LITERALS_TEST
