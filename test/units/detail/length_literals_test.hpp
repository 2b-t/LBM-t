/**
 * \file     length_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a length
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__LENGTH_LITERALS_TEST
#define LBT__UNITS__LENGTH_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/length.hpp"
#include "lbt/units/detail/length_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using LengthLiteralsHelper = UnitLiteralsHelper<lbt::unit::Length>;

      TEST_P(LengthLiteralsHelper, unitConversion) {
        auto const [length, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(length.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(LengthLiteralsTest, LengthLiteralsHelper, ::testing::Values(
          std::make_pair(0.6_km, 6.0e+2L),
          std::make_pair(3.2_m,  3.2L),
          std::make_pair(9.9_dm, 9.9e-1L),
          std::make_pair(7.3_cm, 7.3e-2L),
          std::make_pair(2.6_mm, 2.6e-3L),
          std::make_pair(8.9_um, 8.9e-6L),
          std::make_pair(2.9_pm, 2.9e-12L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__LENGTH_LITERALS_TEST
