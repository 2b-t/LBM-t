/**
 * \file     pressure_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a pressure
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__PRESSURE_LITERALS_TEST
#define LBT__UNITS__PRESSURE_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/pressure.hpp"
#include "lbt/units/detail/pressure_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using PressureLiteralsHelper = UnitLiteralsHelper<lbt::unit::Pressure>;

      TEST_P(PressureLiteralsHelper, unitConversion) {
        auto const [pressure, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(pressure.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(PressureLiteralsTest, PressureLiteralsHelper, ::testing::Values(
          std::make_pair(0.3_Pa,  0.3L),
          std::make_pair(3.2_GPa, 3.2e+9L),
          std::make_pair(3.4_mPa, 3.4e-3L),
          std::make_pair(6.3_uPa, 6.3e-6L),
          std::make_pair(2.3_hPa, 2.3e+2L),
          std::make_pair(1.2_bar, 1.2e+5L),
          std::make_pair(1.4_atm, 141855L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__PRESSURE_LITERALS_TEST
