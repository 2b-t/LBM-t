/**
 * \file     temperature_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a temperature
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__TEMPERATURE_LITERALS_TEST
#define LBT__UNITS__TEMPERATURE_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/temperature.hpp"
#include "lbt/units/detail/temperature_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using TemperatureLiteralsHelper = UnitLiteralsHelper<lbt::unit::Temperature>;

      TEST_P(TemperatureLiteralsHelper, unitConversion) {
        auto const [temperature, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(temperature.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(TemperatureLiteralsTest, TemperatureLiteralsHelper, ::testing::Values(
          std::make_pair(315.3_K,  315.3L),
          std::make_pair(20.4_deg, 293.55L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__TEMPERATURE_LITERALS_TEST
