/**
 * \file     time_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a time
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__TIME_LITERALS_TEST
#define LBT__UNITS__TIME_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/time.hpp"
#include "lbt/units/detail/time_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using TimeLiteralsHelper = UnitLiteralsHelper<lbt::unit::Time>;

      TEST_P(TimeLiteralsHelper, unitConversion) {
        auto const [time, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(time.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(TimeLiteralsTest, TimeLiteralsHelper, ::testing::Values(
          std::make_pair(0.8_d,   6.912e+4L),
          std::make_pair(1.3_h,   4.680e+3L),
          std::make_pair(6.9_min, 4.14e+2L),
          std::make_pair(5.6_s,   5.6L),
          std::make_pair(1.4_ms,  1.4e-3L),
          std::make_pair(2.9_us,  2.9e-6L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__TIME_LITERALS_TEST
