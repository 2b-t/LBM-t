/**
 * \file     velocity_literals_test.hpp
 * \mainpage Contains unit-tests for literals for a velocity
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__VELOCITY_LITERALS_TEST
#define LBT__UNITS__VELOCITY_LITERALS_TEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "lbt/units/detail/velocity.hpp"
#include "lbt/units/detail/velocity_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      using VelocityLiteralsHelper = UnitLiteralsHelper<lbt::unit::Velocity>;

      TEST_P(VelocityLiteralsHelper, unitConversion) {
        auto const [velocity, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(velocity.get(), expected_result);
      }

      INSTANTIATE_TEST_SUITE_P(VelocityLiteralsTest, VelocityLiteralsHelper, ::testing::Values(
          std::make_pair(9.0_kmph, 2.5L),
          std::make_pair(1.7_mps,  1.7L),
          std::make_pair(8.9_cmps, 8.9e-2L),
          std::make_pair(5.3_mmps, 5.3e-3L)
        )
      );

    }
  }
}

#endif // LBT__UNITS__VELOCITY_LITERALS_TEST
