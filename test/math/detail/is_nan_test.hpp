/**
 * \file     is_inf_test.hpp
 * \mainpage Tests for functions checking whether a certain value corresponds to NaN or not
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_NAN_TEST
#define LBT__MATH__IS_NAN_TEST
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "lbt/math/detail/is_nan.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    /// Test function for detecting Nan values
    template <typename T>
    struct IsNanTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(IsNanTest, FloatingPointDataTypes);

    TYPED_TEST(IsNanTest, signalingNanIsNan) {
      constexpr auto nan {std::numeric_limits<TypeParam>::signaling_NaN()};
      EXPECT_TRUE(lbt::math::isNan(nan));
      EXPECT_TRUE(lbt::math::isNan(nan) == std::isnan(nan));
    }

    TYPED_TEST(IsNanTest, quietNanIsNan) {
      constexpr auto nan {std::numeric_limits<TypeParam>::quiet_NaN()};
      EXPECT_TRUE(lbt::math::isNan(nan));
      EXPECT_TRUE(lbt::math::isNan(nan) == std::isnan(nan));
    }

    TYPED_TEST(IsNanTest, positiveNumberIsNotNan) {
      std::vector<TypeParam> const positive_numbers {+0, 1, 100, std::numeric_limits<TypeParam>::max()};
      for (auto const& n: positive_numbers) {
        EXPECT_FALSE(lbt::math::isNan(n));
        EXPECT_TRUE(lbt::math::isNan(n) == std::isnan(n));
      }
    }

    TYPED_TEST(IsNanTest, negativeNumberIsNotNan) {
      std::vector<TypeParam> const negative_numbers {-0, -1, -100, std::numeric_limits<TypeParam>::min()};
      for (auto const& n: negative_numbers) {
        EXPECT_FALSE(lbt::math::isNan(n));
        EXPECT_TRUE(lbt::math::isNan(n) == std::isnan(n));
      }
    }

  }
}

#endif // LBT__MATH__IS_NAN_TEST
