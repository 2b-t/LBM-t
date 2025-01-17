/**
 * \file     is_inf_test.hpp
 * \mainpage Tests for functions checking whether a certain value corresponds to infinity or not
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_INF_TEST
#define LBT__MATH__IS_INF_TEST
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "lbt/math/detail/is_inf.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    /// Test function for detecting positive infinity
    template <typename T>
    struct IsPosInfTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(IsPosInfTest, FloatingPointDataTypes);

    TYPED_TEST(IsPosInfTest, positiveInfinityIsPositiveInfinity) {
      constexpr auto pos_inf {std::numeric_limits<TypeParam>::infinity()};
      EXPECT_TRUE(lbt::math::isPosInf(pos_inf));
      EXPECT_TRUE(lbt::math::isPosInf(pos_inf) == std::isinf(pos_inf));
    }

    TYPED_TEST(IsPosInfTest, negativeInfinityIsNotPositiveInfinity) {
      constexpr auto neg_inf {-std::numeric_limits<TypeParam>::infinity()};
      EXPECT_FALSE(lbt::math::isPosInf(neg_inf));
    }

    TYPED_TEST(IsPosInfTest, positiveNumberIsNotPositiveInfinity) {
      std::vector<TypeParam> const positive_numbers {+0, 1, 100, std::numeric_limits<TypeParam>::max()};
      for (auto const& n: positive_numbers) {
        EXPECT_FALSE(lbt::math::isPosInf(n));
        EXPECT_EQ(lbt::math::isPosInf(n), std::isinf(n));
      }
    }

    TYPED_TEST(IsPosInfTest, negativeNumberIsNotPositiveInfinity) {
      std::vector<TypeParam> const negative_numbers {-0, -1, -100, std::numeric_limits<TypeParam>::min()};
      for (auto const& n: negative_numbers) {
        EXPECT_FALSE(lbt::math::isPosInf(n));
        EXPECT_EQ(lbt::math::isPosInf(n), std::isinf(n));
      }
    }

    /// Test function for detecting negative infinity
    template <typename T>
    struct IsNegInfTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(IsNegInfTest, FloatingPointDataTypes);

    TYPED_TEST(IsNegInfTest, negativeInfinityIsNegativeInfinity) {
      constexpr auto neg_inf {-std::numeric_limits<TypeParam>::infinity()};
      EXPECT_TRUE(lbt::math::isNegInf(neg_inf));
      EXPECT_EQ(lbt::math::isNegInf(neg_inf), std::isinf(neg_inf));
    }

    TYPED_TEST(IsNegInfTest, positiveInfinityIsNotNegativeInfinity) {
      constexpr auto pos_inf {std::numeric_limits<TypeParam>::infinity()};
      EXPECT_FALSE(lbt::math::isNegInf(pos_inf));
    }

    TYPED_TEST(IsNegInfTest, positiveNumberIsNotNegativeInfinity) {
      std::vector<TypeParam> const positive_numbers {+0, 1, 100, std::numeric_limits<TypeParam>::max()};
      for (auto const& n: positive_numbers) {
        EXPECT_FALSE(lbt::math::isNegInf(n));
        EXPECT_TRUE(lbt::math::isPosInf(n) == std::isinf(n));
      }
    }

    TYPED_TEST(IsNegInfTest, negativeNumberIsNotNegativeInfinity) {
      std::vector<TypeParam> const negative_numbers {-0, -1, -100, std::numeric_limits<TypeParam>::min()};
      for (auto const& n: negative_numbers) {
        EXPECT_FALSE(lbt::math::isNegInf(n));
        EXPECT_TRUE(lbt::math::isPosInf(n) == std::isinf(n));
      }
    }

  }
}

#endif // LBT__MATH__IS_INF_TEST
