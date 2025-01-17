/**
 * \file     is_almost_equal_ulps_test.hpp
 * \mainpage Tests for constexpr almost equal comparison functions with ULPS criterion.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_ALMOST_EQUAL_ULPS_TEST
#define LBT__MATH__IS_ALMOST_EQUAL_ULPS_TEST
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lbt/math/detail/is_almost_equal_ulps.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    template <typename T>
    struct IsAlmostEqualUlpsTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(IsAlmostEqualUlpsTest, FloatingPointDataTypes);

    TYPED_TEST(IsAlmostEqualUlpsTest, closeNumberIsAlmostEqual) {
      std::vector<std::pair<TypeParam,TypeParam>> tests {};
      std::vector<TypeParam> const numbers {-1, 0, 1};
      for (auto const& n: numbers) {
        tests.emplace_back(n, std::nextafter(n, static_cast<TypeParam>(-1)));
        tests.emplace_back(n, std::nextafter(n, static_cast<TypeParam>(1)));
      }
      for (auto const& [a, b]: tests) {
        EXPECT_TRUE(lbt::math::isAlmostEqualUlps(a, b));
      }
    }

    TYPED_TEST(IsAlmostEqualUlpsTest, distantNumberIsNotAlmostEqual) {
      std::vector<std::pair<TypeParam,TypeParam>> tests = { {-1, 1}, {1, -1}, {3, 3.14} };
      for (auto const& [a, b]: tests) {
        EXPECT_FALSE(lbt::math::isAlmostEqualUlps(a, b));
      }
    }

  }
}

#endif // LBT__MATH__IS_ALMOST_EQUAL_ULPS_TEST
