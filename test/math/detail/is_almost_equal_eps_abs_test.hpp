/**
 * \file     is_almost_equal_eps_abs_test.hpp
 * \mainpage Tests for almost equal comparison of floating-point numbers with an absolute epsilon
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS_TEST
#define LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS_TEST
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lbt/math/detail/is_almost_equal_eps_abs.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    template <typename T>
    struct IsAlmostEqualEpsAbsTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(IsAlmostEqualEpsAbsTest, FloatingPointDataTypes);

    TYPED_TEST(IsAlmostEqualEpsAbsTest, closeNumberIsAlmostEqual) {
      std::vector<std::pair<TypeParam,TypeParam>> tests {};
      std::vector<TypeParam> const numbers {-1, 0, 1};
      for (auto const& n: numbers) {
        tests.emplace_back(n, std::nextafter(n, static_cast<TypeParam>(-1)));
        tests.emplace_back(n, std::nextafter(n, static_cast<TypeParam>(1)));
      }
      for (auto const& [a, b]: tests) {
        EXPECT_TRUE(lbt::math::isAlmostEqualEpsAbs(a, b));
      }
    }

    TYPED_TEST(IsAlmostEqualEpsAbsTest, distantNumberIsNotAlmostEqual) {
      std::vector<std::pair<TypeParam,TypeParam>> tests = { {-1, 1}, {1, -1}, {3, 3.14} };
      for (auto const& [a, b]: tests) {
        EXPECT_FALSE(lbt::math::isAlmostEqualEpsAbs(a, b));
      }
    }

  }
}

#endif // LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS_TEST
