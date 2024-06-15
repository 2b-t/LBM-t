/**
 * \file     constants_test.hpp
 * \mainpage Tests for constexpr mathematical constants
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__CONSTANTS_TEST
#define LBT__MATH__CONSTANTS_TEST
#pragma once

#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>

#include "lbt/math/detail/constants.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    template <typename T>
    struct PiConstantTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(PiConstantTest, FloatingPointDataTypes);

    TYPED_TEST(PiConstantTest, equalToStdPi) {
      auto const result {lbt::math::pi<TypeParam>};
      auto const expected_result {static_cast<TypeParam>(M_PI)};

      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(result, expected_result);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(result, expected_result);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

    template <typename T>
    struct EulerNumberConstantTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(EulerNumberConstantTest, FloatingPointDataTypes);

    TYPED_TEST(EulerNumberConstantTest, equalToStdEulerNumber) {
      auto const result {lbt::math::e<TypeParam>};
      auto const expected_result {static_cast<TypeParam>(std::exp(1.0))};

      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(result, expected_result);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(result, expected_result);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

    template <typename T>
    struct LogarithmOfTwoConstantTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(LogarithmOfTwoConstantTest, FloatingPointDataTypes);

    TYPED_TEST(LogarithmOfTwoConstantTest, equalToStdLogarithmOfTwo) {
      auto const result {lbt::math::ln2<TypeParam>};
      auto const expected_result {static_cast<TypeParam>(M_LN2)};

      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(result, expected_result);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(result, expected_result);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

  }
}

#endif // LBT__MATH__CONSTANTS_TEST
