/**
 * \file     exp_test.hpp
 * \mainpage Tests for constexpr exponential function
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__EXP_TEST
#define LBT__MATH__EXP_TEST
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lbt/math/detail/exp.hpp"
#include "lbt/math/detail/constants.hpp"
#include "testing_types.hpp"


namespace lbt {
  namespace test {

    /// Tests for exponential function
    template <typename T>
    struct ExpTest: public ::testing::Test {
    };

    TYPED_TEST_SUITE(ExpTest, FloatingPointDataTypes);

    TYPED_TEST(ExpTest, zeroIsUnity) {
      constexpr auto zero {static_cast<TypeParam>(0.0)};
      constexpr auto unity {static_cast<TypeParam>(1.0)};
      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(lbt::math::exp(zero), unity);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(lbt::math::exp(zero), unity);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

    TYPED_TEST(ExpTest, negativeInfinityIsZero) {
      constexpr auto inf {-std::numeric_limits<TypeParam>::infinity()};
      constexpr auto zero {static_cast<TypeParam>(0.0)};
      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(lbt::math::exp(inf), zero);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(lbt::math::exp(inf), zero);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

    TYPED_TEST(ExpTest, positiveInfinityIsPositiveInfinity) {
      constexpr auto inf {std::numeric_limits<TypeParam>::infinity()};
      EXPECT_TRUE(math::isPosInf(lbt::math::exp(inf)));
    }

    TYPED_TEST(ExpTest, nanIsNan) {
      constexpr auto nan {std::numeric_limits<TypeParam>::signaling_NaN()};
      EXPECT_TRUE(std::isnan(lbt::math::exp(nan)));
    }

    TYPED_TEST(ExpTest, unityIsEulersNumber) {
      constexpr auto unity {static_cast<TypeParam>(1.0)};
      constexpr auto eulers_number {lbt::math::e<TypeParam>};
      if constexpr (std::is_same_v<TypeParam,float>) {
        EXPECT_FLOAT_EQ(lbt::math::exp(unity), eulers_number);
      } else if constexpr (std::is_same_v<TypeParam,double>) {
        EXPECT_DOUBLE_EQ(lbt::math::exp(unity), eulers_number);
      } else {
        GTEST_SKIP() << "Test not implemented for given data type!";
      }
    }

    TYPED_TEST(ExpTest, positiveNumbersAreCorrect) {
      std::vector<std::pair<TypeParam,TypeParam>> const tests = { { 0.2, static_cast<TypeParam>(1.2214027581601698339210719946396741703075809415205036412734)},
                                                                  { 1.5, static_cast<TypeParam>(4.4816890703380648226020554601192758190057498683696670567726)},
                                                                  {10.0, static_cast<TypeParam>(22026.465794806716516957900645284244366353512618556781074235426355)},
                                                                  {20.0, static_cast<TypeParam>(4.85165195409790277969106830541540558684638988944847254353610e8)}
                                                                };
      for (auto const& [n, solution]: tests) {
        if constexpr (std::is_same_v<TypeParam,float>) {
          EXPECT_FLOAT_EQ(lbt::math::exp(n), solution);
        } else if constexpr (std::is_same_v<TypeParam,double>) {
          EXPECT_DOUBLE_EQ(lbt::math::exp(n), solution);
        } else {
          GTEST_SKIP() << "Test not implemented for given data type!";
        }
      }
    }

    TYPED_TEST(ExpTest, negativeNumbersAreCorrect) {
      std::vector<std::pair<TypeParam,TypeParam>> const tests = { { -0.4, static_cast<TypeParam>(0.6703200460356393007444329251478260719369809252108121998889)}, 
                                                                  { -1.0, static_cast<TypeParam>(1.0/lbt::math::e<TypeParam>)}, 
                                                                  { -2.3, static_cast<TypeParam>(0.1002588437228037337299406937979871569083970498348733336617)},
                                                                  {-13.0, static_cast<TypeParam>(2.2603294069810543257852772905386894693531424227032186440353e-6)}
                                                                };
      for (auto const& [n, solution]: tests) {
        if constexpr (std::is_same_v<TypeParam,float>) {
          EXPECT_FLOAT_EQ(lbt::math::exp(n), solution);
        } else if constexpr (std::is_same_v<TypeParam,double>) {
          EXPECT_DOUBLE_EQ(lbt::math::exp(n), solution);
        } else {
          GTEST_SKIP() << "Test not implemented for given data type!";
        }
      }
    }

    TYPED_TEST(ExpTest, positiveNumbersEqualToStdExp) {
      std::vector<TypeParam> const tests = { 0.2, 1.5, 10.0, 20.0 };
      for (auto const& n: tests) {
        if constexpr (std::is_same_v<TypeParam,float>) {
          EXPECT_FLOAT_EQ(lbt::math::exp(n), std::exp(n));
        } else if constexpr (std::is_same_v<TypeParam,double>) {
          EXPECT_DOUBLE_EQ(lbt::math::exp(n), std::exp(n));
        } else {
          GTEST_SKIP() << "Test not implemented for given data type!";
        }
      }
    }

     TYPED_TEST(ExpTest, negativeNumbersEqualToStdExp) {
      std::vector<TypeParam> const tests = { -0.4, -1.0, -2.3, -13.0 };
      for (auto const& n: tests) {
        if constexpr (std::is_same_v<TypeParam,float>) {
          EXPECT_FLOAT_EQ(lbt::math::exp(n), std::exp(n));
        } else if constexpr (std::is_same_v<TypeParam,double>) {
          EXPECT_DOUBLE_EQ(lbt::math::exp(n), std::exp(n));
        } else {
          GTEST_SKIP() << "Test not implemented for given data type!";
        }
      }
    }

  }
}

#endif // LBT__MATH__EXP_TEST
