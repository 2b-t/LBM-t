#ifndef LBT_CHARACTERISTIC_NUMBERS_UNITTEST
#define LBT_CHARACTERISTIC_NUMBERS_UNITTEST
#pragma once

/**
 * \file     characteristic_numbers_unittest.hpp
 * \mainpage Tests for characteristic numbers of fluid mechanics
 * \author   Tobit Flatscher (github.com/2b-t)
*/


#include <gtest/gtest.h>

#include "../../src/continuum/characteristic_numbers.hpp"
#include "../../src/general/literals.hpp"


namespace lbt {
  namespace unit {
    namespace test {

      TEST(ReynoldsNumberTest, constructor) {
        constexpr long double expected_result {1234.56};
        lbt::unit::ReynoldsNumber const reynolds_number {expected_result};
        EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
      }

      TEST(ReynoldsNumberTest, constructorFromPhysicalUnits) {
        lbt::unit::Velocity const velocity {1.7};
        lbt::unit::Length const length {0.025};
        lbt::unit::KinematicViscosity const kinematic_viscosity {0.0000016};
        constexpr long double expected_result {26562.5};
        lbt::unit::ReynoldsNumber const reynolds_number {velocity, length, kinematic_viscosity};
        EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
      }

      TEST(ReynoldsNumberTest, constructFromOperator) {
        lbt::unit::Velocity const velocity {1.7};
        lbt::unit::Length const length {0.025};
        lbt::unit::KinematicViscosity const kinematic_viscosity {0.0000016};
        constexpr long double expected_result {26562.5};
        auto const reynolds_number {velocity*length/kinematic_viscosity};
        EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
      }

      TEST(ReynoldsNumberTest, conversionToLongDouble) {
        constexpr long double expected_result {1234.56};
        lbt::unit::ReynoldsNumber const reynolds_number {expected_result};
        long double const result {reynolds_number};
        EXPECT_DOUBLE_EQ(result, expected_result);
      }

      TEST(ReynoldsNumberTest, constructorFromLiterals) {
        using namespace lbt::literals;
        auto const velocity = 1.7_mps;
        auto const length = 2.5_cm;
        auto const kinematic_viscosity = 0.0000016_m2ps;
        constexpr long double expected_result {26562.5};
        auto const reynolds_number = velocity*length/kinematic_viscosity;
        EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
      }

    }
  }
}

#endif // LBT_CHARACTERISTIC_NUMBERS_UNITTEST
