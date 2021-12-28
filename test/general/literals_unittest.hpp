#ifndef LBT_LITERALS_UNITTEST
#define LBT_LITERALS_UNITTEST
#pragma once

/**\file     literals_unittest.hpp
 * \mainpage Contains unit-tests for literals of different physical units
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "../../src/general/literals.hpp"


namespace lbt {
  namespace literals {
    namespace test {
      using namespace lbt::literals;

      /**\class UnitLiteralsHelper
       * \brief Helper class for parametrised length
       * 
       * \tparam T   The type of physical unit (e.g. lbt::unit::Length)
      */
      template <typename T>
      class UnitLiteralsHelper : public testing::Test, public testing::WithParamInterface<std::pair<T,long double>> {
      };

      /// Parametrised unit-tests
      using LengthLiteralsHelper = UnitLiteralsHelper<lbt::unit::Length>;
      TEST_P(LengthLiteralsHelper, unitConversion) {
        auto const [length, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(length.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(LengthLiteralsTest, LengthLiteralsHelper, testing::Values(
          std::make_pair(0.6_km, 600.0L),
          std::make_pair(3.2_m,    3.2L),
          std::make_pair(9.9_dm,   0.99L),
          std::make_pair(7.3_cm,   0.073L),
          std::make_pair(2.6_mm,   0.0026L),
          std::make_pair(8.9_um,   8.9e-6L)
        )
      );

      using TimeLiteralsHelper = UnitLiteralsHelper<lbt::unit::Time>;
      TEST_P(TimeLiteralsHelper, unitConversion) {
        auto const [time, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(time.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(TimeLiteralsTest, TimeLiteralsHelper, testing::Values(
          std::make_pair(0.8_d,   69120.0L),
          std::make_pair(1.3_h,    4680.0L),
          std::make_pair(6.9_min,   414.0L),
          std::make_pair(5.6_s,       5.6L),
          std::make_pair(1.4_ms,      0.0014L),
          std::make_pair(2.9_us,      2.9e-6L)
        )
      );

      using MassLiteralsHelper = UnitLiteralsHelper<lbt::unit::Mass>;
      TEST_P(MassLiteralsHelper, unitConversion) {
        auto const [mass, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(mass.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(MassLiteralsTest, MassLiteralsHelper, testing::Values(
          std::make_pair(9.9_t,  9900.0L),
          std::make_pair(0.7_kg,    0.7L),
          std::make_pair(3.8_g,     0.0038L)
        )
      );

      using AreaLiteralsHelper = UnitLiteralsHelper<lbt::unit::Area>;
      TEST_P(AreaLiteralsHelper, unitConversion) {
        auto const [area, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(area.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(AreaLiteralsTest, AreaLiteralsHelper, testing::Values(
          std::make_pair(0.1_km2, 100000.0L),
          std::make_pair(2.9_m2,       2.9L),
          std::make_pair(7.4_cm2,      0.00074L),
          std::make_pair(6.6_mm2,      6.6e-6L)
        )
      );

      using VolumeLiteralsHelper = UnitLiteralsHelper<lbt::unit::Volume>;
      TEST_P(VolumeLiteralsHelper, unitConversion) {
        auto const [volume, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(volume.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(VolumeLiteralsTest, VolumeLiteralsHelper, testing::Values(
          std::make_pair(0.3_km3, 3.0e+8L),
          std::make_pair(8.6_m3,  8.6L),
          std::make_pair(1.2_cm3, 1.2e-6L),
          std::make_pair(9.8_mm3, 9.8e-9L)
        )
      );

      using VelocityLiteralsHelper = UnitLiteralsHelper<lbt::unit::Velocity>;
      TEST_P(VelocityLiteralsHelper, unitConversion) {
        auto const [velocity, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(velocity.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(VelocityLiteralsTest, VelocityLiteralsHelper, testing::Values(
          std::make_pair(9.0_kmph, 2.5L),
          std::make_pair(1.7_mps,  1.7L),
          std::make_pair(8.9_cmps, 0.089L),
          std::make_pair(5.3_mmps, 0.0053L)
        )
      );

      using KinematicViscosityLiteralsHelper = UnitLiteralsHelper<lbt::unit::KinematicViscosity>;
      TEST_P(KinematicViscosityLiteralsHelper, unitConversion) {
        auto const [kinematic_viscosity, expected_result] = GetParam();
        EXPECT_DOUBLE_EQ(kinematic_viscosity.get(), expected_result);
      }
      INSTANTIATE_TEST_SUITE_P(KinematicViscosityLiteralsTest, KinematicViscosityLiteralsHelper, testing::Values(
          std::make_pair(7.4_m2ps, 7.4L),
          std::make_pair(5.6_St,   0.00056L),
          std::make_pair(3.5_cSt,  3.5e-6L)
        )
      );

    }
  }
}

#endif // LBT_LITERALS_UNITTEST
