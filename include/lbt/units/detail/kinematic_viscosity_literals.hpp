/**
 * \file     kinematic_viscosity_literals.hpp
 * \mainpage Contains definitions for user-defined literals for kinematic viscosity
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__KINEMATIC_VISCOSITY_LITERALS
#define LBT__UNITS__KINEMATIC_VISCOSITY_LITERALS
#pragma once

#include "lbt/units/detail/kinematic_viscosity.hpp"
#include "lbt/units/detail/length.hpp"
#include "lbt/units/detail/length_literals.hpp"
#include "lbt/units/detail/operators.hpp"
#include "lbt/units/detail/prefixes.hpp"
#include "lbt/units/detail/time.hpp"
#include "lbt/units/detail/time_literals.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _m2ps
     * \brief     User-defined literal for a kinematic viscosity given in square meters per second
     * 
     * \param[in] p   The kinematic viscosity in square meters per second
     * \return    A kinematic viscosity in the base unit meters squared per second
    */
    constexpr lbt::unit::KinematicViscosity operator "" _m2ps(long double const m) noexcept {
      return m*lbt::unit::KinematicViscosity{1.0_m*1.0_m/1.0_s};
    }
    /**\fn        operator "" _St
     * \brief     User-defined literal for a kinematic viscosity given in Stokes
     * 
     * \param[in] p   The kinematic viscosity in Stokes
     * \return    A kinematic viscosity in the base unit meters squared per second
    */
    constexpr lbt::unit::KinematicViscosity operator "" _St(long double const s) noexcept {
      return s*lbt::unit::KinematicViscosity{lbt::unit::prefix::deci*lbt::unit::prefix::milli};
    }
    /**\fn        operator "" _cSt
     * \brief     User-defined literal for a kinematic viscosity given in Centistokes
     * 
     * \param[in] p   The kinematic viscosity in Centistokes
     * \return    A kinematic viscosity in the base unit meters squared per second
    */
    constexpr lbt::unit::KinematicViscosity operator "" _cSt(long double const c) noexcept {
      return c*lbt::unit::KinematicViscosity{lbt::unit::prefix::micro};
    }

  }
}

#endif // LBT__UNITS__KINEMATIC_VISCOSITY_LITERALS
