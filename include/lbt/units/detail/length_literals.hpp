/**
 * \file     length_literals.hpp
 * \mainpage Contains definitions for user-defined literals for lengths
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__LENGTH_LITERALS
#define LBT__UNITS__LENGTH_LITERALS
#pragma once

#include "lbt/units/detail/length.hpp"
#include "lbt/units/detail/prefixes.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _km
     * \brief     User-defined literal for a length given in kilometers
     * 
     * \param[in] k   The distance in kilometers
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _km(long double const k) noexcept {
      return k*lbt::unit::Length{lbt::unit::prefix::kilo};
    }
    /**\fn        operator "" _m
     * \brief     User-defined literal for a length given in meters
     * 
     * \param[in] m   The distance in meters
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _m(long double const m) noexcept {
      return m*lbt::unit::Length{lbt::unit::prefix::base};
    }
    /**\fn        operator "" _dm
     * \brief     User-defined literal for a length given in decimeters
     * 
     * \param[in] d   The distance in decimeters
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _dm(long double const d) noexcept {
      return d*lbt::unit::Length{lbt::unit::prefix::deci};
    }
    /**\fn        operator "" _cm
     * \brief     User-defined literal for a length given in centimeters
     * 
     * \param[in] c   The distance in centimeters
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _cm(long double const c) noexcept {
      return c*lbt::unit::Length{lbt::unit::prefix::centi};
    }
    /**\fn        operator "" _mm
     * \brief     User-defined literal for a length given in millimeters
     * 
     * \param[in] m   The distance in millimetres
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _mm(long double const m) noexcept {
      return m*lbt::unit::Length{lbt::unit::prefix::milli};
    }
    /**\fn        operator "" _um
     * \brief     User-defined literal for a length given in micrometers
     * 
     * \param[in] u   The distance in micrometers
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _um(long double const u) noexcept {
      return u*lbt::unit::Length{lbt::unit::prefix::micro};
    }
    /**\fn        operator "" _pm
     * \brief     User-defined literal for a length given in picometers
     * 
     * \param[in] u   The distance in picometers
     * \return    A length in the base unit meters
    */
    constexpr lbt::unit::Length operator "" _pm(long double const u) noexcept {
      return u*lbt::unit::Length{lbt::unit::prefix::pico};
    }

  }
}

#endif // LBT__UNITS__LENGTH_LITERALS
