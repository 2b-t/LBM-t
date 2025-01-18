/**
 * \file     time_literals.hpp
 * \mainpage Contains definitions for user-defined literals for time
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__TIME_LITERALS
#define LBT__UNITS__TIME_LITERALS
#pragma once

#include "lbt/units/detail/prefixes.hpp"
#include "lbt/units/detail/time.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _d
     * \brief     User-defined literal for a time given in days
     * 
     * \param[in] d   The time in days
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _d(long double const d) noexcept {
      return d*lbt::unit::Time{60.0L*60.0L*24.0L};
    }
    /**\fn        operator "" _h
     * \brief     User-defined literal for a time given in hours
     * 
     * \param[in] h   The time in hours
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _h(long double const h) noexcept {
      return h*lbt::unit::Time{60.0L*60.0L};
    }
    /**\fn        operator "" _min
     * \brief     User-defined literal for a time given in minutes
     * 
     * \param[in] m   The time in minutes
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _min(long double const m) noexcept {
      return m*lbt::unit::Time{60.0L};
    }
    /**\fn        operator "" _s
     * \brief     User-defined literal for a time given in seconds
     * 
     * \param[in] s   The time in seconds
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _s(long double const s) noexcept {
      return s*lbt::unit::Time{lbt::unit::prefix::base};
    }
    /**\fn        operator "" _ms
     * \brief     User-defined literal for a time given in milliseconds
     * 
     * \param[in] m   The time in milliseconds
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _ms(long double const m) noexcept {
      return m*lbt::unit::Time{lbt::unit::prefix::milli};
    }
    /**\fn        operator "" _us
     * \brief     User-defined literal for a time given in microseconds
     * 
     * \param[in] u   The time in microseconds
     * \return    A time in the base unit seconds
    */
    constexpr lbt::unit::Time operator "" _us(long double const u) noexcept {
      return u*lbt::unit::Time{lbt::unit::prefix::micro};
    }

  }
}

#endif // LBT__UNITS__TIME_LITERALS
