/**
 * \file     mass_literals.hpp
 * \mainpage Contains definitions for user-defined literals for mass
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__MASS_LITERALS
#define LBT__UNITS__MASS_LITERALS
#pragma once

#include "lbt/units/detail/mass.hpp"
#include "lbt/units/detail/prefixes.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _t
     * \brief     User-defined literal for a mass given in tons
     * 
     * \param[in] t   The mass in tons
     * \return    A mass in the base unit kilograms
    */
    constexpr lbt::unit::Mass operator "" _t(long double const t) noexcept {
      return t*lbt::unit::Mass{lbt::unit::prefix::kilo};
    }
    /**\fn        operator "" _kg
     * \brief     User-defined literal for a mass given in kilograms
     * 
     * \param[in] k   The mass in kilograms
     * \return    A mass in the base unit kilograms
    */
    constexpr lbt::unit::Mass operator "" _kg(long double const k) noexcept {
      return k*lbt::unit::Mass{lbt::unit::prefix::base};
    }
    /**\fn        operator "" _g
     * \brief     User-defined literal for a mass given in grams
     * 
     * \param[in] g   The mass in grams
     * \return    A mass in the base unit kilograms
    */
    constexpr lbt::unit::Mass operator "" _g(long double const g) noexcept {
      return g*lbt::unit::Mass{lbt::unit::prefix::milli};
    }

  }
}

#endif // LBT__UNITS__MASS_LITERALS
