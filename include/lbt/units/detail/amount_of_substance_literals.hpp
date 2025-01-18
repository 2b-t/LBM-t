/**
 * \file     amount_of_substance_literals.hpp
 * \mainpage Contains definitions for user-defined literals for amount of substance
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__AMOUNT_OF_SUBSTANCES_LITERALS
#define LBT__UNITS__AMOUNT_OF_SUBSTANCES_LITERALS
#pragma once

#include "lbt/units/detail/amount_of_substance.hpp"
#include "lbt/units/detail/prefixes.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _t
     * \brief     User-defined literal for an amount of substance given in mole
     * 
     * \param[in] m   The amount of substance in mole
     * \return    An amount of substance in the base unit mole
    */
    constexpr lbt::unit::AmountOfSubstance operator "" _mol(long double const m) noexcept {
      return m*lbt::unit::AmountOfSubstance{lbt::unit::prefix::base};
    }
    /**\fn        operator "" _kg
     * \brief     User-defined literal for an amount of substance given in kilomole
     * 
     * \param[in] k   The amount of substance in kilomole
     * \return    An amount of substance in the base unit mole
    */
    constexpr lbt::unit::AmountOfSubstance operator "" _kmol(long double const k) noexcept {
      return k*lbt::unit::AmountOfSubstance{lbt::unit::prefix::kilo};
    }

  }
}

#endif // LBT__UNITS__AMOUNT_OF_SUBSTANCES_LITERALS
