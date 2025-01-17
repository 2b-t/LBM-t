/**
 * \file     pressure_literals.hpp
 * \mainpage Contains definitions for user-defined literals for pressure
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__PRESSURE_LITERALS
#define LBT__UNITS__PRESSURE_LITERALS
#pragma once

#include "lbt/units/detail/prefixes.hpp"
#include "lbt/units/detail/pressure.hpp"


namespace lbt {
  namespace literals {

    /**\fn        operator "" _Pa
     * \brief     User-defined literal for a pressure given in Pascal
     * 
     * \param[in] p   The pressure in Pascal
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _Pa(long double const p) noexcept {
      return p*lbt::unit::Pressure{lbt::unit::prefix::base};
    }
    /**\fn        operator "" _GPa
     * \brief     User-defined literal for a pressure given in Giga-Pascal
     * 
     * \param[in] p   The pressure in Giga-Pascal
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _GPa(long double const g) noexcept {
      return g*lbt::unit::Pressure{lbt::unit::prefix::giga};
    }
    /**\fn        operator "" _mPa
     * \brief     User-defined literal for a pressure given in milli-Pascal
     * 
     * \param[in] p   The pressure in milli-Pascal
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _mPa(long double const m) noexcept {
      return m*lbt::unit::Pressure{lbt::unit::prefix::milli};
    }
    /**\fn        operator "" _uPa
     * \brief     User-defined literal for a pressure given in micro-Pascal
     * 
     * \param[in] p   The pressure in micro-Pascal
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _uPa(long double const m) noexcept {
      return m*lbt::unit::Pressure{lbt::unit::prefix::micro};
    }
    /**\fn        operator "" _hPa
     * \brief     User-defined literal for a pressure given in hecto-Pascal
     * 
     * \param[in] p   The pressure in hecto-Pascal
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _hPa(long double const h) noexcept {
      return h*lbt::unit::Pressure{lbt::unit::prefix::hecto};
    }
    /**\fn        operator "" _bar
     * \brief     User-defined literal for a pressure given in bar
     * 
     * \param[in] p   The pressure in bar
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _bar(long double const b) noexcept {
      return b*lbt::unit::Pressure{lbt::unit::prefix::hecto*lbt::unit::prefix::kilo};
    }
    /**\fn        operator "" _atm
     * \brief     User-defined literal for a pressure given in atmospheres
     * 
     * \param[in] p   The pressure in atmospheres
     * \return    A pressure in the base unit Pascal
    */
    constexpr lbt::unit::Pressure operator "" _atm(long double const a) noexcept {
      return a*lbt::unit::Pressure{101325};
    }

  }
}

#endif // LBT__UNITS__PRESSURE_LITERALS
