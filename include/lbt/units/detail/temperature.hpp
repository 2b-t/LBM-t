/**
 * \file     temperature.hpp
 * \mainpage Contains unit definition for a temperature
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__TEMPERATURE
#define LBT__UNITS__TEMPERATURE
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class Temperature
     * \brief Unit class for temperature
    */
    class Temperature : public lbt::unit::detail::UnitBase<Temperature> {
      public:
        /**\fn    Temperature
         * \brief Constructor, defaults to 0 degrees (273.15 Kelvin)
         * 
         * \param[in] value   The value to be stored inside the class in Kelvin
        */
        explicit constexpr Temperature(long double const value = 273.15l) noexcept
          : UnitBase{value} {
          return;
        }
        Temperature(Temperature const&) = default;
        Temperature& operator= (Temperature const&) = default;
        Temperature(Temperature&&) = default;
        Temperature& operator= (Temperature&&) = default;
    };

  }
}

#endif // LBT__UNITS__TEMPERATURE
