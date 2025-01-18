/**
 * \file     mass.hpp
 * \mainpage Contains unit definition for a mass
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__MASS
#define LBT__UNITS__MASS
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class Mass
     * \brief Unit class for mass
    */
    class Mass : public lbt::unit::detail::UnitBase<Mass> {
      public:
        /**\fn    Mass
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class in the base unit kilograms
        */
        explicit constexpr Mass(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        Mass(Mass const&) = default;
        Mass& operator= (Mass const&) = default;
        Mass(Mass&&) = default;
        Mass& operator= (Mass&&) = default;
    };

  }
}

#endif // LBT__UNITS__MASS
