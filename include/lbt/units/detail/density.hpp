/**
 * \file     density.hpp
 * \mainpage Contains unit definition for a density
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__DENSITY
#define LBT__UNITS__DENSITY
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class Density
     * \brief Unit class for density
    */
    class Density : public lbt::unit::detail::UnitBase<Density> {
      public:
        /**\fn    Density
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class in the base unit kilograms per cubic meter
        */
        explicit constexpr Density(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        Density(Density const&) = default;
        Density& operator= (Density const&) = default;
        Density(Density&&) = default;
        Density& operator= (Density&&) = default;
    };

  }
}

#endif // LBT__UNITS__DENSITY
