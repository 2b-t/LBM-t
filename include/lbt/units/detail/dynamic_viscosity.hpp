/**
 * \file     dynamic_viscosity.hpp
 * \mainpage Contains unit definition for a dynamic viscosity
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__DYNAMIC_VISCOSITY
#define LBT__UNITS__DYNAMIC_VISCOSITY
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {


    /**\class DynamicViscosity
     * \brief Unit class for fluid dynamic viscosity
    */
    class DynamicViscosity : public lbt::unit::detail::UnitBase<DynamicViscosity> {
      public:
        /**\fn    DynamicViscosity
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class in the base unit Pascal seconds
        */
        explicit constexpr DynamicViscosity(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        DynamicViscosity(DynamicViscosity const&) = default;
        DynamicViscosity& operator= (DynamicViscosity const&) = default;
        DynamicViscosity(DynamicViscosity&&) = default;
        DynamicViscosity& operator= (DynamicViscosity&&) = default;
    };

  }
}

#endif // LBT__UNITS__DYNAMIC_VISCOSITY
