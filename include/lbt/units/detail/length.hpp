/**
 * \file     length.hpp
 * \mainpage Contains unit definition for a length
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__LENGTH
#define LBT__UNITS__LENGTH
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class Length
     * \brief Unit class for lengths and distances
    */
    class Length : public lbt::unit::detail::UnitBase<Length> {
      public:
        /**\fn    Length
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class in the base unit meters
        */
        explicit constexpr Length(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        Length(Length const&) = default;
        Length& operator= (Length const&) = default;
        Length(Length&&) = default;
        Length& operator= (Length&&) = default;
    };
    using Distance = Length;

  }
}

#endif // LBT__UNITS__LENGTH
