/**
 * \file     volume.hpp
 * \mainpage Contains unit definition for a volume
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__VOLUME
#define LBT__UNITS__VOLUME
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class Volume
     * \brief Unit class for three-dimensional volume in the base unit cubic meters
    */
    class Volume : public lbt::unit::detail::UnitBase<Volume> {
      public:
        /**\fn    Volume
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class
        */
        explicit constexpr Volume(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        Volume(Volume const&) = default;
        Volume& operator= (Volume const&) = default;
        Volume(Volume&&) = default;
        Volume& operator= (Volume&&) = default;
    };

  }
}

#endif // LBT__UNITS__VOLUME
