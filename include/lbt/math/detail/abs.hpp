/**
 * \file     abs.hpp
 * \mainpage Function for calculating the absolute value at compile-time
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__ABS
#define LBT__MATH__ABS
#pragma once

#include <type_traits>

#include "lbt/math/detail/is_inf.hpp"
#include "lbt/math/detail/is_nan.hpp"


namespace lbt {
  namespace math {

    /**\fn        abs
     * \brief     Constexpr function for absolute value
     *
     * \tparam    T   Data type of the corresponding number
     * \param[in] x   The number of interest
     * \return    The absolute value of \p x
    */
    template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
    constexpr T abs(T const x) noexcept {
      if constexpr (std::is_floating_point_v<T>) {
        if (math::isNan(x)) {
          return x;
        } else if (math::isPosInf(x)) {
          return x;
        } else if (math::isNegInf(x)) {
          return -x;
        }
      }

      return (x < 0) ? -x : x;
    }

  }
}

#endif // LBT__MATH__ABS
