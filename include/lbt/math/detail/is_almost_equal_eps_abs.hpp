/**
 * \file     is_almost_equal_eps_abs.hpp
 * \mainpage Constexpr functions for floating-point number comparison at compile-time with fixed absolute epsilon.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS
#define LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS
#pragma once

#include <limits>
#include <type_traits>

#include "lbt/math/detail/abs.hpp"


namespace lbt {
  namespace math {

    /**\fn        isAlmostEqualEpsAbs
     * \brief     Constexpr function for comparing two floating point numbers with a given absolute tolerance
     *
     * \tparam    T         Data type of the corresponding number
     * \param[in] a         The first number to be compared
     * \param[in] b         The second number to be compared
     * \param[in] epsilon   The absolute tolerance to be considered
     * \return    Boolean value signaling whether the two values \p a and \p b are equal considering an absolute tolerance
    */
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    constexpr bool isAlmostEqualEpsAbs(T const a, T const b, T epsilon = 10*std::numeric_limits<T>::epsilon()) noexcept {
      return (math::abs(a - b) < epsilon);
    }

  }
}

#endif // LBT__MATH__IS_ALMOST_EQUAL_EPS_ABS
