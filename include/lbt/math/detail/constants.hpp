/**
 * \file     constants.hpp
 * \mainpage Header containing mathematical constant definitions
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__CONSTANTS
#define LBT__MATH__CONSTANTS
#pragma once

#include <type_traits>


namespace lbt {
  namespace math {

    /// Variable templates for Euler's identity pi and Euler's number e
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    inline constexpr T pi = static_cast<T>(3.1415926535897932385L);

    /// Variable templates for Euler's number e
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    inline constexpr T e = static_cast<T>(2.71828182845904523536L);

    /// Variable templates for natural logarithm of two
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    inline constexpr T ln2 = static_cast<T>(0.69314718055994530942L);

  }
}

#endif // LBT__MATH__CONSTANTS
