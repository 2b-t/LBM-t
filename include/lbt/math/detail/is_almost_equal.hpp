/**
 * \file     is_almost_equal.hpp
 * \mainpage Constexpr functions for floating-point number comparison at compile-time with different criteria.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_ALMOST_EQUAL
#define LBT__MATH__IS_ALMOST_EQUAL
#pragma once

#include "lbt/math/detail/is_almost_equal_eps_abs.hpp"
#include "lbt/math/detail/is_almost_equal_eps_rel.hpp"

#if __cplusplus >= 202002L
  /// Constexpr ULPS comparison only possible from C++20 onwards
  #include "lbt/math/detail/is_almost_equal_ulps.hpp"
#endif

#endif // LBT__MATH__IS_ALMOST_EQUAL
