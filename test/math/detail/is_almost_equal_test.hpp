/**
 * \file     is_almost_equal_test.hpp
 * \mainpage Tests for constexpr almost equal comparison functions with different criteria.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__IS_ALMOST_EQUAL_TEST
#define LBT__MATH__IS_ALMOST_EQUAL_TEST
#pragma once

#include "is_almost_equal_eps_abs_test.hpp"
#include "is_almost_equal_eps_rel_test.hpp"

#if __cplusplus >= 202002L
  /// Constexpr ULPS comparison only possible from C++20 onwards
  #include "is_almost_equal_ulps_test.hpp"
#endif

#endif // LBT__MATH__IS_ALMOST_EQUAL_TEST
