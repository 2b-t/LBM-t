/**
 * \file     definitions.hpp
 * \mainpage Algorithmic constants for constexpr function library.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__DEFINITIONS
#define LBT__MATH__DEFINITIONS
#pragma once

#include <cstdint>


namespace lbt {
  namespace math {

    /// Recursion depth for recursive algorithms
    inline constexpr std::int64_t default_max_recursion_depth = 9999;

  }
}

#endif // LBT__MATH__DEFINITIONS
