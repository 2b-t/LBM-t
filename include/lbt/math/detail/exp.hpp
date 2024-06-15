/**
 * \file     exp.hpp
 * \mainpage Constexpr function for calculating the exponential function at compile-time.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__EXP
#define LBT__MATH__EXP
#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "lbt/math/detail/abs.hpp"
#include "lbt/math/detail/ceil.hpp"
#include "lbt/math/detail/ipow.hpp"
#include "lbt/math/detail/is_inf.hpp"
#include "lbt/math/detail/is_nan.hpp"
#include "lbt/math/detail/is_almost_equal_eps_rel.hpp"
#include "lbt/math/detail/constants.hpp"


namespace lbt {
  namespace math {

    /**\fn        exp
     * \brief     Exponential function https://en.wikipedia.org/wiki/Exponential_function calculated by Taylor series expansion 
     *            with Horner's method https://www.pseudorandom.com/implementing-exp#section-7 that can be evaluated as a 
     *            constant expression at compile time
     *
     * \tparam    T   Floating point data type of the corresponding number
     * \param[in] x   The number of interest
     * \return    The resulting exponential function evaluated at point \p x
    */
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    constexpr T exp(T const x) noexcept {
      if (math::isAlmostEqualEpsRel(x, static_cast<T>(0.0))) {
        return static_cast<T>(1.0);
      } else if (math::isNegInf(x)) {
        return static_cast<T>(0.0);
      } else if (math::isPosInf(x)) {
        return std::numeric_limits<T>::infinity();
      } else if (math::isNan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
      } else if (math::isAlmostEqualEpsRel(x, static_cast<T>(1.0))) {
        return math::e<T>;
      } 

      T const abs_x {math::abs(x)};
      constexpr std::int64_t multiplier {12}; // Heuristic constant
      std::int64_t const n {static_cast<std::int64_t>(math::ceil(abs_x*math::e<T>)*multiplier)};
      T taylor_series {1.0};
      for (std::int64_t i = n; i > 0; --i) {
        taylor_series = taylor_series*(abs_x / static_cast<T>(i)) + 1.0;
      };
      if (x < 0.0) {
        return 1.0 / taylor_series;
      }
      return taylor_series;
    }

  }
}

#endif // LBT__MATH__EXP
