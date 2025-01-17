/**
 * \file     pow.hpp
 * \mainpage Constexpr function for calculating the power function at compile-time.
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATH__POW
#define LBT__MATH__POW
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "lbt/math/detail/exp.hpp"
#include "lbt/math/detail/ipow.hpp"
#include "lbt/math/detail/is_inf.hpp"
#include "lbt/math/detail/is_nan.hpp"
#include "lbt/math/detail/is_almost_equal_eps_rel.hpp"
#include "lbt/math/detail/log.hpp"


namespace lbt {
  namespace math {

    /**\fn        pow
     * \brief     Computes the the base \p x raised to the power \p y at compile-time using the identity x^y = exp(y*log(x))
     *
     * \tparam    T   Floating point data type of the corresponding number
     * \param[in] x   The base of the exponentiation
     * \param[in] y   The exponent of the exponentiation
     * \return    The base \p x raised to the exponent \p y
    */
    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    constexpr T pow(T const x, T const y) noexcept {
      auto constexpr pos_inf {std::numeric_limits<T>::infinity()};
      auto constexpr neg_inf {-std::numeric_limits<T>::infinity()};
      auto constexpr nan {-std::numeric_limits<T>::quiet_NaN()};
      
      bool const is_base_almost_zero {math::isAlmostEqualEpsRel(x, static_cast<T>(0.0))};
      bool const is_base_pos {(x > static_cast<T>(0.0))};
      bool const is_base_neg {(x < static_cast<T>(0.0))};
      bool const is_base_neg_inf {math::isNegInf(x)};
      bool const is_base_pos_inf {math::isPosInf(x)};
      bool const is_base_inf {is_base_neg_inf || is_base_pos_inf};
      bool const is_base_nan {math::isNan(x)};
      bool const is_base_finite {!is_base_nan && !is_base_inf};

      bool const is_exp_pos {(y > static_cast<T>(0.0))};
      bool const is_exp_neg {(y < static_cast<T>(0.0))};
      bool const is_exp_neg_inf {math::isNegInf(y)};
      bool const is_exp_pos_inf {math::isPosInf(y)};
      bool const is_exp_inf {is_exp_neg_inf || is_exp_pos_inf};
      bool const is_exp_nan {math::isNan(y)};
      bool const is_exp_finite {!is_exp_nan && !is_exp_inf};
      bool const is_exp_int {math::isAlmostEqualEpsRel(y, math::ceil(y)) && is_exp_finite};

      // Avoid overflow with integer version and ceil function but reducing possible max and min!
      if ((y > static_cast<T>(std::numeric_limits<std::int64_t>::max()-1)) && (!is_exp_pos_inf)) {
        return pow(x, pos_inf);
      } else if ((y < static_cast<T>(std::numeric_limits<std::int64_t>::min())) && (!is_exp_neg_inf)) {
        return pow(x, neg_inf);
      }

      if (is_exp_int) {
        return math::ipow(x, static_cast<std::int64_t>(y));
      } else if (is_base_almost_zero && is_exp_neg && is_exp_finite) {
        return pos_inf;
      } else if (is_base_almost_zero && is_exp_neg_inf) {
        return pos_inf;
      } else if (is_base_almost_zero && is_exp_pos) {
        return static_cast<T>(+0.0);
      } else if (math::isAlmostEqualEpsRel(x, static_cast<T>(-1.0)) && is_exp_inf) {
        return static_cast<T>(1.0);
      } else if (math::isAlmostEqualEpsRel(x, static_cast<T>(1.0))) {
        return static_cast<T>(1.0);
      } else if (math::isAlmostEqualEpsRel(y, static_cast<T>(0.0))) {
        return static_cast<T>(1.0);
      } else if (is_base_finite && is_base_neg && is_exp_finite) {
        return nan;
      } else if ((math::abs(x) < static_cast<T>(1.0)) && is_exp_neg_inf) {
        return pos_inf;
      } else if ((math::abs(x) > static_cast<T>(1.0)) && is_exp_neg_inf) {
        return static_cast<T>(+0);
      } else if ((math::abs(x) < static_cast<T>(1.0)) && is_exp_pos_inf) {
        return static_cast<T>(+0);
      } else if ((math::abs(x) > static_cast<T>(1.0)) && is_exp_pos_inf) {
        return pos_inf;
      } else if (is_base_neg_inf && is_exp_neg) {
        return static_cast<T>(+0.0);
      } else if (is_base_neg_inf && is_exp_pos) {
        return pos_inf;
      } else if (is_base_pos_inf && is_exp_neg) {
        return static_cast<T>(+0.0);
      } else if (is_base_pos_inf && is_exp_pos) {
        return pos_inf;
      } else if (is_base_nan || is_exp_nan) {
        return nan;
      }
      
      return math::exp(y*math::log(x));
    }

  }
}

#endif // LBT__MATH__POW
