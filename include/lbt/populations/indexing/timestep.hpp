/**
 * \file     timestep.hpp
 * \brief    Base class for timestep
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__INDEXING__TIMESTEP
#define LBT__INDEXING__TIMESTEP
#pragma once

#include <ostream>


namespace lbt {

  /**\enum  Timestep
   * \brief Strongly typed enum for even and odd time steps required for AA access pattern
   */
  enum class Timestep: bool {
    Even = false,
    Odd = true
  };

  /**\fn        operator!
   * \brief     Negation operator for the timestep
   *
   * \param[in] ts   Timestep to be negated
   * \return    Negated timestep
   */
  inline constexpr Timestep operator! (Timestep const& ts) noexcept {
    return (ts == Timestep::Even) ? Timestep::Odd : Timestep::Even;
  }

  /**\fn            Timestep output stream operator
   * \brief         Output stream operator for the timestep
   *
   * \param[in,out] os   Output stream
   * \param[in]     ts   Timestep to be printed to output stream
   * \return        Output stream including the type of timestep
   */
  inline std::ostream& operator << (std::ostream& os, Timestep const& ts) noexcept {
    os << ((ts == Timestep::Even) ? "even time step" : "odd time step");
    return os;
  }

}

#endif // LBT__INDEXING__TIMESTEP
