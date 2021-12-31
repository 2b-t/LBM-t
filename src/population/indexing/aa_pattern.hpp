#ifndef LBT_AA_PATTERN
#define LBT_AA_PATTERN

/**
 * \file     aa_pattern.hpp
 * \brief    Class members for indexing of populations with A-A access pattern
 * \author   Tobit Flatscher (github.com/2b-t)
 *
 * \mainpage The A-A access pattern avoids the usage of two distinct populations before and
 *           after streaming by treating even and odd time steps differently: Even time steps
 *           perform only a local collision step with a reverse read of the populations and a
 *           regular write while odd steps perform a combined streaming-collision-streaming
 *           step with a regular read and a reverse write.
 *           This is implemented by different macros that determine the population indices for
 *           even and odd time steps.
*/

#include <array>
#include <cstdint>
#include <iostream>

#include "../../general/type_definitions.hpp"
#include "indexing.hpp"


namespace lbt {

  /**\enum  Timestep
   * \brief Strongly typed enum for even and odd time steps required for AA access pattern
   */
  enum class Timestep: bool { Even = false, Odd = true };

  /**\fn        Negation timestep operator
   * \brief     Negation operator for the timestep
   *
   * \param[in] ts   Timestep to be negated
   * \return    Negated timestep
   */
  constexpr Timestep operator! (Timestep const& ts) noexcept {
    return (ts == Timestep::Even) ? Timestep::Odd : Timestep::Even;
  }

  /**\fn            Timestep output stream operator
   * \brief         Output stream operator for the timestep
   *
   * \param[in,out] os   Output stream
   * \param[in]     ts   Timestep to be printed to output stream
   * \return        Output stream including the type of timestep
   */
  std::ostream& operator << (std::ostream& os, Timestep const& ts) noexcept {
    os << ((ts == Timestep::Even) ? "even time step" : "odd time step");
    return os;
  }

  /**\class  AaPattern
   * \brief  Class that is responsible for indexing according to Bailey's A-A pattern
   *
   * \note   "Accelerating Lattice Boltzmann Fluid Flow Simulations Using Graphics Processors"
   *         P. Bailey, J. Myre, S.D.C. Walsh, D.J. Lilja, M.O. Saar
   *         38th International Conference on Parallel Processing (ICPP), Vienna, Austria (2009)
   *         DOI: 10.1109/ICPP.2009.38
   *
   * \tparam LT   Static lattice::DdQq class containing discretisation parameters
   * \tparam NP   Number of populations stored side by side in a single merged grid (default = 1)
  */
  template <typename LT, std::int32_t NP>
  class AaPattern : public Indexing<LT,NP> {
    // Move protected data members to bottom

    public:
      constexpr AaPattern(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
        : Indexing<LT,NP>{NX,NY,NZ} {
        return;
      }

      /**\fn        oddEven
       * \brief     Function for access of individual indices in odd and even time steps
       * \warning   Reduces the range of unsigned int -> int and might overflow!
       *
       * \param[in] ts           boolean index for odd (1 = true) and even (0 = false) time steps
       * \param[in] odd_index    index to be accessed at an odd time step
       * \param[in] even_index   index to be accessed at an even time step
      */
      template <Timestep TS>
      LBT_FORCE_INLINE constexpr std::int32_t oddEven(std::int32_t const odd_index, std::int32_t const even_index) const noexcept {
        return (TS == Timestep::Odd) ? odd_index : even_index;
      }

      /**\fn        indexRead
       * \brief     Function for determining linear index when reading/writing values before collision depending on even and odd time step.
       * \warning   Inline function! Has to be declared in header!
       *
       * \tparam    TS    Even (0, false) or odd (1, true) time step
       * \param[in] x     x coordinates of current cell and its neighbours [x-1,x,x+1]
       * \param[in] y     y coordinates of current cell and its neighbours [y-1,y,y+1]
       * \param[in] z     z coordinates of current cell and its neighbours [z-1,z,z+1]
       * \param[in] n     Positive (0) or negative (1) index/lattice velocity
       * \param[in] d     Relevant population index
       * \param[in] p     Relevant population (default = 0)
       * \return    Requested linear population index before collision
      */
      template <Timestep TS>
      LBT_FORCE_INLINE constexpr std::int64_t indexRead(lbt::array<std::int32_t,3> const& x,
                                                        lbt::array<std::int32_t,3> const& y,
                                                        lbt::array<std::int32_t,3> const& z,
                                                        std::int32_t               const n,
                                                        std::int32_t               const d,
                                                        std::int32_t               const p) const noexcept {
        return spatialToLinear(x[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DX[(!n)*LT::OFF+d]), 0)],
                               y[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DY[(!n)*LT::OFF+d]), 0)],
                               z[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DZ[(!n)*LT::OFF+d]), 0)],
                               oddEven<TS>(n, !n),
                               d, p);
      }

      /**\fn        indexWrite
       * \brief     Function for determining linear index when reading/writing values after collision depending on even and odd time step.
       * \warning   Inline function! Has to be declared in header!
       *
       * \tparam    TS    Even (0, false) or odd (1, true) time step
       * \param[in] x     x coordinates of current cell and its neighbours [x-1,x,x+1]
       * \param[in] y     y coordinates of current cell and its neighbours [y-1,y,y+1]
       * \param[in] z     z coordinates of current cell and its neighbours [z-1,z,z+1]
       * \param[in] n     Positive (0) or negative (1) index/lattice velocity
       * \param[in] d     Relevant population index
       * \param[in] p     Relevant population (default = 0)
       * \return    Requested linear population index after collision
      */
      template <Timestep TS>
      LBT_FORCE_INLINE constexpr std::int64_t indexWrite(lbt::array<std::int32_t,3> const& x,
                                                         lbt::array<std::int32_t,3> const& y,
                                                         lbt::array<std::int32_t,3> const& z,
                                                         std::int32_t               const n,
                                                         std::int32_t               const d,
                                                         std::int32_t               const p) const noexcept {
        return spatialToLinear(x[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DX[n*LT::OFF+d]), 0)],
                               y[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DY[n*LT::OFF+d]), 0)],
                               z[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DZ[n*LT::OFF+d]), 0)],
                               oddEven<TS>(!n, n),
                               d, p);
      }
  };

}

#endif // LBT_AA_PATTERN
