#ifndef LBT_AB_POPULATION
#define LBT_AB_POPULATION
#pragma once

/**
 * \file     ab_population.hpp
 * \mainpage Class for microscopic populations based on two distinct populations
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#include <cassert>
#include <cstdint>
#include <iostream>

#include "../general/type_definitions.hpp"
#include "indexing/indexing.hpp"


namespace lbt {

  /**\class  AbPopulation
   * \brief  Class that holds macroscopic values with indexing based on the A-B access pattern
   *
   * \tparam LT     Static lattice::DdQq class containing discretisation parameters
   * \tparam NPOP   Number of populations stored side by side in a single merged grid (default = 1)
  */
  template <class LT, std::int32_t NPOP = 1>
  class AbPopulation: public Indexing<LT,NPOP> {
    public:
      AbPopulation() = delete;
      AbPopulation(AbPopulation&) = delete;
      AbPopulation& operator= (AbPopulation&) = delete;
      AbPopulation(AbPopulation&&) = delete;
      AbPopulation& operator= (AbPopulation&&) = delete;

      /**\fn    AbPopulation
       * \brief Class constructor
      */
      AbPopulation(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
        : Indexing<LT,NPOP>{NX, NY, NZ} {
        static_assert((LT::DIM == 2) ? (NZ == 1) : true, "Two-dimensional lattice with NZ != 1.");

        memory_size = static_cast<std::int64_t>(sizeof(T))*NZ*NY*NX*NPOP*LT::ND;
        A = lbt::aligned_alloc(memory_size, LBT_CACHE_LINE_SIZE));
        B = lbt::aligned_alloc(memory_size, LBT_CACHE_LINE_SIZE));

        return;
      }

      /**\fn    ~AbPopulation
       * \brief Class destructor
      */
      ~AbPopulation() noexcept {
        lbt::aligned_free(A);
        lbt::aligned_free(B);
        std::clog << "See you, comrade!" << std::endl;
        return;
      }

      /**\fn        read
       * \brief     Function for accessing values before collision depending on even and odd time step.
       * \warning   Sloppy syntax: avoid usage, use the full expression below for indexing
       *
       * \tparam    TS   Even (0, false) or odd (1, true) time step
       * \param[in] x    x coordinate of current cell
       * \param[in] y    y coordinate of current cell
       * \param[in] z    z coordinate of current cell
       * \param[in] n    Positive (0) or negative (1) index/lattice velocity
       * \param[in] d    Relevant population index
       * \param[in] p    Relevant population (default = 0)
       * \return    Requested linear population index before collision (reading)
      */
      template <Timestep TS>
      LBT_FORCE_INLINE auto& read(std::int32_t const x, std::int32_t const y, std::int32_t const z,
                                  std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) noexcept {
        if constexpr (TS == Timestep::Even) {
          return A[indexRead<TS>(x,y,z,n,d,p)];
        } else {
          return B[indexRead<TS>(x,y,z,n,d,p)];
        }
      }

      template <Timestep TS>
      LBT_FORCE_INLINE auto const& read(std::int32_t const x, std::int32_t const y, std::int32_t const z,
                                        std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) const noexcept {
        if constexpr (TS == Timestep::Even) {
          return A[indexRead<TS>(x,y,z,n,d,p)];
        } else {
          return B[indexRead<TS>(x,y,z,n,d,p)];
        }
      }

      /**\fn        write
       * \brief     Function for accessing values after collision depending on even and odd time step.
       * \warning   Sloppy syntax: avoid usage, use the full expression below for indexing
       *
       * \tparam    TS   Even (0, false) or odd (1, true) time step
       * \param[in] x    x coordinate of current cell
       * \param[in] y    y coordinate of current cell
       * \param[in] z    z coordinate of current cell
       * \param[in] n    Positive (0) or negative (1) index/lattice velocity
       * \param[in] d    Relevant population index
       * \param[in] p    Relevant population (default = 0)
       * \return    Requested linear population index after collision (writing)
      */
      template <Timestep TS>
      LBT_FORCE_INLINE auto& write(std::int32_t const x, std::int32_t const y, std::int32_t const z,
                                   std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) noexcept {
        // TODO: include streaming as well
        if constexpr (TS == Timestep::Even) {
          return B[indexWrite<TS>(x,y,z,n,d,p)];
        } else {
          return A[indexWrite<TS>(x,y,z,n,d,p)];
        }
      }

      template <Timestep TS>
      LBT_FORCE_INLINE auto const& write(std::int32_t const x, std::int32_t const y, std::int32_t const z,
                                         std::int32_t const n, std::int32_t const d, std::int32_t const p) const noexcept {
        // TODO: include streaming as well
        if constexpr (TS == Timestep::Even) {
          return B[indexWrite<TS>(x,y,z,n,d,p)];
        } else {
          return A[indexWrite<TS>(x,y,z,n,d,p)];
        }
      }

    protected:
      std::int64_t memory_size;
      T* A;
      T* B;
  };

}

#endif // LBT_AB_POPULATION
