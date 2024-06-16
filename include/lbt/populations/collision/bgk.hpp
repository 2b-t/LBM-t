/**
 * \file     bgk.hpp
 * \mainpage Class for BGK collision operator
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__COLLISION__BGK
#define LBT__COLLISION__BGK
#pragma once

#include <memory>

#include "lbt/continuums/continuum_base.hpp"
#include "lbt/populations/collision/collision_base.hpp"
#include "lbt/populations/indexing/timestep.hpp"
#include "lbt/units/characteristic_numbers.hpp"
#include "lbt/converter.hpp"


namespace lbt {
  namespace collision {

    /**\class  Bgk
     * \brief  Bhatnagar-Gross-Krook (BGK) collision operator for arbitrary lattice
     *
     * \note   "A Model for Collision Processes in Gases. I. Small Amplitude Processes in Charged
     *          and Neutral One-Component Systems"
     *         P.L. Bhatnagar, E.P. Gross, M. Krook
     *         Physical Review 94 (1954)
     *         DOI: 10.1103/PhysRev.94.511
     *
     * \tparam LT     Static lattice::DdQq class containing discretisation parameters
    */

    template <class LT>
    class Bgk: public CollisionBase<LT> {
      public:
        Bgk(std::shared_ptr<lbt::Population<LT>> population, std::shared_ptr<lbt::ContinuumBase<typename LT::type>> continuum,
          std::shared_ptr<lbt::Converter> converter, lbt::unit::ReynoldsNumber const& reynolds_number,
          double const lbm_velocity, double const lbm_length);
        Bgk() = default;
        Bgk(Bgk const&) = default;
        Bgk& operator= (Bgk const&) = default;
        Bgk(Bgk&&) = default;
        Bgk& operator= (Bgk&&) = default;
    };

    template <class LT>
    Bgk<LT>::Bgk(std::shared_ptr<lbt::Population<LT>> population, std::shared_ptr<lbt::ContinuumBase<typename LT::type>> continuum,
      std::shared_ptr<lbt::Converter> converter, lbt::unit::ReynoldsNumber const& reynolds_number,
      double const lbm_velocity, double const lbm_length)
    : CollisionBase<LT>{population, continuum, converter} {
      return;
    }

    // TODO(tobit): Implement collide-stream

  }

}

#endif // LBT__COLLISION__BGK
