/**
 * \file     collision_base.hpp
 * \mainpage Base class for all collision operators
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__COLLISION__COLLISION_BASE
#define LBT__COLLISION__COLLISION_BASE
#pragma once

#include <memory>

#include "lbt/continuums/continuum_base.hpp"
#include "lbt/populations/indexing/timestep.hpp"
#include "lbt/populations/population.hpp"
#include "lbt/units/characteristic_numbers.hpp"
#include "lbt/converter.hpp"


namespace lbt {
  namespace collision {

    /**\class  CollisionBase
     * \brief  Base class for all collision operators
     *
     * \tparam LT   Static lattice::DdQq class containing discretisation parameters
    */
    template <class LT>
    class CollisionBase {
      public:
        template<lbt::Timestep TS>
        void initialize();

        template<lbt::Timestep TS>
        void collideAndStream(bool const is_save);
      
      protected:
        // Allow population and unit converter to be nullptrs
        CollisionBase(std::shared_ptr<lbt::Population<LT>> population,
          std::shared_ptr<lbt::ContinuumBase<typename LT::type>> continuum,
          std::shared_ptr<lbt::Converter> converter);
        CollisionBase() = delete;
        CollisionBase(CollisionBase const&) = default;
        CollisionBase& operator= (CollisionBase const&) = default;
        CollisionBase(CollisionBase&&) = default;
        CollisionBase& operator= (CollisionBase&&) = default;

        std::shared_ptr<lbt::Population<LT>> population_;
        std::shared_ptr<lbt::ContinuumBase<typename LT::type>> continuum_;
        std::shared_ptr<lbt::Converter> converter_;
    };

    template <class LT>
    CollisionBase<LT>::CollisionBase(std::shared_ptr<lbt::Population<LT>> population,
      std::shared_ptr<lbt::ContinuumBase<typename LT::type>> continuum,
      std::shared_ptr<lbt::Converter> converter)
    : population_{population}, continuum_{continuum}, converter_{converter} {
      return;
    }

    template <class LT> template<lbt::Timestep TS>
    void CollisionBase<LT>::initialize() {
      // TODO(tobit): Initialize population
      return;
    }

    template <class LT> template<lbt::Timestep TS>
    void CollisionBase<LT>::collideAndStream(bool const is_save) {
      // Implement collide stream through CRTP
      return;
    }

  }

}

#endif // LBT__COLLISION__COLLISION_BASE
