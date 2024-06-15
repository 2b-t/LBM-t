/**
 * \file     air.hpp
 * \mainpage Contains methods for calculating physical properties of air
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATERIALS__AIR
#define LBT__MATERIALS__AIR
#pragma once

#include "lbt/materials/detail/ideal_gas.hpp"
#include "lbt/units/literals.hpp"
#include "lbt/units/units.hpp"


namespace lbt {
  namespace material {

    namespace physical_constants {
      using namespace lbt::literals;

      class Air {
        public:
          static constexpr auto molecular_weight = 28.966_gpmol;
          static constexpr auto c = 120.0_K;
          static constexpr auto t_0 = 291.15_K;
          static constexpr auto mu_0 = 18.27_uPas;

        protected:
          Air() = default;
          Air(Air const&) = default;
          Air& operator= (Air const&) = default;
          Air(Air&&) = default;
          Air& operator= (Air&&) = default;
      };

    }

    using Air = IdealGas<physical_constants::Air>;

  }
}

#endif // LBT__MATERIALS__AIR
