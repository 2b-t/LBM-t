/**
 * \file     hydrogen.hpp
 * \mainpage Contains methods for calculating physical properties of hydrogen
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATERIALS__HYDROGEN
#define LBT__MATERIALS__HYDROGEN
#pragma once

#include "lbt/materials/detail/ideal_gas.hpp"
#include "lbt/units/literals.hpp"
#include "lbt/units/units.hpp"


namespace lbt {
  namespace material {

    namespace physical_constants {
      using namespace lbt::literals;

      class Hydrogen {
        public:
          static constexpr auto molecular_weight = 2.016_gpmol;
          static constexpr auto c = 72.0_K;
          static constexpr auto t_0 = 293.85_K;
          static constexpr auto mu_0 = 8.76_uPas;

        protected:
          Hydrogen() = default;
          Hydrogen(Hydrogen const&) = default;
          Hydrogen& operator= (Hydrogen const&) = default;
          Hydrogen(Hydrogen&&) = default;
          Hydrogen& operator= (Hydrogen&&) = default;
      };

    }

    using Hydrogen = IdealGas<physical_constants::Hydrogen>;

  }
}

#endif // LBT__MATERIALS__HYDROGEN
