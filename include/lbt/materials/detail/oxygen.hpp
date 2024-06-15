/**
 * \file     oxygen.hpp
 * \mainpage Contains methods for calculating physical properties of oxygen
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__MATERIALS__OXYGEN
#define LBT__MATERIALS__OXYGEN
#pragma once

#include "lbt/materials/detail/ideal_gas.hpp"
#include "lbt/units/literals.hpp"
#include "lbt/units/units.hpp"


namespace lbt {
  namespace material {

    namespace physical_constants {
      using namespace lbt::literals;

      class Oxygen {
        public:
          static constexpr auto molecular_weight = 31.9988_gpmol;
          static constexpr auto c = 127.0_K;
          static constexpr auto t_0 = 292.25_K;
          static constexpr auto mu_0 = 20.18_uPas;

        protected:
          Oxygen() = default;
          Oxygen(Oxygen const&) = default;
          Oxygen& operator= (Oxygen const&) = default;
          Oxygen(Oxygen&&) = default;
          Oxygen& operator= (Oxygen&&) = default;
      };

    }

    using Oxygen = IdealGas<physical_constants::Oxygen>;

  }
}

#endif // LBT__MATERIALS__OXYGEN
