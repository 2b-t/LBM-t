/**
 * \file     air.hpp
 * \mainpage Contains methods for calculating physical properties of air
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT_MATERIAL_AIR
#define LBT_MATERIAL_AIR
#pragma once

#include "../../unit/literals.hpp"
#include "../../unit/units.hpp"
#include "ideal_gas.hpp"


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
      };

    }

    using Air = IdealGas<physical_constants::Air>;

  }
}

#endif // LBT_MATERIAL_AIR
