/**
 * \file     population.hpp
 * \mainpage Class for microscopic populations
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__POPULATIONS__POPULATION
#define LBT__POPULATIONS__POPULATION
#pragma once

#include <cstdint>

#include "lbt/populations/aa_population.hpp"


namespace lbt {
  // Convenient alias for population based on A-A access pattern
  template <class LT, std::int32_t NP = 1>
  using Population = AaPopulation<LT, NP>;
}

#endif // LBT__POPULATIONS__POPULATION
