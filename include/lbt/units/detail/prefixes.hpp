/**
 * \file     prefixes.hpp
 * \mainpage Contains common SI prefixes for unit conversion
             See https://www.nist.gov/pml/owm/metric-si-prefixes
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__PREFIXES
#define LBT__UNITS__PREFIXES
#pragma once


namespace lbt {
  namespace unit {
    namespace prefix {

      inline constexpr long double quetta {1.0e+30l};
      inline constexpr long double ronna  {1.0e+27l};
      inline constexpr long double yotta  {1.0e+24l};
      inline constexpr long double zetta  {1.0e+21l};
      inline constexpr long double exa    {1.0e+18l};
      inline constexpr long double peta   {1.0e+15l};
      inline constexpr long double tera   {1.0e+12l};
      inline constexpr long double giga   {1.0e+09l};
      inline constexpr long double mega   {1.0e+06l};
      inline constexpr long double kilo   {1.0e+03l};
      inline constexpr long double hecto  {1.0e+02l};
      inline constexpr long double deka   {1.0e+01l};
      inline constexpr long double base   {1.0l};
      inline constexpr long double deci   {1.0e-01l};
      inline constexpr long double centi  {1.0e-02l};
      inline constexpr long double milli  {1.0e-03l};
      inline constexpr long double micro  {1.0e-06l};
      inline constexpr long double nano   {1.0e-09l};
      inline constexpr long double pico   {1.0e-12l};
      inline constexpr long double femto  {1.0e-15l};
      inline constexpr long double atto   {1.0e-18l};
      inline constexpr long double zepto  {1.0e-21l};
      inline constexpr long double yocto  {1.0e-24l};
      inline constexpr long double ronto  {1.0e-27l};
      inline constexpr long double quecto {1.0e-30l};

    }
  }
}

#endif // LBT__UNITS__PREFIXES
