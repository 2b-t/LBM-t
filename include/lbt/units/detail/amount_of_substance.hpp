/**
 * \file     amount_of_substance.hpp
 * \mainpage Contains unit definition for an amount of substance
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__UNITS__AMOUNT_OF_SUBSTANCE
#define LBT__UNITS__AMOUNT_OF_SUBSTANCE
#pragma once

#include "lbt/units/detail/unit_base.hpp"


namespace lbt {
  namespace unit {

    /**\class AmountOfSubstance
     * \brief Unit class for amount of substance
    */
    class AmountOfSubstance : public lbt::unit::detail::UnitBase<AmountOfSubstance> {
      public:
        /**\fn    AmountOfSubstance
         * \brief Constructor
         * 
         * \param[in] value   The value to be stored inside the class in the base unit kilograms
        */
        explicit constexpr AmountOfSubstance(long double const value = 0.0l) noexcept
          : UnitBase{value} {
          return;
        }
        AmountOfSubstance(AmountOfSubstance const&) = default;
        AmountOfSubstance& operator= (AmountOfSubstance const&) = default;
        AmountOfSubstance(AmountOfSubstance&&) = default;
        AmountOfSubstance& operator= (AmountOfSubstance&&) = default;
    };

  }
}

#endif // LBT__UNITS__AMOUNT_OF_SUBSTANCE
