/**
 * \file     type.hpp
 * \mainpage Holds enum for types of boundary conditions
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__BOUNDARIES__TYPE
#define LBT__BOUNDARIES__TYPE
#pragma once


namespace lbt {
  namespace boundary {

    /**\enum  Type
     * \brief Strongly typed for different boundary condition types
    */
    enum class Type {
      Velocity,
      Pressure
    };

  }
}

#endif // LBT__BOUNDARIES__TYPE
