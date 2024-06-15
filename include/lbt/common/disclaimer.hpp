/**
 * \file     disclaimer.hpp
 * \mainpage Get disclaimer and compiler settings
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__COMMON__DISCLAIMER
#define LBT__COMMON__DISCLAIMER
#pragma once

#include <string>


namespace lbt {

  /**\fn     disclaimer
   * \brief  Return a small disclaimer and compiler settings to console
   * 
   * \return A string containing the disclaimer message
  */
  std::string disclaimer() noexcept;

}

#endif // LBT__COMMON__DISCLAIMER
