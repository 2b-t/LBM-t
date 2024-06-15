/**
 * \file     continuum.hpp
 * \mainpage Class for continuum properties based
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__CONTINUUMS__CONTINUUM
#define LBT__CONTINUUMS__CONTINUUM
#pragma once

#include "lbt/common/use_vtk.hpp"

#ifdef LBT_USE_VTK
  #include "lbt/continuums/vtk_continuum.hpp"

  namespace lbt {
    template <typename T>
    using Continuum = VtkContinuum<T>;
  }
#else
  #include "lbt/continuums/simple_continuum.hpp"

  namespace lbt {
    template <typename T>
    using Continuum = SimpleContinuum<T>;
  }
#endif // LBT_USE_VTK

#endif // LBT__CONTINUUMS__CONTINUUM
