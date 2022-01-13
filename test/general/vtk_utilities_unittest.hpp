#ifndef LBT_VTK_UTILITITES_UNITTEST
#define LBT_VTK_UTILITITES_UNITTEST
#pragma once


/**\file     vtk_utilities_unittest.hpp
 * \mainpage Unit tests for handling geometries with VTK
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#include <gtest/gtest.h>

#if __has_include (<vtkSmartPointer.h>)
  #include "../../src/general/vtk_utilities.hpp"


  namespace lbt {
    namespace test {

      TEST(VtkUtilities, exportImageDataToVtk) {
        GTEST_SKIP() << "Unit tests involving file import and export not implemented yet!";
      }

      TEST(VtkUtilities, exportImageDataToMhd) {
        GTEST_SKIP() << "Unit tests involving file import and export not implemented yet!";
      }

    }
  }
#endif

#endif // LBT_VTK_UTILITITES_UNITTEST
