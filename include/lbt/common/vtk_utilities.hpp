/**
 * \file     vtk_utilities.hpp
 * \mainpage Functions for handling geometries with VTK
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__COMMON__VTK_UTILITITES
#define LBT__COMMON__VTK_UTILITITES
#pragma once

#include "lbt/common/use_vtk.hpp"

#ifdef LBT_USE_VTK
  #include <filesystem>
  #include <string>

  #include <vtkImageData.h>
  #include <vtkSmartPointer.h>


  namespace lbt {

    /// Enum class for supported data types
    enum class DataType {VTK, MHD};

    /**\fn        saveImageDataToVtk
     * \brief     Export a scalar to a *.vtk-file
     *
     * \param[in] image_data    The meta-image data to be allocated
     * \param[in] output_path   The output path where the file should be saved
     * \param[in] filename      The filename of the exported file without the file ending
    */
    void saveImageDataToVtk(vtkSmartPointer<vtkImageData> const& image_data, 
                            std::filesystem::path const& output_path, std::string const& filename) noexcept;

    /**\fn        saveImageDataToMhd
     * \brief     Export a scalar to a meta-image *.mhd-file
     *
     * \param[in] image_data    The meta-image data to be allocated
     * \param[in] filename      The filename of the exported file without the file ending
     * \param[in] output_path   The output path where the file should be saved
     * \param[in] is_compress   Boolean variable signaling whether the output should be compressed or not
    */
    void saveImageDataToMhd(vtkSmartPointer<vtkImageData> const& image_data, 
                            std::filesystem::path const& output_path, std::string const& filename, bool const is_compress) noexcept;
  }
#endif // LBT_USE_VTK

#endif // LBT__COMMON__VTK_UTILITITES
