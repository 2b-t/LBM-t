/**
 * \file     continuum_base.hpp
 * \mainpage Base class for continuum properties
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__CONTINUUMS__CONTINUUM_BASE
#define LBT__CONTINUUMS__CONTINUUM_BASE
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>

#include "lbt/units/units.hpp"


namespace lbt {

  /**\class  ContinuumBase
   * \brief  Base class for the macroscopic variables
   *
   * \tparam T   Floating data type used for simulation
   * \tparam Dummy argument used for SFINAE
  */
  template <typename T>
  class ContinuumBase {
    public:
      /**\fn        setP
       * \brief     Set the pressure to a given value \param value at the coordinates \param x, \param y and \param z
       *
       * \param[in] x       The x-coordinate of cell
       * \param[in] y       The y-coordinate of cell
       * \param[in] z       The z-coordinate of cell
       * \param[in] value   The value the pressure should be set to
      */
      virtual void setP(std::int32_t const x, std::int32_t const y, std::int32_t const z, lbt::unit::Pressure const value) noexcept = 0;

      /**\fn        setU
       * \brief     Set the x-velocity to a given value \param value at the coordinates \param x, \param y and \param z
       *
       * \param[in] x       The x-coordinate of cell
       * \param[in] y       The y-coordinate of cell
       * \param[in] z       The z-coordinate of cell
       * \param[in] value   The value the pressure should be set to
      */
      virtual void setU(std::int32_t const x, std::int32_t const y, std::int32_t const z, lbt::unit::Velocity const value) noexcept = 0;

      /**\fn        setV
       * \brief     Set the y-velocity to a given value \param value at the coordinates \param x, \param y and \param z
       *
       * \param[in] x       The x-coordinate of cell
       * \param[in] y       The y-coordinate of cell
       * \param[in] z       The z-coordinate of cell
       * \param[in] value   The value the pressure should be set to
      */
      virtual void setV(std::int32_t const x, std::int32_t const y, std::int32_t const z, lbt::unit::Velocity const value) noexcept = 0;

      /**\fn        setW
       * \brief     Set the z-velocity to a given value \param value at the coordinates \param x, \param y and \param z
       *
       * \param[in] x       The x-coordinate of cell
       * \param[in] y       The y-coordinate of cell
       * \param[in] z       The z-coordinate of cell
       * \param[in] value   The value the pressure should be set to
      */
      virtual void setW(std::int32_t const x, std::int32_t const y, std::int32_t const z, lbt::unit::Velocity const value) noexcept = 0;

      /**\fn        getP
       * \brief     Get the pressure at the coordinates \param x, \param y and \param z
       *
       * \param[in] x   The x-coordinate of cell
       * \param[in] y   The y-coordinate of cell
       * \param[in] z   The z-coordinate of cell
       * \return    The pressure value at the coordinates \param x, \param y and \param z
      */
      virtual lbt::unit::Pressure getP(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;

      /**\fn        getU
       * \brief     Get the x-velocity at the coordinates \param x, \param y and \param z
       *
       * \param[in] x   The x-coordinate of cell
       * \param[in] y   The y-coordinate of cell
       * \param[in] z   The z-coordinate of cell
       * \return    The x-velocity value at the coordinates \param x, \param y and \param z
      */
      virtual lbt::unit::Velocity getU(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;

      /**\fn        getV
       * \brief     Get the y-velocity at the coordinates \param x, \param y and \param z
       *
       * \param[in] x       The x-coordinate of cell
       * \param[in] y       The y-coordinate of cell
       * \param[in] z       The z-coordinate of cell
       * \return    The y-velocity value at the coordinates \param x, \param y and \param z
      */
      virtual lbt::unit::Velocity getV(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;

      /**\fn        getW
       * \brief     Get the z-velocity at the coordinates \param x, \param y and \param z
       *
       * \param[in] x   The x-coordinate of cell
       * \param[in] y   The y-coordinate of cell
       * \param[in] z   The z-coordinate of cell
       * \return    The z-velocity value at the coordinates \param x, \param y and \param z
      */
      virtual lbt::unit::Velocity getW(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;

      /**\fn    save
       * \brief Function for exporting the domain to a file
       * 
       * \param[in] timestamp   The current timestamp
      */
      virtual void save(double const timestamp) const noexcept = 0;

      /**\fn        initializeUniform
       * \brief     Initialize the continuum with uniform values
       *
       * \param[in] p   The pressure to be set for the entire flow field
       * \param[in] u   The x-velocity to be set to the entire flow field
       * \param[in] v   The y-velocity to be set to the entire flow field
       * \param[in] w   The z-velocity to be set to the entire flow field
      */
      void initializeUniform(lbt::unit::Pressure const p, lbt::unit::Velocity const u, lbt::unit::Velocity const v, lbt::unit::Velocity const w = 0);

    protected:
      /**\fn    ContinuumBase
       * \brief Class constructor
       * 
       * \param[in] NX            Simulation domain resolution in x-direction
       * \param[in] NY            Simulation domain resolution in y-direction
       * \param[in] NZ            Simulation domain resolution in z-direction
       * \param[in] output_path   The path where the output files should be written to
      */
      ContinuumBase(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ, std::filesystem::path const& output_path) noexcept
        : NX{NX}, NY{NY}, NZ{NZ}, output_path{output_path} {
        static_assert(std::is_floating_point_v<T>);
        return;
      }

      ContinuumBase() = delete;
      ContinuumBase(ContinuumBase const&) = default;
      ContinuumBase& operator = (ContinuumBase const&) = default;
      ContinuumBase(ContinuumBase&&) = default;
      ContinuumBase& operator = (ContinuumBase&&) = default;

      std::int32_t NX;
      std::int32_t NY;
      std::int32_t NZ;
      std::filesystem::path output_path;
  };

  template <typename T>
  void ContinuumBase<T>::initializeUniform(lbt::unit::Pressure const p, lbt::unit::Velocity const u,
                                           lbt::unit::Velocity const v, lbt::unit::Velocity const w) {
    for(std::int32_t z = 0; z < NZ; ++z) {
      for(std::int32_t y = 0; y < NY; ++y) {
        for(std::int32_t x = 0; x < NX; ++x) {
          setP(x, y, z, p);
          setU(x, y, z, u);
          setV(x, y, z, v);
          setW(x, y, z, w);
        }
      }
    }
    return;
  }

}

#endif // LBT__CONTINUUMS__CONTINUUM_BASE
