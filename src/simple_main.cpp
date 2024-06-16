/**
 * \file     main.cpp
 * \mainpage Mainpage of the LB-t computational fluid dynamics solver based on the incompressible Lattice-Boltzmann method
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>

#include "lbt/common/openmp_manager.hpp"
#include "lbt/continuums/continuum.hpp"
#include "lbt/lattices/lattices.hpp"
#include "lbt/materials/materials.hpp"
#include "lbt/populations/collision_operators.hpp"
#include "lbt/populations/population.hpp"
#include "lbt/units/characteristic_numbers.hpp"
#include "lbt/units/literals.hpp"
#include "lbt/converter.hpp"


int main(int argc, char* argv[]) {
  using namespace lbt::literals;

  // Flow and material properties
  constexpr auto velocity {1.7_mps};
  constexpr auto length {1.2_m};
  using Material = lbt::material::Air;
  constexpr auto temperature {20.0_deg};
  constexpr auto pressure {1.0_atm};
  constexpr auto density {Material::equationOfState(temperature, pressure)};
  constexpr auto kinematic_viscosity {Material::kinematicViscosity(temperature, pressure)};
  constexpr lbt::unit::ReynoldsNumber reynolds_number {velocity, length, kinematic_viscosity};

  // Solver settings: Floating type, lattice and collision operator
  constexpr auto simulation_time {3.0_min};
  constexpr auto save_time_step {simulation_time/100.0};
  using T = double;
  using Lattice = lbt::lattice::D2Q9<T>;
  using CollisionOperator = lbt::collision::Bgk<Lattice>;
  constexpr auto lbm_speed_of_sound {Lattice::CS};
  constexpr auto lbm_velocity {0.5*lbm_speed_of_sound};
  constexpr double lbm_density {1.0};
  constexpr std::int32_t NX {100};
  constexpr std::int32_t NY {100};
  constexpr std::int32_t NZ {100};
  constexpr double lbm_length {static_cast<long double>(NX)};
  auto const output_path {std::filesystem::current_path()};

  auto const unit_converter {std::make_shared<lbt::Converter>(
    length, lbm_length,
    velocity, lbm_velocity,
    density, lbm_density
  )};

  // Import geometry either from scenario or using VTK

  auto& omp_manager = lbt::OpenMpManager::getInstance();
  omp_manager.setThreadsNum(lbt::OpenMpManager::getThreadsMax());

  // Continuum, population and collision operator
  auto continuum {std::make_shared<lbt::Continuum<T>>(NX, NY, NZ, output_path)};
  continuum->initializeUniform(pressure, velocity, 0.0_mps, 0.0_mps);
  // continuum->initialize(lambda)
  // Initialize with lambda, allow user to use lambda with normalized coordinates [0, 1]
  // Prepare lambdas for uniform and Poiseuille profile etc.

  auto population {std::make_shared<lbt::Population<Lattice>>(NX, NY, NZ)};
  auto collision_operator {std::make_shared<CollisionOperator>(
    population, continuum, unit_converter,
    reynolds_number, lbm_velocity, lbm_length
  )};
  collision_operator->initialize<lbt::Timestep::Even>();

  // Geometry and boundary conditions
  // std::vector<BoundaryCondition> boundary_conditions {};
  // Append boundary conditions, allow specifying range for boundary condition easily

  // Convert time-steps to simulation time
  auto const NT {static_cast<std::int64_t>(unit_converter->toLbm(simulation_time))};
  auto const NT_SAVE {static_cast<std::int64_t>(unit_converter->toLbm(save_time_step))};
  for (std::int64_t i = 0; i < NT; i += 2) {
    // Loop performs two combined iterations at once

    bool is_save {(i % NT_SAVE) == 0};
    collision_operator->collideAndStream<lbt::Timestep::Even>(is_save);
    // Enforce boundary condition on continuum

    is_save = (((i + 1) % NT_SAVE) == 0);
    collision_operator->collideAndStream<lbt::Timestep::Odd>(is_save);
    // Enforce boundary condition on continuum

    if (is_save) {
      auto const simulation_time {unit_converter->toPhysical<lbt::unit::Time>(i)};
      continuum->save(static_cast<double>(simulation_time.get()));
    }
  }

  return EXIT_SUCCESS;
}
