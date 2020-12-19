#ifndef LBT_D2Q9
#define LBT_D2Q9

/**
 * \file     D2Q9.hpp
 * \mainpage Discretisation parameters for D2Q9-lattice
 * \author   Tobit Flatscher (github.com/2b-t)
 *
 * \warning  Static classes with more complex members such as std::vector and std::array require C++17
*/

#include <array>
#include <memory>

#include "../general/constexpr_math.hpp"
#include "../general/memory_alignment.hpp"


namespace lattice
{
    /**\class  lattice::D2Q9P10
     * \brief  Class for D2Q9 lattice with padding to 10
     *
     * \note   "Lattice BGK models for Navier-Stokes equation"
     *         Y.H. Qian, D. D'Humières, P. Lallemand
     *         Europhysics Letters (EPL) Vol. 17 (1992)
     *         DOI: 10.1209/0295-5075/17/6/001
     *
     * \tparam T   Floating data type used for simulation
    */
    template <typename T = double>
    class D2Q9P10 final
    {
        public:
            /// lattice discretisation parameters
            static constexpr unsigned int    DIM =  2;
            static constexpr unsigned int SPEEDS =  9;
            static constexpr unsigned int HSPEED = (SPEEDS + 1)/2;

            /// linear memory layout padding
            static constexpr unsigned int PAD = 1;
            static constexpr unsigned int  ND = SPEEDS + PAD;
            static constexpr unsigned int OFF = ND/2;

            /// discrete velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DX =
            { 0,  1,  0,  1, -1,   // positive velocities
              0, -1,  0, -1,  1 }; // negative velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DY =
            { 0,  0,  1,  1,  1,   // positive velocities
              0,  0, -1, -1, -1 }; // negative velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DZ =
            { 0,  0,  0,  0,  0,
              0,  0,  0,  0,  0 };

            /// corresponding weights
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> W =
            { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,   // positive velocities
              4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0 }; // negative velocities

            /// logical mask for relevant populations
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> MASK =
            { 1, 1, 1, 1, 1,
              0, 1, 1, 1, 1 };

            /// lattice speed of sound
            static constexpr T CS = 1.0/cem::sqrt(3.0);
    };

    /// Alias default lattice
    template<typename T> using D2Q9 = D2Q9P10<T>;


    /**\class  lattice::D2Q9P12
     * \brief  Class for D2Q9 lattice with padding to 12
     *
     * \note   "Lattice BGK models for Navier-Stokes equation"
     *         Y.H. Qian, D. D'Humières, P. Lallemand
     *         Europhysics Letters (EPL) Vol. 17 (1992)
     *         DOI: 10.1209/0295-5075/17/6/001
     *
     * \tparam T   Floating data type used for simulation
    */
    template <typename T = double>
    class D2Q9P12 final
    {
        public:
            /// lattice discretisation parameters
            static constexpr unsigned int    DIM =  2;
            static constexpr unsigned int SPEEDS =  9;
            static constexpr unsigned int HSPEED = (SPEEDS + 1)/2;

            /// linear memory layout padding
            static constexpr unsigned int PAD = 3;
            static constexpr unsigned int  ND = SPEEDS + PAD;
            static constexpr unsigned int OFF = ND/2;

            /// discrete velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DX =
            { 0,  1,  0,  1, -1,  0,   // positive velocities
              0, -1,  0, -1,  1,  0 }; // negative velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DY =
            { 0,  0,  1,  1,  1,  0,   // positive velocities
              0,  0, -1, -1, -1,  0 }; // negative velocities
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> DZ =
            { 0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0 };

            /// corresponding weights
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> W =
            { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,  0.0,   // positive velocities
              4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,  0.0 }; // negative velocities

            /// logical mask for relevant populations
            __attribute__((aligned(CACHE_LINE))) alignas(CACHE_LINE) static constexpr std::array<T, ND> MASK =
            { 1, 1, 1, 1, 1, 0,
              0, 1, 1, 1, 1, 0 };

            /// lattice speed of sound
            static constexpr T CS = 1.0/cem::sqrt(3.0);
    };
}

#endif // LBT_D2Q9
