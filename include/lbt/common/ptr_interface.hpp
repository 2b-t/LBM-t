/**
 * \file     ptr_interface.hpp
 * \mainpage Interface class for pointers
 * \author   Tobit Flatscher (github.com/2b-t)
*/

#ifndef LBT__COMMON__PTR_INTERFACE
#define LBT__COMMON__PTR_INTERFACE
#pragma once

#include <memory>
#include <utility>


namespace lbt {

  /**\class  PtrInterface
   * \brief  Interface class that gives easy access to shared pointers
   *         by using CRTP
   *
   * \tparam T   The class that inherits from this interface
  */
  template <typename T>
  class PtrInterface {
    public:
      // Convenient aliases
      using SharedPtr = std::shared_ptr<T>;
      using UniquePtr = std::unique_ptr<T>;

      /**\fn        make_shared
       * \brief     Create a shared pointer
       *
       * \tparam    Args   The variadic arguments
       * \param[in] args   The arguments to be passed to the constructor of T
       * \return    A shared pointer of type T
      */
      template <typename... Args>
      [[nodiscard]]
      static SharedPtr make_shared(Args&&... args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
      }

      /**\fn        make_unique
       * \brief     Create a unique pointer
       *
       * \tparam    Args   The variadic arguments
       * \param[in] args   The arguments to be passed to the constructor of T
       * \return    A unique pointer of type T
      */
      template <typename... Args>
      [[nodiscard]]
      static UniquePtr make_unique(Args&&... args) {
        return std::make_unique<T>(std::forward<Args>(args)...);
      }

    protected:
      PtrInterface() = default;
      PtrInterface(PtrInterface const&) = default;
      PtrInterface(PtrInterface&&) = default;
      PtrInterface& operator=(PtrInterface const&) = default;
      PtrInterface& operator=(PtrInterface&&) = default;
  };

}

#endif // LBT__COMMON__PTR_INTERFACE
