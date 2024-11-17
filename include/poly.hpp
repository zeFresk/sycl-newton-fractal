#pragma once

#include <array>
#include "comp.hpp"

template <typename T>
constexpr T pw(T&& z, int p) {
	if (p == 0)
		return decltype(z){0.};
	if (p == 1)
		return z;

	if (p % 2 == 0) {
		return pw(z*z, p/2);
	} else {
		return z * pw(z*z, (p-1)/2);
	}
}

template <typename T>
using comp_t = comp<T>;

template <class Real, int N> class Polynome {
public:
  constexpr Polynome(std::array<comp_t<Real>, N> &&coeffs)
      : coeffs_(coeffs) {}

  constexpr comp_t<Real> apply_coeff(comp_t<Real> z,
                                           std::size_t idx) const {
    return idx > 0 ? pw(coeffs_[idx] * z, idx) : coeffs_[idx];
  }

  constexpr comp_t<Real> apply(comp_t<Real> z) const {
    decltype(z) ret = 0.f;
    for (int i = 0; i < N; ++i) {
      ret += apply_coeff(z, i);
    }
    return ret;
  }

  constexpr Polynome<Real, N - 1> derivative() const {
    std::array<comp_t<Real>, N - 1> arr;
    for (std::size_t i = 0; i < N - 1; ++i) {
      arr[i] = coeffs_[i + 1] * static_cast<float>(i + 2);
    }
    return Polynome<Real, N - 1>{std::move(arr)};
  }

  constexpr auto const& coeffs() const { return coeffs_; }

private:
  std::array<comp_t<Real>, N> coeffs_;
};
