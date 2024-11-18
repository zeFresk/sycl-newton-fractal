#pragma once

#include "comp.hpp"
#include <array>
#include <stdexcept>

template <typename T>
constexpr T pw(T&& z, int p) {
	if (p == 0)
		return decltype(z){ 0. };
	if (p == 1)
		return z;

	if (p % 2 == 0) {
		return pw(z * z, p / 2);
	} else {
		return z * pw(z * z, (p - 1) / 2);
	}
}

template <typename T>
using comp_t = comp<T>;

template <class Real, int N>
class Polynome {
    public:
	constexpr Polynome(std::array<comp_t<Real>, N>&& coeffs) : coeffs_(coeffs) {}

	constexpr comp_t<Real> apply_coeff(comp_t<Real> z, std::size_t idx) const {
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
		return Polynome<Real, N - 1>{ std::move(arr) };
	}

	constexpr auto const& coeffs() const { return coeffs_; }
	constexpr int degree() const { return N - 1; }
	constexpr int effective_degree() const {
		auto ret = 0;
		for (int i = 0; i < N; ++i) {
			if (!coeffs_[i].is_zero())
				ret = i;
		}
		return ret;
	}

	constexpr std::array<comp_t<Real>, N - 1> roots(std::size_t max_iters = 10000,
							comp_t<Real> z0 = comp_t<Real>{ 1., 1. }) const {
		std::array<comp_t<Real>, N - 1> ret;
		auto d = derivative();
		auto z = z0;
		int k = 0;
		do {
			for (std::size_t i = 0; i < max_iters; ++i) {
				auto pz = apply(z);
				auto dpz = d.apply(z);
				if (dpz.is_zero() || !dpz.is_normal() || !pz.is_normal() || (pz/dpz).is_zero())
					break;
				z = z - (pz / dpz);
			}
			if (dist_squared(apply(z), comp<Real>{ 0. }) > 1e-32) {
				switch (k++) {
				case 0:
					z = comp<Real>{ Real{z.re}, 0. };
					break;
				case 1:
					z = comp<Real>{ 0., Real{z.im} };
					break;
				case 2:
					z = z * comp<Real>{ 0.5, 0. };
					break;
				case 3:
					z = z * comp<Real>{ 0., 0.5 };
					break;
				case 4:
					z = z * z;
					break;
				default:
					//throw std::runtime_error("Could not converge");
					(void)0;
				}
				max_iters *= 2;
			}
		} while (k < 6 && dist_squared(apply(z), comp<Real>{ 0. }) > 1e-32);

		ret[0] = z;

		if (effective_degree() > 1) {
			auto subp = *this / Polynome<Real, 2>{ { z, 1 } };
			auto subr = subp.roots(max_iters, z0);
			for (std::size_t i = 0; i < ret.size()-1; ++i) {
				ret[i + 1] = subr[i];
			}
		}
		return ret;
	}

    private:
	std::array<comp_t<Real>, N> coeffs_;
};

template <typename Real, int L, int M>
constexpr Polynome<Real, (L > M ? L : M)> operator+(Polynome<Real, L> const& lhs, Polynome<Real, M> const& rhs) {
	constexpr auto K = L > M ? L : M;
	std::array<comp_t<Real>, K> ret;
	for (int i = 0; i < K; ++i) {
		if (L > i)
			ret[i] += lhs.coeffs()[i];
		if (M > i)
			ret[i] += rhs.coeffs()[i];
	}
	return Polynome<Real, K>{ std::move(ret) };
}

template <typename Real, int L, int M>
constexpr Polynome<Real, (L > M ? L : M)> operator-(Polynome<Real, L> const& lhs, Polynome<Real, M> const& rhs) {
	auto coeffs_rhs = rhs.coeffs();
	for (int i = 0; i < M; ++i)
		coeffs_rhs[i] = -rhs.coeffs()[i];
	auto mrhs = Polynome<Real, M>{ std::move(coeffs_rhs) };
	return lhs + mrhs;
}

template <typename Real, int L, int M>
constexpr Polynome<Real, L - 1 + M - 1 + 1> operator*(Polynome<Real, L> const& lhs, Polynome<Real, M> const& rhs) {
	std::array<comp_t<Real>, L - 1 + M - 1 + 1> ret;
	for (std::size_t i = 0; i < L; ++i) {
		for (std::size_t j = 0; j < M; ++j) {
			auto k = i + j;
			auto a = lhs.coeffs()[i] * rhs.coeffs()[j];
			ret[k] += a;
		}
	}
	return Polynome<Real, L - 1 + M - 1 + 1>(std::move(ret));
}

template <typename Real, int M, int Q, int R>
constexpr Polynome<Real, R> impl_div(Polynome<Real, M> const& rhs, Polynome<Real, Q> const& quot,
				     Polynome<Real, R> const& rem) {
	if (rem.effective_degree() <= rhs.effective_degree())
		return rem;

	auto a = rem.coeffs()[R - 1] / rhs.coeffs()[M - 1];
	std::array<comp_t<Real>, R - M + 1> t_arr;
	t_arr[rem.effective_degree() - rhs.effective_degree()] = a;
	auto polyt = Polynome<Real, R - M + 1>{ std::move(t_arr) };
	return impl_div(rhs, quot + polyt, rem - (polyt * rhs));
}

template <typename Real, int L, int M>
constexpr Polynome<Real, L> operator/(Polynome<Real, L> const& lhs, Polynome<Real, M> const& rhs) {
	static_assert(M <= L, "Divisor polynomial can't be larger than dividend");
	return impl_div(rhs, Polynome<Real, 1>{ { 0.0 } }, lhs);
}

template <typename Real>
constexpr Polynome<Real, 2> polynomFromRoots(comp<Real> const& r) {
	std::array<comp<Real>, 2> arr{-r, comp<Real>{1., 0.}};
	return Polynome<Real, 2>{std::move(arr)};
}
template <typename Real, typename... Args>
constexpr Polynome<Real, 1+1+sizeof...(Args)> polynomFromRoots(comp<Real> const& r, Args&&... roots) {
	return polynomFromRoots(r) * polynomFromRoots(std::forward<Args>(roots)...);
}
