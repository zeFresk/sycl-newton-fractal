#pragma once

#include <cassert>
#include <limits>

template <typename T>
constexpr T mabs(T v) {
	return (v < T{}) ? -v : v;
}

template <typename T>
class comp {
    public:
	T re, im;

	constexpr comp(T&& re_ = T{}, T&& im_ = T{}) : re(re_), im(im_) {}
	constexpr comp(comp const&) noexcept = default;
	constexpr comp& operator=(comp const&) noexcept = default;
	constexpr comp(comp&&) noexcept = default;
	constexpr comp& operator=(comp&&) noexcept = default;

	constexpr comp& operator+=(comp const& oth) {
		re += oth.re;
		im += oth.im;
		return *this;
	}
	friend constexpr comp operator+(comp lhs, comp const& rhs) {
		lhs += rhs;
		return lhs;
	}
	constexpr comp& operator-=(comp const& oth) {
		re -= oth.re;
		im -= oth.im;
		return *this;
	}
	friend constexpr comp operator-(comp lhs, comp const& rhs) {
		lhs -= rhs;
		return lhs;
	}

	constexpr comp operator-() const {
		auto ret = *this;
		ret.re = -re;
		ret.im = -im;
		return ret;
	}

	constexpr comp& operator*=(comp const& oth) {
		auto nre = (re * oth.re) - (im * oth.im);
		auto nim = (re * oth.im) + (im * oth.re);
		re = nre;
		im = nim;
		return *this;
	}
	friend constexpr comp operator*(comp lhs, comp const& rhs) {
		lhs *= rhs;
		return lhs;
	}

	constexpr comp& operator/=(comp const& oth) {
		auto denom = oth.re * oth.re + oth.im * oth.im;
		assert(denom > 0.);
		re = ((re * oth.re) + (im * oth.im)) / denom;
		im = ((im * oth.re) - (re * oth.im)) / denom;
		return *this;
	}
	friend constexpr comp operator/(comp lhs, comp const& rhs) {
		lhs /= rhs;
		return lhs;
	}

	friend constexpr bool operator==(comp const& lhs, comp const& rhs) {
		return lhs.re == rhs.re && lhs.im == rhs.im;
	}

	constexpr bool is_zero(T epsilon = T{}) const {
		//return ((im <= 1e-9) && (im >= -(1e-9))) && ((re <= 1e-9) && (re >= -(1e-9)));
		return !((re * re + im * im) > epsilon * epsilon);
	}

	constexpr bool is_normal() const {
		return re != std::numeric_limits<T>::infinity() && re != -std::numeric_limits<T>::infinity();
	}

	template <typename O>
	friend O& operator<<(O& os, comp const& cpx) {
		if (mabs(cpx.im) < 1e-9) {
			os << cpx.re;
		} else if (mabs(cpx.re) < 1e-9) {
			os << cpx.im << "i";
		} else {
			os << "(" << cpx.re << "+" << cpx.im << "i)";
		}
		return os;
	}
};

template <typename T>
constexpr T dist_squared(comp<T> const& lhs, comp<T> const& rhs) {
	auto l = lhs.re - rhs.re;
	auto r = lhs.im - rhs.im;
	return l * l + r * r;
}
