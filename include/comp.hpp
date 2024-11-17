#pragma once

#include <cassert>

template <typename T> class comp {
public:
  T re, im;

  constexpr comp(T &&re_ = T{}, T &&im_ = T{}) : re(re_), im(im_) {}
  constexpr comp(comp const &) noexcept = default;
  constexpr comp &operator=(comp const &) noexcept = default;
  constexpr comp(comp &&) noexcept = default;
  constexpr comp &operator=(comp &&) noexcept = default;

  constexpr comp &operator+=(comp const &oth) {
    re += oth.re;
    im += oth.im;
    return *this;
  }
  friend constexpr comp operator+(comp lhs, comp const &rhs) {
    lhs += rhs;
    return lhs;
  }
  constexpr comp &operator-=(comp const &oth) {
    re -= oth.re;
    im -= oth.im;
    return *this;
  }
  friend constexpr comp operator-(comp lhs, comp const &rhs) {
    lhs -= rhs;
    return lhs;
  }

  constexpr comp &operator*=(comp const &oth) {
    re = (re * oth.re) - (im * oth.im);
    im = (re * oth.im) + (im * oth.re);
    return *this;
  }
  friend constexpr comp operator*(comp lhs, comp const &rhs) {
    lhs *= rhs;
    return lhs;
  }

  constexpr comp &operator/=(comp const &oth) {
    auto denom = oth.re * oth.re + oth.im * oth.im;
    assert(denom > 0.);
    re = ((re * oth.re) + (im * oth.im)) / denom;
    im = ((im * oth.re) - (re * oth.im)) / denom;
    return *this;
  }
  friend constexpr comp operator/(comp lhs, comp const &rhs) {
    lhs /= rhs;
    return lhs;
  }

  constexpr bool is_zero() const {
    //return ((im <= 1e-9) && (im >= -(1e-9))) && ((re <= 1e-9) && (re >= -(1e-9)));
    return !((re * re + im * im) > 0.);
  }
};
