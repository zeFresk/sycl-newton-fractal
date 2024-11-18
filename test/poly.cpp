#include <gtest/gtest.h>

#include <algorithm>

#include "poly.hpp"

TEST(Polynome, constexpr_init_1) {
	[[maybe_unused]] static constexpr Polynome<float, 1> p{ { 0. } };
	EXPECT_FLOAT_EQ(p.coeffs()[0].re, 0.f);
	EXPECT_FLOAT_EQ(p.coeffs()[0].im, 0.f);
}

TEST(Polynome, constexpr_init_4) { [[maybe_unused]] static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } }; }

TEST(Polynome, constexpr_copy) {
	static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } };
	static constexpr auto p2 = p;
	EXPECT_EQ(p.coeffs(), p2.coeffs());
}

TEST(Polynome, op_equal) {
	static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } };
	Polynome<float, 4> p2{ { 0., 0., 0., 0. } };
	p2 = p;
	EXPECT_EQ(p.coeffs(), p2.coeffs());
}

TEST(Polynome, constexpr_move) {
	static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } };
	static constexpr auto p2{ std::move(p) };
	EXPECT_EQ(p.coeffs(), p2.coeffs());
}

TEST(Polynome, op_equal_move) {
	static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } };
	Polynome<float, 4> p2{ { 0., 0., 0., 0. } };
	p2 = std::move(p);
	EXPECT_EQ(p.coeffs(), p2.coeffs());
}

TEST(Polynome, constexpr_binary_add) {
	static constexpr Polynome<float, 4> p{ { 1., 0., 0., 1. } };
	static constexpr Polynome<float, 4> p2{ { 0., 2., 3., 3. } };
	static constexpr auto r = p + p2;
	static constexpr std::array<comp_t<float>, 4> expected{ { 1., 2., 3., 4. } };
	for (std::size_t i = 0; i < r.coeffs().size(); ++i) {
		EXPECT_FLOAT_EQ(r.coeffs()[i].re, expected[i].re);
		EXPECT_FLOAT_EQ(r.coeffs()[i].im, expected[i].im);
	}
}

TEST(Polynome, constexpr_binary_sub) {
	static constexpr Polynome<float, 4> p{ { 1., 2., 3., 4. } };
	static constexpr Polynome<float, 4> p2{ { 0., 1., 2., 3. } };
	static constexpr auto r = p - p2;
	static constexpr std::array<comp_t<float>, 4> expected{ { 1., 1., 1., 1. } };
	for (std::size_t i = 0; i < r.coeffs().size(); ++i) {
		EXPECT_FLOAT_EQ(r.coeffs()[i].re, expected[i].re);
		EXPECT_FLOAT_EQ(r.coeffs()[i].im, expected[i].im);
	}
}

TEST(Polynome, constexpr_binary_mul) {
	static constexpr Polynome<float, 4> p{ { 3., 7., 4., 1. } }; // x3 4x2 7x 3
	static constexpr Polynome<float, 4> p2{ { -4., -2., -3., 8. } }; // 8x3 -3x2 -2x -4
	static constexpr auto r = p * p2;
	static constexpr std::array<comp_t<float>, 7> expected{ { -12., -34., -39., -9., 42., 29., 8. } };
	for (std::size_t i = 0; i < r.coeffs().size(); ++i) {
		EXPECT_FLOAT_EQ(r.coeffs()[i].re, expected[i].re);
		EXPECT_FLOAT_EQ(r.coeffs()[i].im, expected[i].im);
	}
}

TEST(Polynome, constexpr_degree) {
	static constexpr Polynome<float, 4> p{ { 3., 7., 4., 1. } }; // x3 4x2 7x 3
	static constexpr Polynome<float, 7> r{ { -12., -34., -39., -9., 42., 29., 8. } };
	EXPECT_EQ(p.degree(), 3);
	EXPECT_EQ(r.degree(), 6);
}

TEST(Polynome, constexpr_effective_degree) {
	static constexpr Polynome<float, 4> p{ { 0., 0., 1., 0. } }; // x2
	EXPECT_EQ(p.effective_degree(), 2);
}

TEST(Polynome, constexpr_binary_div) {
	static constexpr Polynome<float, 3> p{ { 0., 0., 1. } }; // x2
	static constexpr Polynome<float, 2> p2{ { 0., 1. } }; // x
	static constexpr auto r = p / p2;
	static constexpr std::array<comp_t<float>, 2> expected{ std::array<comp<float>, 2>{ 0., 1. } };
	for (int i = 0; i < r.effective_degree(); ++i) {
		EXPECT_FLOAT_EQ(r.coeffs()[i].re, expected[i].re);
		EXPECT_FLOAT_EQ(r.coeffs()[i].im, expected[i].im);
	}
}

TEST(Polynome, binary_div) {
	Polynome<float, 3> p{ { 0., 0., 1. } }; // x2
	Polynome<float, 2> p2{ { 0., 1. } }; // x
	auto r = p / p2;
	static constexpr std::array<comp_t<float>, 2> expected{ std::array<comp<float>, 2>{ 0., 1. } };
	for (int i = 0; i < r.effective_degree(); ++i) {
		EXPECT_FLOAT_EQ(r.coeffs()[i].re, expected[i].re);
		EXPECT_FLOAT_EQ(r.coeffs()[i].im, expected[i].im);
	}
}

static constexpr auto epsilon = 1e-12;

TEST(Polynome, root_x) {
	Polynome<float, 2> p{ { 0., 1. } }; // x = 0 => 0
	auto roots = p.roots();
	static constexpr std::array<comp<float>, 1> expected{ 0.0 };
	static constexpr std::array<comp<float>, 1> unexpected{ 1.0 };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, constexpr_root_x) {
	static constexpr Polynome<float, 2> p{ { 0., 1. } }; // x = 0 => 0
	static constexpr auto roots = p.roots();
	static constexpr std::array<comp<float>, 1> expected{ 0.0 };
	static constexpr std::array<comp<float>, 1> unexpected{ 1.0 };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, root_xm1) {
	Polynome<float, 2> p{ { -1, 1. } }; // x -1 = 0 => 1
	auto roots = p.roots();
	static constexpr std::array<comp<float>, 1> expected{ 1.0 };
	static constexpr std::array<comp<float>, 1> unexpected{ 0.0 };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, root_x2m1) {
	Polynome<float, 3> p{ { -1, 0., 1. } }; // x2 -1 = 0 => +-1
	auto uroots = p.roots();
	auto roots = uroots;
	std::sort(roots.begin(), roots.end(), [](auto l, auto r) { return l.re < r.re; });
	static constexpr std::array<comp<float>, 2> expected{ comp<float>(-1.), comp<float>(1.) };
	static constexpr std::array<comp<float>, 2> unexpected{ comp<float>(0.0), comp<float>(0.0) };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, root_x3) {
	Polynome<float, 4> p{ { 0., 0., 0., 1. } }; // x3 -1 = 0 => x = +-1
	auto uroots = p.roots(1000000, comp_t<float>{ 0., 0. });
	auto roots = uroots;
	std::sort(roots.begin(), roots.end(), [](auto l, auto r) { return l.re < r.re; });
	static constexpr std::array<comp<float>, 3> expected{ comp<float>(0.0), comp<float>(0.0), comp<float>(0.0) };
	static constexpr std::array<comp<float>, 3> unexpected{ comp<float>(-1.), comp<float>(1.), comp<float>(1.) };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, constexpr_root_xm1) {
	static constexpr Polynome<float, 2> p{ { -1, 1. } }; // x -1 = 0 => 1
	static constexpr auto roots = p.roots();
	static constexpr std::array<comp<float>, 1> expected{ 1.0 };
	static constexpr std::array<comp<float>, 1> unexpected{ 0.0 };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, constexpr_root_x2m1) {
	static constexpr Polynome<float, 3> p{ { -1, 0., 1. } }; // x2 -1 = 0 => +-1
	static constexpr auto uroots = p.roots();
	auto roots = uroots;
	std::sort(roots.begin(), roots.end(), [](auto l, auto r) { return l.re < r.re; });
	static constexpr std::array<comp<float>, 2> expected{ comp<float>(-1.), comp<float>(1.) };
	static constexpr std::array<comp<float>, 2> unexpected{ comp<float>(0.0), comp<float>(0.0) };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, constexpr_root_x3) {
	static constexpr Polynome<float, 4> p{ { 0., 0., 0., 1. } }; // x3 -1 = 0 => x = +-1
	static constexpr auto uroots = p.roots(1000000, comp_t<float>{ 0., 0. });
	auto roots = uroots;
	std::sort(roots.begin(), roots.end(), [](auto l, auto r) { return l.re < r.re; });
	static constexpr std::array<comp<float>, 3> expected{ comp<float>(0.0), comp<float>(0.0), comp<float>(0.0) };
	static constexpr std::array<comp<float>, 3> unexpected{ comp<float>(-1.), comp<float>(1.), comp<float>(1.) };
	EXPECT_EQ(roots.size(), expected.size());
	for (std::size_t i = 0; i < roots.size(); ++i) {
		EXPECT_NEAR(roots[i].re, expected[i].re, epsilon);
		EXPECT_NEAR(roots[i].im, expected[i].im, epsilon);
		EXPECT_GT(dist_squared(roots[i], unexpected[i]), epsilon);
	}
}

TEST(Polynome, constexpr_polyfromroot) {
	static constexpr auto p = polynomFromRoots(comp<float>{ 1. }, comp<float>{ 2. }, comp<float>{ 3. });
	static constexpr std::array<comp<float>, 3> expected{ comp<float>(1.0), comp<float>(2.0), comp<float>(3.0) };
	for (std::size_t i = 0; i < expected.size(); ++i) {
		auto pz = p.apply(expected[i]);
		EXPECT_LT(dist_squared(pz, comp<float>{ 0. }), epsilon);
	}
}

TEST(Polynome, constexpr_polyfromroot_array) {
	static constexpr auto arr = std::array<comp<float>, 3>{comp<float>{ 1. }, comp<float>{ 2. }, comp<float>{ 3. }};
	static constexpr auto p = polynomFromRoots(arr);
	static constexpr std::array<comp<float>, 3> expected{ comp<float>(1.0), comp<float>(2.0), comp<float>(3.0) };
	for (std::size_t i = 0; i < expected.size(); ++i) {
		auto pz = p.apply(expected[i]);
		EXPECT_LT(dist_squared(pz, comp<float>{ 0. }), epsilon);
	}
}
