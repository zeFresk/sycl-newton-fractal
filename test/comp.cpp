#include <gtest/gtest.h>

#include "comp.hpp"

TEST(Comp, constexpr_init_default) {
	static constexpr comp<float> z;
	EXPECT_FLOAT_EQ(z.re, 0.f);
	EXPECT_FLOAT_EQ(z.im, 0.f);
}

TEST(Comp, constexpr_op_cmpeq) {
	static constexpr comp<float> x{ 1.0, 2.0 };
	static constexpr comp<float> x2{ 1.0, 2.0 };
	static constexpr comp<float> x3{ 1.1, 2.0 };
	EXPECT_EQ(x, x2);
	EXPECT_EQ(x2, x);
	EXPECT_NE(x, x3);
	EXPECT_NE(x3, x);
}

TEST(Comp, constexpr_copy) {
	static constexpr comp<float> x{ 1.0, 2.0 };
	static constexpr comp<float> z{ x };
	EXPECT_FLOAT_EQ(z.re, 1.f);
	EXPECT_FLOAT_EQ(z.im, 2.f);
}

TEST(Comp, op_equal) {
	static constexpr comp<float> x{ 1.0, 2.0 };
	comp<float> z{};
	z = x;
	EXPECT_FLOAT_EQ(z.re, 1.f);
	EXPECT_FLOAT_EQ(z.im, 2.f);
}

TEST(Comp, constexpr_move) {
	static constexpr comp<float> x{ 1.0, 2.0 };
	static constexpr comp<float> z{ std::move(x) };
	EXPECT_FLOAT_EQ(z.re, 1.f);
	EXPECT_FLOAT_EQ(z.im, 2.f);
}

TEST(Comp, op_equal_move) {
	static constexpr comp<float> x{ 1.0, 2.0 };
	comp<float> z{};
	z = std::move(x);
	EXPECT_FLOAT_EQ(z.re, 1.f);
	EXPECT_FLOAT_EQ(z.im, 2.f);
}

TEST(Comp, constexpr_unary_minus) {
	static constexpr comp<float> x{ 1., 2. };
	static constexpr auto z = -x;
	EXPECT_FLOAT_EQ(z.re, -1.f);
	EXPECT_FLOAT_EQ(z.im, -2.f);
}

TEST(Comp, constexpr_binary_add) {
	static constexpr comp<float> x{ 1., 2. };
	static constexpr auto z = x + x;
	EXPECT_FLOAT_EQ(z.re, 2.f);
	EXPECT_FLOAT_EQ(z.im, 4.f);
}

TEST(Comp, constexpr_binary_sub) {
	static constexpr comp<float> x{ 1., 2. };
	static constexpr auto z = x - x;
	EXPECT_FLOAT_EQ(z.re, 0.f);
	EXPECT_FLOAT_EQ(z.im, 0.f);
}

TEST(Comp, constexpr_binary_mul) {
	static constexpr comp<float> x{ 1., 2. };
	static constexpr auto z = x * x;
	EXPECT_FLOAT_EQ(z.re, -3.f);
	EXPECT_FLOAT_EQ(z.im, 4.f);
}

TEST(Comp, constexpr_binary_div) {
	static constexpr comp<float> x{ 1., 2. };
	static constexpr auto z = x / x;
	EXPECT_FLOAT_EQ(z.re, 1.f);
	EXPECT_FLOAT_EQ(z.im, 0.f);
}

TEST(Comp, constexpr_is_zero_false) {
	static constexpr comp<float> x{ 1., 2. };
	EXPECT_FALSE(x.is_zero());
}

TEST(Comp, constexpr_is_zero_true) {
	static constexpr comp<float> x{ 0., 0. };
	EXPECT_TRUE(x.is_zero());
}
