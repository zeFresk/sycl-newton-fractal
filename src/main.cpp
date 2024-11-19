#include <iostream>
#include <chrono>

#include <CL/sycl.hpp>

#include <SFML/Graphics.hpp>

#include "comp.hpp"
#include "poly.hpp"
#include "interface.hpp"
#include "compute.hpp"

using real_t = double;

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
	static constexpr std::array<comp<real_t>, 3> roots{ comp<real_t>{ 1. },
							    comp<real_t>{ -0.5, -0.866025403784439 },
							    comp<real_t>(-0.500000000000000, 0.866025403784439) };
	static constexpr auto center = comp_t<real_t>(-0.4, 0.);
	static constexpr real_t inc = 0.001f;
	static constexpr std::size_t width = 1080;
	static constexpr std::size_t height = 1080;
	static constexpr int cycles = 25;

	auto computer = std::make_shared<FractalComputer<real_t, 4>>( roots, center, inc, width, height, cycles );
	auto interface = Interface{ computer };

	std::cout << "Initialized with:\n";
	std::cout << "Poly: ";
	for (auto const& p : computer->getPoly().coeffs())
		std::cout << p << " ";
	std::cout << "\nRoots: ";
	for (auto const& r : computer->getRoots())
		std::cout << r << " ";
	std::cout << "\nCenter: " << computer->getCenter() << "\n";
	std::cout << "Increment: " << inc << "\n";
	std::cout << "Window (w, h): (" << width << ", " << height << ")\n";
	std::cout << "cycles: " << cycles << std::endl;

	computer->printDeviceInfos(std::cout);

	interface.play();

	return 0;
}
