#include <iostream>
#include <chrono>

#include <CL/sycl.hpp>

#include <SFML/Graphics.hpp>

#include "comp.hpp"
#include "poly.hpp"

using namespace cl;

constexpr auto compute_top_left(auto center, auto inc, auto w, auto h) {
	auto left = inc * static_cast<float>(w / 2);
	auto top = inc * static_cast<float>(h / 2);
	return center + comp_t<decltype(inc)>(-left, -top);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
	static constexpr std::array<comp<float>, 3> roots{ comp<float>{ 1. }, comp<float>{ -0.5, -0.866025403784439 },
							   comp<float>(-0.500000000000000, 0.866025403784439) };
	static constexpr auto poly = polynomFromRoots(roots);
	static constexpr auto deri = poly.derivative();
	static constexpr auto center = comp_t<float>(0., 0.);
	static constexpr auto inc = 0.01f;
	static constexpr std::size_t width = 1080;
	static constexpr std::size_t height = 1080;
	static constexpr auto top_left = compute_top_left(center, inc, width, height);
	static constexpr int cycles = 1000;

	std::cout << "Initialized with:\n";
	std::cout << "Poly: ";
	for (auto const& p : poly.coeffs())
		std::cout << p.re << " ";
	std::cout << "\nRoots: ";
	for (auto const& r : roots)
		std::cout << "(" << r.re << "+" << r.im << ") ";
	std::cout << "\nCenter: " << center.re << " + " << center.im << "i\n";
	std::cout << "Increment: " << inc << "\n";
	std::cout << "Window (w, h): (" << width << ", " << height << ")\n";
	std::cout << "cycles: " << cycles << std::endl;

	auto exception_handler = [](sycl::exception_list exceptions) {
		for (std::exception_ptr const& e : exceptions) {
			try {
				std::rethrow_exception(e);
			} catch (sycl::exception const& e) {
				std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
			}
		}
	};

	// sycl::default_selector selector{};
	// sycl::gpu_selector selector{};
	sycl::device d;
	try {
		std::cout << "GPU...";
		d = sycl::device(sycl::gpu_selector_v);
		std::cout << " ok!" << std::endl;
	} catch (sycl::exception const& e) {
		std::cout << "Cannot select a GPU\n" << e.what() << "\n";
		std::cout << "Using a CPU device\n";
		d = sycl::device(sycl::cpu_selector_v);
	}

	std::cout << "Using " << d.get_info<sycl::info::device::name>();
	sycl::queue q{ d, exception_handler };

	// Print device information
	std::cout << "Device: " << d.get_info<sycl::info::device::name>() << std::endl;
	std::cout << "Platform: " << d.get_platform().get_info<sycl::info::platform::name>() << std::endl;
	std::cout << "Vendor: " << d.get_info<sycl::info::device::vendor>() << std::endl;
	std::cout << std::flush;

	sycl::buffer<comp_t<float>, 2> zs{ sycl::range<2>{ width, height } };
	sycl::buffer<comp_t<float>, 2> pzs{ sycl::range<2>{ width, height } };
	sycl::buffer<comp_t<float>, 2> dpzs{ sycl::range<2>{ width, height } };
	sycl::buffer<float, 3> disroot{ sycl::range<3>{ width, height, roots.size() } };
	sycl::buffer<int, 2> closestRoot{ sycl::range<2>{ width, height } };

	auto start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler& cgh) {
		sycl::accessor writeZs{ zs, cgh, sycl::write_only, sycl::no_init };
		cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
			writeZs[id] = top_left + comp_t<float>(id[0] * inc, id[1] * inc);
		});
	});
	for (int i = 0; i < cycles; ++i) {
		q.submit([&](sycl::handler& cgh) {
			sycl::accessor apzs{ pzs, cgh, sycl::write_only, sycl::no_init };
			sycl::accessor azns{ zs, cgh, sycl::read_only };
			cgh.parallel_for(sycl::range<2>{ width, height },
					 [=](sycl::id<2> id) { apzs[id] = poly.apply(azns[id]); });
		});
		q.submit([&](sycl::handler& cgh) {
			sycl::accessor adpzs{ dpzs, cgh, sycl::write_only, sycl::no_init };
			sycl::accessor azns{ zs, cgh, sycl::read_only };
			cgh.parallel_for(sycl::range<2>{ width, height },
					 [=](sycl::id<2> id) { adpzs[id] = deri.apply(azns[id]); });
		});
		q.submit([&](sycl::handler& cgh) {
			sycl::accessor apzs{ pzs, cgh, sycl::read_only };
			sycl::accessor adpzs{ dpzs, cgh, sycl::read_only };
			sycl::accessor azns{ zs, cgh, sycl::read_write };
			cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
				azns[id] = adpzs[id].is_zero() ? azns[id] : azns[id] - (apzs[id] / adpzs[id]);
			});
		});
	}

	q.submit([&](sycl::handler& cgh) {
		sycl::accessor drw{ disroot, cgh, sycl::write_only, sycl::no_init };
		sycl::accessor azns{ zs, cgh, sycl::read_only };
		cgh.parallel_for(sycl::range<3>{ width, height, roots.size() }, [=](sycl::id<3> id) {
			drw[id] = dist_squared(azns[sycl::id<2>{ id.get(0), id.get(1) }], roots[id.get(2)]);
		});
	});

	q.submit([&](sycl::handler& cgh) {
		sycl::accessor crw{ closestRoot, cgh, sycl::write_only, sycl::no_init };
		sycl::accessor adr{ disroot, cgh, sycl::read_only };
		cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
			crw[id] = 0;
			for (int i = 1; i < (int)roots.size(); ++i) {
				crw[id] = adr[sycl::id<3>(id[0], id[1], i)] < adr[sycl::id<3>(id[0], id[1], crw[id])] ?
						  i :
						  crw[id];
			}
		});
	});

	try {
		q.wait_and_throw();
	} catch (sycl::exception const& e) {
		std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1e9;
	static constexpr auto N = poly.coeffs().size();
	static constexpr auto FLOPS_PER_ITEM_PER_ITER =
		(N + 1) * N + N + N * (N - 1) + N - 1 + 2 + (3 * roots.size()) + (roots.size() - 1);
	auto nb_flop = FLOPS_PER_ITEM_PER_ITER * (width * height) * cycles;
	auto flops = nb_flop / elapsed_sec;
	std::cout << "Finished in: " << elapsed_sec << "\n"
		  << "FLOPS: " << flops << std::endl;

	auto ha = closestRoot.get_host_access(sycl::read_only);
	std::vector<int> out(ha.begin(), ha.end());

	if ((width * height) < 128) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; ++j)
				std::cout << ha[sycl::id<2>(i, j)] << " ";
			std::cout << "\n";
		}
	}

	std::map<int, int> counts;
	for (auto r : out) {
		auto it = counts.find(r);
		counts[r] = (it == counts.end()) ? 1 : it->second + 1;
	}

	std::cout << "Found:\n";
	for (auto const& [k, v] : counts)
		std::cout << "[" << k << "]: " << v << "\n";

	std::array<unsigned char, width * height * 4> pix;
	for (std::size_t i = 0; i < out.size(); ++i) {
		auto idx = i * 4;
		pix[idx + 0] = (out[i] == 0) ? 255 : 0;
		pix[idx + 1] = (out[i] == 1) ? 255 : 0;
		pix[idx + 2] = (out[i] == 2) ? 255 : 0;
		pix[idx + 3] = 255;
	}

	sf::RenderWindow window(sf::VideoMode(width, height), "Newton fractal viewer");
	sf::Texture texture;
	texture.create(width, height);
	sf::Sprite sprite(texture);
	texture.update(pix.data());

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			} else if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Escape) {
					window.close();
				} /* else if (event.key.code == sf::Keyboard::Left) {
					center_.re -= 10.0f * inc; // Adjust panning speed based on zoom
				} else if (event.key.code == sf::Keyboard::Right) {
					center_.re += 10.0f * inc;
				} else if (event.key.code == sf::Keyboard::Up) {
					center_.im -= 10.0f * inc;
				} else if (event.key.code == sf::Keyboard::Down) {
					centerY += 10.0f / zoom;
				} else if (event.key.code == sf::Keyboard::Add ||
					   event.key.code == sf::Keyboard::Equal) {
					zoom += 0.1f;
				} else if (event.key.code == sf::Keyboard::Subtract ||
					   event.key.code == sf::Keyboard::Dash) {
					zoom -= 0.1f;
					zoom = std::max(0.1f, zoom); // Prevent zoom from going below 0.1
				}
			} else if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Left) {
					// Handle mouse drag start (optional)
				}
			} else if (event.type == sf::Event::MouseButtonReleased) {
				if (event.mouseButton.button == sf::Mouse::Left) {
					// Handle mouse drag end (optional)
				}
			} else if (event.type == sf::Event::MouseMoved) {
				if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					// Update centerX and centerY based on mouse movement
					centerX = event.mouseMove.x;
					centerY = event.mouseMove.y;
				}*/
			}
		}
		window.clear();
		window.draw(sprite);
		window.display();
	}
	return 0;
}
