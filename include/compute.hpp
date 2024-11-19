#pragma once

#include <vector>
#include <chrono>
//#include <mdspan>

#include <CL/sycl.hpp>

#include "poly.hpp"
#include "comp.hpp"

constexpr auto compute_top_left(auto center, auto inc, auto w, auto h) {
	auto left = inc * static_cast<decltype(inc)>(w / 2);
	auto top = inc * static_cast<decltype(inc)>(h / 2);
	return center + comp_t<decltype(inc)>(-left, -top);
}

template <typename T, int N>
class FractalComputer {
	std::array<comp<T>, N - 1> roots;
	Polynome<T, N> poly;
	Polynome<T, N - 1> deri;
	comp<T> center;
	T inc;
	std::size_t width;
	std::size_t height;
	int cycles;

	bool needCompute;
	float lastTimePerComputation;
	float lastFLOPS;

	cl::sycl::device device;
	cl::sycl::queue queue;
	cl::sycl::buffer<comp<T>, 2> zs;
	cl::sycl::buffer<comp<T>, 2> pzs;
	cl::sycl::buffer<comp<T>, 2> dpzs;
	cl::sycl::buffer<T, 3> disroot;
	cl::sycl::buffer<int, 2> closestRoot;
	std::vector<int> cache;

	void resize() {
		zs = cl::sycl::buffer<comp<T>, 2>{ cl::sycl::range<2>{ width, height } };
		pzs = cl::sycl::buffer<comp<T>, 2>{ cl::sycl::range<2>{ width, height } };
		dpzs = cl::sycl::buffer<comp<T>, 2>{ cl::sycl::range<2>{ width, height } };
		disroot = cl::sycl::buffer<T, 3>{ cl::sycl::range<3>{ width, height, N } };
		closestRoot = cl::sycl::buffer<int, 2>{ cl::sycl::range<2>{ width, height } };
		cache.resize(width * height, -1);
	}

    public:
	FractalComputer(std::array<comp<T>, N - 1> const& roots_, comp<T> const& center_, T const& inc_,
			std::size_t width_, std::size_t height_, std::size_t cycles_)
		: roots{ roots_ }, poly{ polynomFromRoots(roots) }, deri{ poly.derivative() }, center{ center_ },
		  inc{ inc_ }, width{ width_ }, height{ height_ }, cycles{ static_cast<int>(cycles_) },
		  needCompute{ true }, lastTimePerComputation{ -1 }, lastFLOPS{ -1 },
		  zs{ cl::sycl::range<2>{ width, height } }, pzs{ cl::sycl::range<2>{ width, height } },
		  dpzs{ cl::sycl::range<2>{ width, height } }, disroot{ cl::sycl::range<3>{ width, height, N } },
		  closestRoot{ cl::sycl::range<2>{ width, height } }, cache(width * height, -1) {
		try {
			std::cout << "GPU...";
			device = cl::sycl::device(cl::sycl::gpu_selector_v);
			std::cout << " ok!" << std::endl;
		} catch (cl::sycl::exception const& e) {
			std::cout << "Cannot select a GPU\n" << e.what() << "\n";
			std::cout << "Using a CPU device\n";
			device = cl::sycl::device(cl::sycl::cpu_selector_v);
		}

		auto exception_handler = [](cl::sycl::exception_list exceptions) {
			for (std::exception_ptr const& e : exceptions) {
				try {
					std::rethrow_exception(e);
				} catch (cl::sycl::exception const& e) {
					std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
				}
			}
		};

		queue = cl::sycl::queue{ device, std::move(exception_handler) };
	}

	template <typename O>
	void printDeviceInfos(O& os) const {
		os << "Device: " << device.get_info<cl::sycl::info::device::name>()
		   << "\nPlatform: " << device.get_platform().get_info<cl::sycl::info::platform::name>()
		   << "\nVendor: " << device.get_info<cl::sycl::info::device::vendor>() << "\n";
	}

	void updatePoly(Polynome<T, N> const& newP) {
		poly = newP;
		deri = newP.derivative();
		roots = poly.roots();
		needCompute = true;
	}

	void updatePolyFromRoots(std::array<comp<T>, N> const& newR) {
		roots = newR;
		poly = polynomFromRoots(newR);
		deri = poly.derivative();
		needCompute = true;
	}

	void updateCenter(comp<T> const& newC) {
		center = newC;
		needCompute = true;
	}

	void updateInc(T const& newI) {
		inc = newI;
		needCompute = true;
	}

	void updateWidth(std::size_t newW) {
		width = newW;
		needCompute = true;
		resize();
	}

	void updateHeight(std::size_t newH) {
		height = newH;
		needCompute = true;
		resize();
	}

	void updateCycles(std::size_t newC) {
		cycles = static_cast<int>(newC);
		needCompute = true;
	}

	Polynome<T, N> const& getPoly() const { return poly; }
	std::array<comp<T>, N - 1> const& getRoots() const { return roots; }
	comp<T> const& getCenter() const { return center; }
	T const& getIncrement() const { return inc; }
	std::size_t getWidth() const { return width; }
	std::size_t getHeight() const { return height; }
	std::size_t getCycles() const { return static_cast<std::size_t>(cycles); }

	void move(comp<T> const& vec) { updateCenter(center + vec); }
	void moveUp(int fac) { move({ 0., inc * fac }); }
	void moveDown(int fac) { move({ 0., -inc * fac }); }
	void moveLeft(int fac) { move({ -inc * fac, 0. }); }
	void moveRight(int fac) { move({ inc * fac, 0. }); }

	void zoomIn(int fac) { updateInc(inc * pw(0.9, fac)); }
	void zoomOut(int fac) { updateInc(inc * pw(1.1, fac)); }

	void increaseIters(int fac) { updateCycles(static_cast<decltype(cycles)>(cycles * pw(1.1, fac))); }
	void decreaseIters(int fac) { updateCycles(static_cast<decltype(cycles)>(cycles * pw(0.9, fac))); }

	std::vector<int> const& compute() {
		using namespace cl;
		if (!needCompute)
			return cache;
		//return std::mdspan(cache.data(), height, width);

		auto start = std::chrono::high_resolution_clock::now();
		auto top_left = compute_top_left(center, inc, width, height);
		auto polyc = this->poly;
		auto inc = this->inc;
		auto deric = this->deri;
		auto rootc = this->roots;

		queue.submit([&](sycl::handler& cgh) {
			sycl::accessor writeZs{ zs, cgh, sycl::write_only, sycl::no_init };
			cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
				writeZs[id] = top_left + comp_t<T>(id[0] * inc, id[1] * inc);
			});
		});
		for (int i = 0; i < cycles; ++i) {
			queue.submit([&](sycl::handler& cgh) {
				sycl::accessor apzs{ pzs, cgh, sycl::write_only, sycl::no_init };
				sycl::accessor azns{ zs, cgh, sycl::read_only };
				cgh.parallel_for(sycl::range<2>{ width, height },
						 [=](sycl::id<2> id) { apzs[id] = polyc.apply(azns[id]); });
			});
			queue.submit([&](sycl::handler& cgh) {
				sycl::accessor adpzs{ dpzs, cgh, sycl::write_only, sycl::no_init };
				sycl::accessor azns{ zs, cgh, sycl::read_only };
				cgh.parallel_for(sycl::range<2>{ width, height },
						 [=](sycl::id<2> id) { adpzs[id] = deric.apply(azns[id]); });
			});
			queue.submit([&](sycl::handler& cgh) {
				sycl::accessor apzs{ pzs, cgh, sycl::read_only };
				sycl::accessor adpzs{ dpzs, cgh, sycl::read_only };
				sycl::accessor azns{ zs, cgh, sycl::read_write };
				cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
					azns[id] = adpzs[id].is_zero() ? azns[id] : azns[id] - (apzs[id] / adpzs[id]);
				});
			});
		}

		queue.submit([&](sycl::handler& cgh) {
			sycl::accessor drw{ disroot, cgh, sycl::write_only, sycl::no_init };
			sycl::accessor azns{ zs, cgh, sycl::read_only };
			cgh.parallel_for(sycl::range<3>{ width, height, roots.size() }, [=](sycl::id<3> id) {
				drw[id] = dist_squared(azns[sycl::id<2>{ id.get(0), id.get(1) }], rootc[id.get(2)]);
			});
		});

		queue.submit([&](sycl::handler& cgh) {
			sycl::accessor crw{ closestRoot, cgh, sycl::write_only, sycl::no_init };
			sycl::accessor adr{ disroot, cgh, sycl::read_only };
			cgh.parallel_for(sycl::range<2>{ width, height }, [=](sycl::id<2> id) {
				crw[id] = 0;
				for (int i = 1; i < (int)rootc.size(); ++i) {
					crw[id] = adr[sycl::id<3>(id[0], id[1], i)] <
								  adr[sycl::id<3>(id[0], id[1], crw[id])] ?
							  i :
							  crw[id];
				}
			});
		});

		try {
			queue.wait_and_throw();
		} catch (sycl::exception const& e) {
			std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = end - start;
		auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1e9;
		static constexpr auto FLOPS_PER_ITEM_PER_ITER =
			(N + 1) * N + N + N * (N - 1) + N - 1 + 2 + (3 * (N - 1)) + ((N - 1) - 1);
		auto nb_flop = FLOPS_PER_ITEM_PER_ITER * (width * height) * cycles;
		auto flops = nb_flop / elapsed_sec;

		lastTimePerComputation = elapsed_sec;
		lastFLOPS = flops;

		auto ha = closestRoot.get_host_access(sycl::read_only);
		std::copy(ha.begin(), ha.end(), cache.begin());
		return cache;
	}
};
