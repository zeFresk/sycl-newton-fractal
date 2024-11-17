#include <iostream>
#include <chrono>

#include <CL/sycl.hpp>

#include "comp.hpp"
#include "poly.hpp"

using namespace cl;

constexpr auto compute_top_left(auto center, auto inc, auto w, auto h) {
  auto left = inc * static_cast<float>(w / 2);
  auto top = inc * static_cast<float>(h / 2);
  return center + comp_t<decltype(inc)>(-left, -top);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
  static constexpr auto poly = Polynome<float, 4>({-1, 0, 0, 1});
  static constexpr auto deri = poly.derivative();
  static constexpr auto center = comp_t<float>(0, 0);
  static constexpr auto inc = 0.01f;
  static constexpr std::size_t width = 1920;
  static constexpr std::size_t height = 1080;
  static constexpr auto top_left = compute_top_left(center, inc, width, height);
  static constexpr int cycles = 25;

  std::cout << "Initialized with:\n";
  std::cout << "Poly: ";
  for (auto const& p : poly.coeffs())
	  std::cout << p.re << " ";
  std::cout << "\nCenter: " << center.re << " + " << center.im << "i\n";
  std::cout << "Increment: " << inc;
  std::cout << "Window (w, h): (" << width << ", " << height << ")\n";
  std::cout << "cycles: " << cycles << std::endl;

  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
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
  } catch (sycl::exception const &e) {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    std::cout << "Using a CPU device\n";
    d = sycl::device(sycl::cpu_selector_v);
  }

  std::cout << "Using " << d.get_info<sycl::info::device::name>();
  sycl::queue q{d, exception_handler};

  // Print device information
  std::cout << "Device: " << d.get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "Platform: "
            << d.get_platform().get_info<sycl::info::platform::name>()
            << std::endl;
  std::cout << "Vendor: " << d.get_info<sycl::info::device::vendor>()
            << std::endl;
  std::cout << std::flush;

  sycl::buffer<comp_t<float>, 2> zs{sycl::range<2>{width, height}};
  sycl::buffer<comp_t<float>, 2> pzs{sycl::range<2>{width, height}};
  sycl::buffer<comp_t<float>, 2> dpzs{sycl::range<2>{width, height}};


  auto start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor writeZs{zs, cgh, sycl::write_only, sycl::no_init};
    cgh.parallel_for(sycl::range<2>{width, height}, [=](sycl::id<2> id) {
      writeZs[id] = top_left + comp_t<float>(id[0] * inc, id[1] * inc);
    });
  });
  for (int i = 0; i < cycles; ++i) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor apzs{pzs, cgh, sycl::write_only, sycl::no_init};
      sycl::accessor azns{zs, cgh, sycl::read_only};
      cgh.parallel_for(sycl::range<2>{width, height},
                       [=](sycl::id<2> id) { apzs[id] = poly.apply(azns[id]); });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor adpzs{dpzs, cgh, sycl::write_only, sycl::no_init};
      sycl::accessor azns{zs, cgh, sycl::read_only};
      cgh.parallel_for(sycl::range<2>{width, height},
                       [=](sycl::id<2> id) { adpzs[id] = deri.apply(azns[id]); });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor apzs{pzs, cgh, sycl::read_only};
      sycl::accessor adpzs{dpzs, cgh, sycl::read_only};
      sycl::accessor azns{zs, cgh, sycl::read_write};
      cgh.parallel_for(sycl::range<2>{width, height}, [=](sycl::id<2> id) {
        azns[id] =
            adpzs[id].is_zero() ? azns[id] : azns[id] - (apzs[id] / adpzs[id]);
      });
    });
  }

  try {
    q.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = end - start;
  auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1e9;
  static constexpr auto N = poly.coeffs().size();
  static constexpr auto FLOPS_PER_ITEM_PER_ITER = (N+1)*N + N + N*(N-1) + N-1 + 2;
  auto nb_flop = FLOPS_PER_ITEM_PER_ITER * (width * height) * cycles;
  auto flops = nb_flop / elapsed_sec;
  std::cout << "Finished in: " << elapsed_sec << "\n"
	  << "FLOPS: " << flops << std::endl;

  return 0;
}
