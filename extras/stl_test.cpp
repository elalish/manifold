// benchmark from https://github.com/mikekazakov/pstld/tree/master
// extended with more sort options
#include <cxxabi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/parallel.h"

static constexpr size_t g_Iterations = 5;
static constexpr size_t g_IterationsDiscard = 1;
static constexpr size_t g_Sizes[] = {1'000, 10'000, 100'000, 1'000'000,
                                     10'000'000};

template <class Tp>
inline void noopt(Tp const &value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline void noopt(Tp &value) {
  asm volatile("" : "+r,m"(value) : : "memory");
}

template <class Setup, class Work, class Cleanup>
std::chrono::high_resolution_clock::duration measure(Setup setup, Work work,
                                                     Cleanup cleanup) {
  std::array<std::chrono::high_resolution_clock::duration, g_Iterations> runs;
  for (size_t i = 0; i != g_Iterations; ++i) {
    setup();
    const auto start = std::chrono::high_resolution_clock::now();
    work();
    const auto end = std::chrono::high_resolution_clock::now();
    cleanup();
    runs[i] = end - start;
  }
  std::sort(runs.begin(), runs.end());
  return std::accumulate(runs.begin() + g_IterationsDiscard,
                         runs.end() - g_IterationsDiscard,
                         std::chrono::high_resolution_clock::duration{});
}

template <class Setup, class Work>
std::chrono::high_resolution_clock::duration measure(Setup setup, Work work) {
  return measure(setup, work, [] {});
}

namespace benchmarks {
template <auto Policy>
struct all_of {  // 25.6.1
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size, 42.); },
                   [&] {
                     noopt(manifold::all_of(Policy, v.begin(), v.end(),
                                            [](auto e) { return e < 50.; }));
                   });
  }
};

template <auto Policy>
struct for_each {  // 25.6.4
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size, 42.); },
                   [&] {
                     manifold::for_each(Policy, v.begin(), v.end(),
                                        [](auto &e) { e += 1.; });
                     noopt(v);
                   });
  }
};

template <auto Policy>
struct count {  // 25.6.9
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure(
        [&] { v = std::vector<double>(size, 42.); },
        [&] {
          noopt(manifold::count_if(Policy, v.begin(), v.end(),
                                   [](double v) { return v == 42.; }));
        });
  }
};

template <auto Policy>
struct copy {  // 25.7.1
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v1 = std::vector<double>(size);
          std::generate(std::begin(v1), std::end(v1),
                        [&dist, &mt] { return dist(mt); });
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::copy(Policy, v1.begin(), v1.end(), v2.begin());
          noopt(v2);
        });
  }
};

template <auto Policy>
struct copy_if_28 {  // 25.7.1
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v1 = std::vector<double>(size, 42.);
          for (size_t i = 0; i < size; i++)
            if (dist(mt) <= 0.8) v1[i] = 24;
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::copy_if(Policy, v1.begin(), v1.end(), v2.begin(),
                            [](double v) { return v == 42; });
          noopt(v2);
        });
  }
};

template <auto Policy>
struct copy_if_55 {  // 25.7.1
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v1 = std::vector<double>(size, 42.);
          for (size_t i = 0; i < size; i++)
            if (dist(mt) <= 0.5) v1[i] = 24;
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::copy_if(Policy, v1.begin(), v1.end(), v2.begin(),
                            [](double v) { return v == 42; });
          noopt(v2);
        });
  }
};

template <auto Policy>
struct copy_if_82 {  // 25.7.1
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v1 = std::vector<double>(size, 42.);
          for (size_t i = 0; i < size; i++)
            if (dist(mt) <= 0.2) v1[i] = 24;
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::copy_if(Policy, v1.begin(), v1.end(), v2.begin(),
                            [](double v) { return v == 42; });
          noopt(v2);
        });
  }
};

template <auto Policy>
struct copy_if_1001 {  // 25.7.1
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v1 = std::vector<double>(size, 42.);
          for (size_t i = 0; i < size; i++)
            if (dist(mt) <= 0.01) v1[i] = 24;
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::copy_if(Policy, v1.begin(), v1.end(), v2.begin(),
                            [](double v) { return v == 42; });
          noopt(v2);
        });
  }
};

template <auto Policy>
struct fill {  // 25.7.6
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size); },
                   [&] {
                     manifold::fill(Policy, v.begin(), v.end(), 42.);
                     noopt(v);
                   });
  }
};

template <auto Policy>
struct sort_Rnd {  // 25.8.2.1, semi-random input
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_real_distribution<double> dist{0., 1.};
          v = std::vector<double>(size);
          std::generate(std::begin(v), std::end(v),
                        [&dist, &mt] { return dist(mt); });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Rnd_size {  // 25.8.2.1, semi-random input
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_int_distribution<size_t> dist{
              0, std::numeric_limits<size_t>::max()};
          v = std::vector<size_t>(size);
          std::generate(std::begin(v), std::end(v),
                        [&dist, &mt] { return dist(mt); });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Rnd_size_merge {  // 25.8.2.1, semi-random input
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_int_distribution<size_t> dist{
              0, std::numeric_limits<size_t>::max()};
          v = std::vector<size_t>(size);
          std::generate(std::begin(v), std::end(v),
                        [&dist, &mt] { return dist(mt); });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end(),
                                std::less<size_t>());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Eq {  // 25.8.2.1, equal input
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size, 42.); },
                   [&] {
                     manifold::stable_sort(Policy, v.begin(), v.end());
                     noopt(v);
                   });
  }
};

template <auto Policy>
struct sort_Eq_size {  // 25.8.2.1, equal input
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure([&] { v = std::vector<size_t>(size, 42); },
                   [&] {
                     manifold::stable_sort(Policy, v.begin(), v.end());
                     noopt(v);
                   });
  }
};

template <auto Policy>
struct sort_Eq_size_merge {  // 25.8.2.1, equal input
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure([&] { v = std::vector<size_t>(size, 42); },
                   [&] {
                     manifold::stable_sort(Policy, v.begin(), v.end(),
                                           std::less<size_t>());
                     noopt(v);
                   });
  }
};

template <auto Policy>
struct sort_Asc {  // 25.8.2.1, ascending
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure(
        [&] {
          v = std::vector<double>(size);
          std::iota(v.begin(), v.end(), 0.);
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Asc_size {  // 25.8.2.1, ascending
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          v = std::vector<size_t>(size);
          std::iota(v.begin(), v.end(), 0);
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Asc_size_merge {  // 25.8.2.1, ascending
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          v = std::vector<size_t>(size);
          std::iota(v.begin(), v.end(), 0);
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end(),
                                std::less<size_t>());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_roughly_Asc {  // 25.8.2.1, ascending + random swaps
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_int_distribution<size_t> dist{0, size - 1};
          v = std::vector<double>(size);
          std::iota(v.begin(), v.end(), 0.);
          for (int i = 0; i < std::sqrt(size); i++) {
            std::swap(v[dist(mt)], v[dist(mt)]);
          }
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_roughly_Asc_size {  // 25.8.2.1, ascending + random swaps
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_int_distribution<size_t> dist{0, size - 1};
          v = std::vector<size_t>(size);
          std::iota(v.begin(), v.end(), 0.);
          for (int i = 0; i < std::sqrt(size); i++) {
            std::swap(v[dist(mt)], v[dist(mt)]);
          }
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_roughly_Asc_size_merge {  // 25.8.2.1, ascending + random swaps
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          std::mt19937 mt{42};
          std::uniform_int_distribution<size_t> dist{0, size - 1};
          v = std::vector<size_t>(size);
          std::iota(v.begin(), v.end(), 0.);
          for (int i = 0; i < std::sqrt(size); i++) {
            std::swap(v[dist(mt)], v[dist(mt)]);
          }
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end(),
                                std::less<size_t>());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Des {  // 25.8.2.1, descending
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure(
        [&] {
          v = std::vector<double>(size);
          std::generate(v.begin(), v.end(),
                        [v = std::numeric_limits<double>::max()]() mutable {
                          return v -= 1.;
                        });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Des_size {  // 25.8.2.1, descending
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          v = std::vector<size_t>(size);
          std::generate(v.begin(), v.end(),
                        [v = std::numeric_limits<size_t>::max()]() mutable {
                          return v -= 1;
                        });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end());
          noopt(v);
        });
  }
};

template <auto Policy>
struct sort_Des_size_merge {  // 25.8.2.1, descending
  auto operator()(size_t size) {
    std::vector<size_t> v;
    return measure(
        [&] {
          v = std::vector<size_t>(size);
          std::generate(v.begin(), v.end(),
                        [v = std::numeric_limits<size_t>::max()]() mutable {
                          return v -= 1;
                        });
        },
        [&] {
          manifold::stable_sort(Policy, v.begin(), v.end(),
                                std::less<size_t>());
          noopt(v);
        });
  }
};

template <auto Policy>
struct reduce {  // 25.10.4
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size, 42.); },
                   [&] {
                     noopt(manifold::reduce(Policy, v.begin(), v.end(), 0,
                                            std::plus<double>()));
                   });
  }
};

template <auto Policy>
struct transform_reduce {  // 25.10.6
  auto operator()(size_t size) {
    std::vector<double> v;
    return measure([&] { v = std::vector<double>(size, 42.); },
                   [&] {
                     noopt(manifold::transform_reduce(
                         Policy, v.begin(), v.end(), 0., std::plus<>{},
                         [](auto d) { return d + 1.; }));
                   });
  }
};

template <auto Policy>
struct exclusive_scan {  // 25.10.8
  auto operator()(size_t size) {
    std::vector<double> v1, v2;
    return measure(
        [&] {
          v1 = std::vector<double>(size, 1.01);
          v2 = std::vector<double>(size);
        },
        [&] {
          manifold::exclusive_scan(Policy, v1.begin(), v1.end(), v2.begin(),
                                   1.02, std::multiplies<>{});
          noopt(v1);
        });
  }
};

template <class T>
static std::string demangle() {
  const char *name = typeid(T).name();
  char s[1024];
  size_t len = sizeof(s);
  int status;
  std::string norm = abi::__cxa_demangle(name, s, &len, &status);
  norm.erase(0, std::string_view{"benchmarks::"}.length());
  norm.erase(norm.find_first_of('<'));
  return norm;
}

}  // namespace benchmarks

struct Result {
  std::string name;
  std::array<double, std::size(g_Sizes)> speedups;
};

template <template <auto> class Benchmark>
Result record() {
  auto micro = [](auto d) {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::micro>>(d)
        .count();
  };
  using Seq = Benchmark<manifold::ExecutionPolicy::Seq>;
  using Par = Benchmark<manifold::ExecutionPolicy::Par>;
  Result r;
  r.name = benchmarks::demangle<Seq>();
  for (size_t i = 0; i != std::size(g_Sizes); ++i)
    r.speedups[i] = micro(Seq{}(g_Sizes[i])) / micro(Par{}(g_Sizes[i]));
  return r;
}

int main() {
  std::vector<Result> results;
  results.emplace_back(record<benchmarks::all_of>());
  results.emplace_back(record<benchmarks::for_each>());
  results.emplace_back(record<benchmarks::count>());
  results.emplace_back(record<benchmarks::copy>());
  results.emplace_back(record<benchmarks::copy_if_28>());
  results.emplace_back(record<benchmarks::copy_if_55>());
  results.emplace_back(record<benchmarks::copy_if_82>());
  results.emplace_back(record<benchmarks::copy_if_1001>());
  results.emplace_back(record<benchmarks::fill>());
  results.emplace_back(record<benchmarks::sort_Rnd>());
  results.emplace_back(record<benchmarks::sort_Rnd_size>());
  results.emplace_back(record<benchmarks::sort_Rnd_size_merge>());
  results.emplace_back(record<benchmarks::sort_Eq>());
  // we optimized it to a copy...
  // results.emplace_back(record<benchmarks::sort_Eq_size>());
  results.emplace_back(record<benchmarks::sort_Eq_size_merge>());
  results.emplace_back(record<benchmarks::sort_Asc>());
  // optimized to a copy as well
  // results.emplace_back(record<benchmarks::sort_Asc_size>());
  results.emplace_back(record<benchmarks::sort_Asc_size_merge>());
  results.emplace_back(record<benchmarks::sort_roughly_Asc>());
  results.emplace_back(record<benchmarks::sort_roughly_Asc_size>());
  results.emplace_back(record<benchmarks::sort_roughly_Asc_size_merge>());
  results.emplace_back(record<benchmarks::sort_Des>());
  results.emplace_back(record<benchmarks::sort_Des_size>());
  results.emplace_back(record<benchmarks::sort_Des_size_merge>());
  results.emplace_back(record<benchmarks::reduce>());
  results.emplace_back(record<benchmarks::transform_reduce>());
  results.emplace_back(record<benchmarks::exclusive_scan>());

  const auto max_name_len =
      std::max_element(results.begin(), results.end(), [](auto &a, auto &b) {
        return a.name.length() < b.name.length();
      })->name.length();

  printf("%*s", int(max_name_len + 1), "");
  for (auto s : g_Sizes) {
    if (s >= 1'000'000)
      printf("%4luM ", s / 1'000'000);
    else if (s >= 1'000)
      printf("%4luK ", s / 1'000);
  }
  printf("\n");

  for (auto &r : results) {
    printf("%-*s ", int(max_name_len), r.name.c_str());
    for (auto v : r.speedups) printf("%5.2f ", v);
    printf("\n");
  }
}
