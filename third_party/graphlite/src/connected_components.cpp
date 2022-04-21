#include <deque>

#include "graph.h"

namespace manifold {

int ConnectedComponents(std::vector<int>& components, const Graph& graph) {
  if (!graph.size()) {
    return 0;
  }
  components.resize(graph.size());
  std::fill(components.begin(), components.end(), -1);

  std::deque<int> queue;
  const auto begin = graph.begin();
  const auto end = graph.end();
  int numComponent = 0;
  for (auto it = begin; it != end; ++it) {
    const int& root = *it;
    if (components[root] >= 0) continue;  // skip visited nodes

    // new component
    components[root] = numComponent;
    queue.emplace_back(root);
    // traverse all connected nodes
    while (!queue.empty()) {
      const auto [n_begin, n_end] = graph.neighbors(queue.front());
      queue.pop_front();
      for (auto n_it = n_begin; n_it != n_end; ++n_it) {
        const int& neighbor = *n_it;
        if (components[neighbor] < 0) {
          // traversed node is labeled with the root component label
          components[neighbor] = numComponent;
          queue.emplace_back(neighbor);
        }
      }
    }
    ++numComponent;
  }
  return numComponent;
}
}  // namespace manifold