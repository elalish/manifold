#include <deque>

#include "graph.h"

namespace manifold {

Components ConnectedComponents(const Graph& graph) {
  Components components;
  if (!graph.size()) {
    return components;
  }
  components.nodeLabels.resize(graph.size());
  std::fill(components.nodeLabels.begin(), components.nodeLabels.end(), -1);

  std::deque<int> queue;
  const auto begin = graph.begin();
  const auto end = graph.end();
  for (auto it = begin; it != end; ++it) {
    const int& root = *it;
    if (components.nodeLabels[root] >= 0) continue;  // skip visited nodes

    // new component
    components.nodeLabels[root] = root;
    components.componentLabels.push_back(root);
    queue.emplace_back(root);
    // traverse all connected nodes
    while (!queue.empty()) {
      const auto [n_begin, n_end] = graph.neighbors(queue.front());
      queue.pop_front();
      for (auto n_it = n_begin; n_it != n_end; ++n_it) {
        const int& neighbor = *n_it;
        if (components.nodeLabels[neighbor] < 0) {
          // traversed node is labeled with the root component label
          components.nodeLabels[neighbor] = root;
          queue.emplace_back(neighbor);
        }
      }
    }
  }
  return components;
}
}  // namespace manifold