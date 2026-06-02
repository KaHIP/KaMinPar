/*******************************************************************************
 * Implements the explicit downcasts of abstract graphs to concrete graphs.
 *
 * @file:   reify_graph.h
 * @author: Daniel Seemaier
 * @date:   25.12.2024
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "kaminpar-dist/datastructures/abstract_distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist::graph {

template <typename Lambda>
decltype(auto) reified(const AbstractDistributedGraph &graph, Lambda &&l) {
  if (auto *csr_graph = dynamic_cast<const DistributedCSRGraph *>(&graph); csr_graph != nullptr) {
    return l(*csr_graph);
  } else if (
      auto *compressed_graph = dynamic_cast<const DistributedCompressedGraph *>(&graph);
      compressed_graph != nullptr
  ) {
    return l(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename ConcreteGraph> [[nodiscard]] bool is(const AbstractDistributedGraph &graph) {
  return dynamic_cast<const ConcreteGraph *>(&graph) != nullptr;
}

template <typename ConcreteGraph>
[[nodiscard]] ConcreteGraph &as_concrete_graph(const AbstractDistributedGraph &graph) {
  KASSERT(is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name());
  return *static_cast<ConcreteGraph *>(&graph);
}

template <typename ConcreteGraph>
ConcreteGraph &as_concrete_graph(AbstractDistributedGraph &graph) {
  KASSERT(is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name());
  return dynamic_cast<ConcreteGraph &>(graph);
}

template <template <typename> typename Component> struct ReifiedGraphComponent {
  using ComponentVariant = std::variant<
      std::monostate,
      Component<DistributedCSRGraph>,
      Component<DistributedCompressedGraph>>;

  ComponentVariant obj;

  [[nodiscard]] bool empty() const {
    return std::holds_alternative<std::monostate>(obj);
  }

  template <typename ConcreteGraph> [[nodiscard]] bool holds() const {
    return std::holds_alternative<Component<ConcreteGraph>>(obj);
  }

  template <typename ConcreteGraph, typename... Args>
  Component<ConcreteGraph> &emplace(Args &&...args) {
    return obj.template emplace<Component<ConcreteGraph>>(std::forward<Args>(args)...);
  }

  template <typename ConcreteGraph, typename... Args>
  Component<ConcreteGraph> &ensure(Args &&...args) {
    if (!holds<ConcreteGraph>()) {
      return emplace<ConcreteGraph>(std::forward<Args>(args)...);
    }

    return get<ConcreteGraph>();
  }

  template <typename ConcreteGraph> Component<ConcreteGraph> &get() {
    return std::get<Component<ConcreteGraph>>(obj);
  }

  template <typename ConcreteGraph> const Component<ConcreteGraph> &get() const {
    return std::get<Component<ConcreteGraph>>(obj);
  }

  template <typename Lambda> void if_present(Lambda &&l) {
    std::visit(
        [&](auto &component) {
          using ComponentT = std::remove_cvref_t<decltype(component)>;
          if constexpr (!std::is_same_v<ComponentT, std::monostate>) {
            std::forward<Lambda>(l)(component);
          }
        },
        obj
    );
  }

  template <typename Lambda> void if_present(Lambda &&l) const {
    std::visit(
        [&](const auto &component) {
          using ComponentT = std::remove_cvref_t<decltype(component)>;
          if constexpr (!std::is_same_v<ComponentT, std::monostate>) {
            std::forward<Lambda>(l)(component);
          }
        },
        obj
    );
  }

  template <typename GraphLike, typename Lambda>
  decltype(auto) with(GraphLike &&graph, Lambda &&l) {
    return dispatch(std::forward<GraphLike>(graph), [&](auto &&concrete_graph) -> decltype(auto) {
      using ConcreteGraph = std::remove_cvref_t<decltype(concrete_graph)>;
      return std::forward<Lambda>(l)(
          get<ConcreteGraph>(), std::forward<decltype(concrete_graph)>(concrete_graph)
      );
    });
  }

  template <typename GraphLike, typename Lambda>
  decltype(auto) with(GraphLike &&graph, Lambda &&l) const {
    return dispatch(std::forward<GraphLike>(graph), [&](auto &&concrete_graph) -> decltype(auto) {
      using ConcreteGraph = std::remove_cvref_t<decltype(concrete_graph)>;
      return std::forward<Lambda>(l)(
          get<ConcreteGraph>(), std::forward<decltype(concrete_graph)>(concrete_graph)
      );
    });
  }

private:
  template <typename GraphLike, typename Lambda>
  static decltype(auto) dispatch(GraphLike &&graph, Lambda &&l) {
    if constexpr (requires { std::forward<GraphLike>(graph).reified(std::forward<Lambda>(l)); }) {
      return std::forward<GraphLike>(graph).reified(std::forward<Lambda>(l));
    } else {
      return reified(std::forward<GraphLike>(graph), std::forward<Lambda>(l));
    }
  }
};

} // namespace kaminpar::dist::graph

namespace kaminpar::dist {

template <template <typename> typename Component>
using ReifiedGraphComponent = graph::ReifiedGraphComponent<Component>;

} // namespace kaminpar::dist
