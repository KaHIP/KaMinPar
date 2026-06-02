/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   graph.h
 * @author: Daniel Seemaier
 * @date:   17.11.2023
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

template <typename Lambda> decltype(auto) reified(Graph &graph, Lambda &&l) {
  AbstractGraph *abstract_graph = graph.underlying_graph();

  if (auto *csr_graph = dynamic_cast<CSRGraph *>(abstract_graph); csr_graph != nullptr) {
    return std::forward<Lambda>(l)(*csr_graph);
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(abstract_graph);
      compressed_graph != nullptr) {
    return std::forward<Lambda>(l)(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename Lambda> decltype(auto) reified(const Graph &graph, Lambda &&l) {
  const AbstractGraph *abstract_graph = graph.underlying_graph();

  if (const auto *csr_graph = dynamic_cast<const CSRGraph *>(abstract_graph);
      csr_graph != nullptr) {
    return std::forward<Lambda>(l)(*csr_graph);
  }

  if (const auto *compressed_graph = dynamic_cast<const CompressedGraph *>(abstract_graph);
      compressed_graph != nullptr) {
    return std::forward<Lambda>(l)(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename ConcreteGraph> [[nodiscard]] bool is(const Graph &graph) {
  return dynamic_cast<const ConcreteGraph *>(graph.underlying_graph()) != nullptr;
}

template <typename ConcreteGraph> [[nodiscard]] ConcreteGraph &as_concrete_graph(Graph &graph) {
  KASSERT(is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name());

  return *static_cast<ConcreteGraph *>(graph.underlying_graph());
}

template <typename ConcreteGraph>
[[nodiscard]] const ConcreteGraph &as_concrete_graph(const Graph &graph) {
  KASSERT(is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name());

  return *static_cast<const ConcreteGraph *>(graph.underlying_graph());
}

/*!
 * Encapsulates an object of a class `Component` that should be instantiated the concrete graph
 * classes, e.g., CSRGraph or CompressedGraph.
 *
 * `Component` may only take one template argument: the concretized graph class.
 */
template <template <typename> typename Component> struct ReifiedGraphComponent {
  using ComponentVariant =
      std::variant<std::monostate, Component<CSRGraph>, Component<CompressedGraph>>;

  ComponentVariant obj;

  [[nodiscard]] bool empty() const {
    return std::holds_alternative<std::monostate>(obj);
  }

  template <typename ConcretizedGraph> [[nodiscard]] bool holds() const {
    return std::holds_alternative<Component<ConcretizedGraph>>(obj);
  }

  /*!
   * Emplaces a `Component<ConcretizedGraph>` object.
   *
   * @param args Forwarded to the `Component` ctor.
   * @tparam ConcretizedGraph The concretized graph class.
   *
   * @return Reference to the emplaced object.
   */
  template <typename ConcretizedGraph, typename... Args>
  Component<ConcretizedGraph> &emplace(Args &&...args) {
    return obj.template emplace<Component<ConcretizedGraph>>(std::forward<Args>(args)...);
  }

  template <typename ConcretizedGraph, typename... Args>
  Component<ConcretizedGraph> &ensure(Args &&...args) {
    if (!holds<ConcretizedGraph>()) {
      return emplace<ConcretizedGraph>(std::forward<Args>(args)...);
    }

    return get<ConcretizedGraph>();
  }

  /*!
   * Returns a reference to the emplaced object. Must be compatible to the previous `emplace()`
   * call.
   *
   * @return Reference to the emplaced object.
   */
  template <typename ConcretizedGraph> Component<ConcretizedGraph> &get() {
    return std::get<Component<ConcretizedGraph>>(obj);
  }

  template <typename ConcretizedGraph> const Component<ConcretizedGraph> &get() const {
    return std::get<Component<ConcretizedGraph>>(obj);
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
    return dispatch(
        std::forward<GraphLike>(graph), [&](auto &&concretized_graph) -> decltype(auto) {
          using ConcretizedGraph = std::remove_cvref_t<decltype(concretized_graph)>;
          return std::forward<Lambda>(l)(
              get<ConcretizedGraph>(), std::forward<decltype(concretized_graph)>(concretized_graph)
          );
        }
    );
  }

  template <typename GraphLike, typename Lambda>
  decltype(auto) with(GraphLike &&graph, Lambda &&l) const {
    return dispatch(
        std::forward<GraphLike>(graph), [&](auto &&concretized_graph) -> decltype(auto) {
          using ConcretizedGraph = std::remove_cvref_t<decltype(concretized_graph)>;
          return std::forward<Lambda>(l)(
              get<ConcretizedGraph>(), std::forward<decltype(concretized_graph)>(concretized_graph)
          );
        }
    );
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

} // namespace kaminpar::shm
