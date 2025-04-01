#pragma once

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

template <typename Object, typename Distribution> class DistributionDecorator {
public:
  template <typename ProbabilitiesIterator, typename ObjectsIterator>
  DistributionDecorator(
      ProbabilitiesIterator probailities_begin,
      ProbabilitiesIterator probabilities_end,
      ObjectsIterator objects_begin,
      ObjectsIterator objects_end
  )
      : distribution(probailities_begin, probabilities_end),
        objects(objects_begin, objects_end) {}

  template <typename ObjectProbabilityPairIterator>
  DistributionDecorator(ObjectProbabilityPairIterator begin, ObjectProbabilityPairIterator end)
      : distribution(end - begin) {
    StaticArray<double> probabilities(end - begin);

    for (auto pair = begin; pair != end; pair++) {
      auto [obj, prob] = pair;
      objects[pair - begin] = obj;
      probabilities[pair - begin] = prob;
    }
  }

  Object operator()() {
    return objects[distribution()];
  }
  Distribution underlying_distribution() {
    return distribution;
  }

private:
  Distribution distribution;
  StaticArray<Object> objects;
};

} // namespace kaminpar::shm::sparsification
