// A random distribution over {0, ..., n-1} with probailies propotional to given values
// Implemented using the alias Method


#include "kaminpar-common/random.h"
namespace kaminpar::shm::sparsification {
template <typename T> class FiniteRandomDistribution {
public:
  FiniteRandomDistribution(std::initializer_list<T> values) {
    UniformRandomDistribution(values.begin(), values.end());
  }

  template <class Iterator> FiniteRandomDistribution(Iterator begin, Iterator end) {
    size_t size = end - begin;
    _probabilities.resize(size);
    _aliases.resize(size);
    double sum = 0;
    for (Iterator current = begin; current != end; current++) {
      sum += *current;
      _probabilities[current - begin] = *current;
    }

    std::vector<size_t> too_small, too_large;
    for (size_t i = 0; i != _probabilities.size(); i++) {
      _probabilities[i] *= size / sum;
      if (_probabilities[i] < 1)
        too_small.push_back(i);
      else if (_probabilities[i] > 1)
        too_large.push_back(i);
      _aliases[i] = i; // default alias to not get issues from rounding errors
    }

    while (!too_large.empty() && !too_small.empty()) {
      auto large = too_large.back();
      too_large.pop_back();
      auto small = too_small.back();
      too_small.pop_back();

      _aliases[small] = large;
      _probabilities[large] -= 1 - _probabilities[small];

      if (_probabilities[large] < 1)
        too_small.push_back(large);
      if (_probabilities[large] > 1)
        too_large.push_back(large);

    }
  }

  size_t operator()() {
    size_t i = Random::instance().random_index(0, _probabilities.size());
    return Random::instance().random_bool(_probabilities[i]) ? i : _aliases[i];
  }

private:
  std::vector<double> _probabilities;
  std::vector<size_t> _aliases;
};
} // namespace kaminpar::shm::sparsification