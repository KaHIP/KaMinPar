#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace kaminpar {
template <typename Key, typename Value, typename Derived> class DynamicMapBase {
public:
  static constexpr std::size_t INVALID_POS_MASK =
      ~(std::numeric_limits<std::size_t>::max() >> 1); // MSB is set
  static constexpr std::size_t INITIAL_CAPACITY = 16;

  explicit DynamicMapBase() : _capacity(32), _size(0), _timestamp(1), _data(nullptr) {
    initialize(INITIAL_CAPACITY);
  }

  DynamicMapBase(const DynamicMapBase &) = delete;
  DynamicMapBase &operator=(const DynamicMapBase &other) = delete;

  DynamicMapBase(DynamicMapBase &&other) = default;
  DynamicMapBase &operator=(DynamicMapBase &&other) = default;

  ~DynamicMapBase() = default;

  std::size_t capacity() const {
    return _capacity;
  }

  std::size_t size() const {
    return _size;
  }

  void initialize(const std::size_t capacity) {
    _size = 0;
    _capacity = align_to_next_power_of_two(capacity);
    _timestamp = 1;
    const size_t alloc_size = static_cast<const Derived *>(this)->size_in_bytes();
    _data = std::make_unique<uint8_t[]>(alloc_size);
    std::memset(_data.get(), 0, alloc_size);
    static_cast<Derived *>(this)->initialize_impl();
  }

  bool contains(const Key key) const {
    const std::size_t pos = find(key);
    return pos < INVALID_POS_MASK;
  }

  Value &operator[](const Key key) {
    std::size_t pos = find(key);
    if (pos < INVALID_POS_MASK) {
      return get_value(pos);
    } else {
      if (_size + 1 > (_capacity * 2) / 5) {
        grow();
        pos = find(key);
      }
      return static_cast<Derived *>(this)->add_element_impl(key, Value(), pos & ~INVALID_POS_MASK);
    }
  }

  const Value &get(const Key key) const {
    return get_value(find(key));
  }

  const Value *get_if_contained(const Key key) const {
    const std::size_t pos = find(key);
    if (pos < INVALID_POS_MASK) {
      return &get_value(pos);
    } else {
      return nullptr;
    }
  }

  const Value *end() const {
    return nullptr;
  }

  void clear() {
    _size = 0;
    ++_timestamp;
  }

private:
  inline std::size_t find(const Key key) const {
    return static_cast<const Derived *>(this)->find_impl(key);
  }

  void grow() {
    const std::size_t old_size = _size;
    const std::size_t old_capacity = _capacity;
    const std::size_t old_timestamp = _timestamp;
    const std::size_t new_capacity = 2UL * _capacity;
    const std::unique_ptr<std::uint8_t[]> old_data = std::move(_data);
    const std::uint8_t *old_data_begin = old_data.get();
    initialize(new_capacity);
    static_cast<Derived *>(this)->rehash_impl(
        old_data_begin, old_size, old_capacity, old_timestamp
    );
  }

  Value &get_value(const size_t pos) const {
    return static_cast<const Derived *>(this)->value_at_pos(pos);
  }

  constexpr std::size_t align_to_next_power_of_two(const std::size_t size) const {
    return std::pow(2.0, std::ceil(std::log2(static_cast<double>(size))));
  }

protected:
  std::size_t _capacity;
  std::size_t _size;
  std::size_t _timestamp;
  std::unique_ptr<std::uint8_t[]> _data;
};

template <typename Key, typename Value>
class DynamicFlatMap final : public DynamicMapBase<Key, Value, DynamicFlatMap<Key, Value>> {
  struct MapElement {
    Key key;
    Value value;
    std::size_t timestamp;
  };

  using Base = DynamicMapBase<Key, Value, DynamicFlatMap<Key, Value>>;
  using Base::INVALID_POS_MASK;
  friend Base;

public:
  explicit DynamicFlatMap() : Base(), _elements(nullptr) {
    initialize_impl();
  }

  DynamicFlatMap(const DynamicFlatMap &) = delete;
  DynamicFlatMap &operator=(const DynamicFlatMap &other) = delete;

  DynamicFlatMap(DynamicFlatMap &&other) = default;
  DynamicFlatMap &operator=(DynamicFlatMap &&other) = default;

  ~DynamicFlatMap() = default;

  void free_internal_data() {
    _size = 0;
    _timestamp = 0;
    _data = nullptr;
    _elements = nullptr;
  }

  std::size_t size_in_bytes() const {
    return _capacity * sizeof(MapElement);
  }

private:
  std::size_t find_impl(const Key key) const {
    std::size_t hash = key & (_capacity - 1);
    while (_elements[hash].timestamp == _timestamp) {
      if (_elements[hash].key == key) {
        return hash;
      }
      hash = (hash + 1) & (_capacity - 1);
    }
    return hash | INVALID_POS_MASK;
  }

  Value &value_at_pos(const std::size_t pos) const {
    return _elements[pos].value;
  }

  Value &add_element_impl(Key key, Value value, const std::size_t pos) {
    _elements[pos] = MapElement{key, value, _timestamp};
    _size++;
    return _elements[pos].value;
  }

  void initialize_impl() {
    _elements = reinterpret_cast<MapElement *>(_data.get());
  }

  void rehash_impl(
      const std::uint8_t *old_data_begin,
      [[maybe_unused]] const std::size_t old_size,
      const std::size_t old_capacity,
      const std::size_t old_timestamp
  ) {
    const MapElement *elements = reinterpret_cast<const MapElement *>(old_data_begin);
    for (size_t i = 0; i < old_capacity; ++i) {
      if (elements[i].timestamp == old_timestamp) {
        const size_t pos = find_impl(elements[i].key) & ~INVALID_POS_MASK;
        add_element_impl(elements[i].key, elements[i].value, pos);
      }
    }
  }

  using Base::_capacity;
  using Base::_data;
  using Base::_size;
  using Base::_timestamp;
  MapElement *_elements;
};
} // namespace kaminpar

