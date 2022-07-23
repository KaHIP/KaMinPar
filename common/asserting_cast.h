namespace kaminpar {
template <typename To, typename From>
To asserting_cast(const From value) {
    // @todo correct casts
    // KASSERT(value <= std::numeric_limits<To>::max());
    // KASSERT(value >= std::numeric_limits<To>::lowest());
    return static_cast<To>(value);
}
} // namespace kaminpar

