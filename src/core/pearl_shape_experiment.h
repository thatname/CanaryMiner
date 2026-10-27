#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace om {
constexpr const char *kPearlShapeExperimentProfile = "shape4096x65536-v1";
constexpr const char *kPearlShapeExperimentSelector = "4096x65536";
constexpr std::uint32_t kPearlShapeExperimentM = 4096;
constexpr std::uint32_t kPearlShapeExperimentN = 65536;
constexpr const char *kPearlExactK8192Profile = "exact16384x131072-k8192-v1";
constexpr const char *kPearlExactK8192Shape = "16384x131072";
constexpr const char *kPearlExactK8192Tile = "n64 k64 4x4/2x4";
inline const char *pearlProductFixedTile() noexcept { return kPearlExactK8192Tile; }
inline bool pearlProductFixedSelectionActive() noexcept {
    const char *value = std::getenv("CANARY_PEARL_PRODUCT_FIXED_SELECTION");
    return value && !std::strcmp(value, "1");
}
inline bool pearlProductFixedSelectionMatches(const char *shape, const char *tile) noexcept {
#if defined(CANARY_PEARL_AMPERE_BUILD)
    // The SM86 build's fixed selection (product_main.h fixedSelectionFor(86)).
    return shape && tile && !std::strcmp(shape, "4096x32768") && !std::strcmp(tile, "ptx k64 4x4/4x2");
#else
    return shape && tile && !std::strcmp(shape, kPearlExactK8192Shape) &&
        !std::strcmp(tile, pearlProductFixedTile());
#endif
}
enum class PearlShapeExperiment { Disabled, Shape4096x65536V1, Exact16384x131072K8192V1, Invalid };
inline bool pearlShapeExperimentEnabled(PearlShapeExperiment value) noexcept {
    return value == PearlShapeExperiment::Shape4096x65536V1 ||
        value == PearlShapeExperiment::Exact16384x131072K8192V1;
}
inline bool pearlShapeExperimentTileAllowed(PearlShapeExperiment value, const char *tile) noexcept {
    return !tile || (value == PearlShapeExperiment::Exact16384x131072K8192V1 &&
        !std::strcmp(tile,kPearlExactK8192Tile));
}

// Parsing alone grants no admission. Missing profile leaves legacy shape
// handling untouched; evidence callers must retain their old override checks.
inline PearlShapeExperiment pearlParseShapeExperiment(const char *profile,
                                                       const char *shape, const char *tile=nullptr) noexcept {
    if (!profile) return PearlShapeExperiment::Disabled;
    if (!std::strcmp(profile,kPearlExactK8192Profile))
        return shape && !std::strcmp(shape,kPearlExactK8192Shape) && tile &&
            !std::strcmp(tile,kPearlExactK8192Tile)
            ? PearlShapeExperiment::Exact16384x131072K8192V1 : PearlShapeExperiment::Invalid;
    return !std::strcmp(profile,kPearlShapeExperimentProfile) && shape &&
           !std::strcmp(shape,kPearlShapeExperimentSelector)
        ? PearlShapeExperiment::Shape4096x65536V1 : PearlShapeExperiment::Invalid;
}
constexpr bool pearlShapeExperimentCompiled() noexcept {
#if defined(CANARY_PEARL_ENABLE_SHAPE_EXPERIMENT) && CANARY_PEARL_ENABLE_SHAPE_EXPERIMENT == 1 && \
    defined(CANARY_PEARL_ONLY_BUILD) && CANARY_PEARL_ONLY_BUILD == 1
    return true;
#else
    return false;
#endif
}

constexpr bool pearlExactK8192Compiled() noexcept {
#if defined(CANARY_PEARL_EXPERIMENT_COMMON_DIM) && CANARY_PEARL_EXPERIMENT_COMMON_DIM == 8192 &&     defined(CANARY_PEARL_EXPERIMENT_LARGE_SHAPES)
    return pearlShapeExperimentCompiled();
#else
    return false;
#endif
}

// These context facts must come from the native caller's existing validated
// mode/stop/owned-guard/override admission, never directly from environment flags.
inline bool pearlShapeExperimentAdmission(PearlShapeExperiment value,
    bool genericNativeEvidence, bool boundedStop, bool loopbackOwnedGuard,
    bool conflictingOverride) noexcept {
    if (value == PearlShapeExperiment::Disabled) return true;
    return pearlShapeExperimentEnabled(value) &&
        (value == PearlShapeExperiment::Exact16384x131072K8192V1 ?
            pearlExactK8192Compiled() : pearlShapeExperimentCompiled()) && genericNativeEvidence && boundedStop &&
        loopbackOwnedGuard && !conflictingOverride;
}

class PearlShapeExperimentBinding final {
 public:
    PearlShapeExperimentBinding(const char *profile,const char *shape,const char *tile=nullptr) noexcept
        : captured_(pearlParseShapeExperiment(profile,shape,tile)),
          failed_(captured_ == PearlShapeExperiment::Invalid) {}
    PearlShapeExperiment value() const noexcept { return captured_; }
    bool enabled() const noexcept {
        return pearlShapeExperimentEnabled(captured_);
    }
    std::uint32_t m() const noexcept { return captured_ == PearlShapeExperiment::Exact16384x131072K8192V1 ? 16384 : kPearlShapeExperimentM; }
    std::uint32_t n() const noexcept { return captured_ == PearlShapeExperiment::Exact16384x131072K8192V1 ? 131072 : kPearlShapeExperimentN; }
    bool tileAllowed(const char *tile) const noexcept { return pearlShapeExperimentTileAllowed(captured_,tile); }
    bool failed() const noexcept { return failed_; }
    // Call at initial admission and again before prepare/search side effects.
    // Any observed change or context failure permanently revokes this instance.
    bool revalidate(const char *profile,const char *shape,
        bool genericNativeEvidence,bool boundedStop,bool loopbackOwnedGuard,
        bool conflictingOverride,const char *tile=nullptr) noexcept {
        if (failed_ || pearlParseShapeExperiment(profile,shape,tile) != captured_ ||
            !pearlShapeExperimentAdmission(captured_,genericNativeEvidence,
                boundedStop,loopbackOwnedGuard,conflictingOverride)) failed_ = true;
        else admitted_ = true;
        return !failed_;
    }
    bool admitProductFixed(const char *shape, const char *tile) noexcept {
        if (failed_ || captured_ != PearlShapeExperiment::Disabled ||
            !pearlProductFixedSelectionMatches(shape, tile)) failed_ = true;
        else admitted_ = true;
        return !failed_;
    }
    // Does not replace strict native catalog/kernel/rank/work validation.
    bool observeShape(std::uint32_t m,std::uint32_t n) noexcept {
        if (failed_ || !admitted_ || (enabled() &&
            (m != this->m() || n != this->n()))) failed_ = true;
        return !failed_;
    }
 private:
    const PearlShapeExperiment captured_;
    bool failed_;
    bool admitted_ = false;
};
} // namespace om
