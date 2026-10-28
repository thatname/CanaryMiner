#pragma once

#include <cstdint>
#include <string_view>

namespace om::pearl {

enum class AttemptOverlapMode { Disabled, APrepV1, Invalid, APrep64V1 };
constexpr bool attemptOverlapEnabled(AttemptOverlapMode mode) noexcept {
    return mode == AttemptOverlapMode::APrepV1 || mode == AttemptOverlapMode::APrep64V1;
}
constexpr unsigned attemptOverlapThreads(AttemptOverlapMode mode) noexcept {
    return mode == AttemptOverlapMode::APrep64V1 ? 64 : 256;
}
constexpr const char *attemptOverlapName(AttemptOverlapMode mode) noexcept {
    return mode == AttemptOverlapMode::APrepV1 ? "a-prep-v1" :
           mode == AttemptOverlapMode::APrep64V1 ? "a-prep64-v1" : "invalid";
}
// Match the device reduction's balanced-tree schedule. A ROOT operation must
// receive exactly two CVs, so a reduction stage must never collapse to one.
constexpr unsigned attemptMerkleReductionThreads(uint32_t count, unsigned cap) noexcept {
    if ((cap != 64 && cap != 256) || count <= 2 || (count & (count - 1))) return 0;
    unsigned threads = cap;
    while (threads > 1 && 2u * threads > count / 2) threads >>= 1;
    return threads;
}
inline AttemptOverlapMode parseAttemptOverlap(const char *text) noexcept {
    if (!text) return AttemptOverlapMode::Disabled;
    return std::string_view(text) == "a-prep-v1" ? AttemptOverlapMode::APrepV1 :
           std::string_view(text) == "a-prep64-v1" ? AttemptOverlapMode::APrep64V1 :
                                                   AttemptOverlapMode::Invalid;
}
// Admitted on any pool and in the synthetic bench: the overlap changes only
// which stream A is prepared on, never the work or the proof. Formal timing
// and the per-stage profiler still refuse it, since both assume one stream.
inline bool attemptOverlapAdmission(AttemptOverlapMode mode, bool compiled,
                                    bool formal, bool profiling, bool synthetic,
                                    std::string_view host, int port) noexcept {
    if (mode == AttemptOverlapMode::Disabled) return true;
    return attemptOverlapEnabled(mode) && compiled && !formal && !profiling &&
           (synthetic || (!host.empty() && port > 0 && port <= 65535));
}

// This is the actual CUDA overlap driver's ordering, also executable with a
// host fake. Preparation of lookahead is uncredited until launch promotion;
// it never creates an additional active progress/ledger attempt. Ops must
// drain both streams without throwing, even after an earlier failure.
struct PreparedAttemptLookahead {
    bool ready = false;
    unsigned slot = 0;
    uint64_t nonce = 0;
    void discard() noexcept { ready = false; }
};

// Backend validates complete job/plan identity before this scheduler. A cache
// only becomes reusable after BOTH streams drain successfully. Exactly one
// launch/retirement is credited, even when a successor was prepared.
template <class Ops>
bool runSingleAttemptLookahead(uint64_t nonce, bool hint, uint64_t next,
                              PreparedAttemptLookahead &cache, Ops &ops) noexcept {
    const bool hit = cache.ready && cache.nonce == nonce;
    const unsigned slot = hit ? cache.slot : 0;
    cache.discard();
    bool ok = false;
    const bool prepareNext = hint && next != nonce;
    try {
        ok = (hit || ops.prepare(slot,nonce)) && ops.launch(slot,nonce) &&
             (!prepareNext || ops.prepare(slot ^ 1,next)) && ops.retire(slot,nonce);
    } catch (...) { ok = false; }
    const bool drained = ops.drain();
    if (ok && drained && prepareNext) {
        cache.slot = slot ^ 1; cache.nonce = next; cache.ready = true;
    }
    return ok && drained;
}

template <class Ops>
bool runAttemptOverlapBatch(uint64_t first, uint64_t attempts, Ops &ops) noexcept {
    bool ok = false;
    try {
        ok = [&]() {
            if (!attempts || attempts - 1 > UINT64_MAX - first) return false;
            if (!ops.prepare(0, first)) return false;
            for (uint64_t i = 0; i < attempts; ++i) {
                const unsigned slot = static_cast<unsigned>(i & 1);
                const uint64_t nonce = first + i; // range checked above
                if (!ops.launch(slot, nonce)) return false;
                if (i + 1 < attempts && !ops.prepare(slot ^ 1, nonce + 1))
                    return false;
                // Includes compute completion, scan readback, capture, and
                // proof opening. Only then is this A slot reusable.
                if (!ops.retire(slot, nonce)) return false;
            }
            return true;
        }();
    } catch (...) {
        ok = false;
    }
    const bool drained = ops.drain();
    return ok && drained;
}

} // namespace om::pearl
