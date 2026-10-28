// Glue between the run loop and the product modules (host only, no CUDA types, so the CPU CLI stubs compile it too).

#pragma once

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "../core/canary_version.h"
#include "../core/run.h"
#include "config.h"
#include "device_policy.h"
#include "fee.h"
#include "json_dom.h"
#include "product_cli.h"
#include "shutdown.h"
#include "stats.h"
#include "stats_server.h"
#include "status_view.h"

namespace om {

inline uint64_t steadyNowMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline bool readTextFile(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[4096];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0 && out->size() <= (2u << 20)) out->append(buf, n);
    fclose(f);
    return true;
}

/** Write `text` to `path` through a temp file: flush, fsync, rename over the target. */
inline bool writeFileAtomic(const std::string &path, const std::string &text, std::string *error) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) { if (error) *error = "cannot write " + tmp; return false; }
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size() && fflush(f) == 0;
#ifdef _WIN32
    ok = ok && _commit(_fileno(f)) == 0;
#else
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = (fclose(f) == 0) && ok;
    if (!ok) { std::filesystem::remove(tmp, ec); if (error) *error = "cannot flush " + tmp; return false; }
    std::filesystem::rename(tmp, path, ec);
    if (ec) { if (error) *error = "cannot replace " + path + ": " + ec.message(); return false; }
    return true;
}

/** Load a config file, or defaults when `path` is empty. Errors are one clear line. */
inline bool loadProductConfig(const std::string &path, ProductConfig *cfg, std::string *error) {
    if (path.empty()) { *cfg = ProductConfig{}; return true; }
    std::string text;
    if (!readTextFile(path, &text)) { if (error) *error = "cannot read config file '" + path + "'"; return false; }
    std::string perr;
    if (!parseConfig(text, cfg, &perr)) { if (error) *error = "config file '" + path + "': " + perr; return false; }
    return true;
}

inline std::string buildArchLabel() {
#if defined(CANARY_PEARL_ENABLE_SM120_NATIVE_EVIDENCE)
    return "SM120";
#elif defined(CANARY_PEARL_AMPERE_BUILD)
    return "SM86";
#else
    return "host-only";
#endif
}

inline int compiledCc() {
#if defined(CANARY_PEARL_ENABLE_SM120_NATIVE_EVIDENCE)
    return 120;
#elif defined(CANARY_PEARL_AMPERE_BUILD)
    return 86;
#else
    return 0;
#endif
}

/** The qualified fixed selection of the live miner for a build architecture (shape and tile). Without it the native falls into its
 *  kernel/tile auto-tuner, which never settles on a shared desktop GPU. Returns false for a host-only build (no kernels). */
inline bool fixedSelectionFor(int cc, std::string *shape, std::string *tile) {
    if (cc == 120) { *shape = "16384x131072"; *tile = "n64 k64 4x4/2x4"; return true; }
    // RTX 3080: the SM86 kernel (noisy_gemm_ampere.cuh) measured 107 against 81 T MAC/s for ptx k64 4x4/4x2.
    // n=65536 halves the per-attempt A preparation's share. The 3080's 5 MB L2 never held B at n=32768
    // either, so unlike a 4090 the wider shape loses nothing in the GEMM: 93.9 against 88.6 T MAC/s end to end.
    if (cc == 86) { *shape = "4096x65536"; *tile = "amp k128 2x4/4x4"; return true; }
    return false;
}

/** CANARY_PEARL_PRODUCT_FIXED_SELECTION admits only the SM120 K8192 shape/tile; other builds pin through SHAPE/TILE alone. */
inline bool productFixedSelectionFlagFor(int cc) noexcept { return cc == 120; }

inline const char *productTmaDataflowFor(int cc) noexcept {
    return cc == 120 ? "combined-v1" : nullptr;
}

/** Pin the fixed selection through the native's own CANARY_PEARL_SHAPE / CANARY_PEARL_TILE controls unless the operator already
 *  set them. Must run before the algorithm object is created (it reads them at construction). */
inline void applyProductFixedSelection() {
    std::string shape, tile;
    if (!fixedSelectionFor(compiledCc(), &shape, &tile)) return;
    const std::pair<const char *, const std::string *> vars[] = {{"CANARY_PEARL_SHAPE", &shape}, {"CANARY_PEARL_TILE", &tile}};
    for (const auto &v : vars) {
#ifdef _WIN32
        _putenv_s(v.first, v.second->c_str());
#else
        setenv(v.first, v.second->c_str(), 0);
#endif
    }
    if (const char *dataflow = productTmaDataflowFor(compiledCc())) {
#ifdef _WIN32
        _putenv_s("CANARY_PEARL_TMA_DATAFLOW", dataflow);
#else
        setenv("CANARY_PEARL_TMA_DATAFLOW", dataflow, 0);
#endif
    }
    if (!productFixedSelectionFlagFor(compiledCc())) return;
#ifdef _WIN32
    _putenv_s("CANARY_PEARL_PRODUCT_FIXED_SELECTION", "1");
#else
    setenv("CANARY_PEARL_PRODUCT_FIXED_SELECTION", "1", 1);
#endif
}

inline std::string versionText() {
    // H-031 Amendment 2: the donation destination is never presented (no address, account or digest).
    return std::string("CanaryMiner ") + CANARY_MINER_VERSION + "  build=" + buildArchLabel() + "  donation-policy=" + feePolicyBuildLabel() +
           "  donation=" + (feeDestinationConfigured() ? "compiled-in" : "not compiled in");
}

/** `status`: GET the local stats endpoint of a running miner. */
inline int commandStatus(const ProductConfig &cfg, bool rawJson) {
    if (!cfg.api.enabled) {
        fprintf(stderr, "status: the stats endpoint is disabled in this configuration\n");
        return kExitStatusDown;
    }
    int code = 0;
    std::string body, err;
    const std::string host = cfg.api.bind == "localhost" ? std::string("127.0.0.1") : cfg.api.bind;
    if (!httpGetLocal(host, cfg.api.port, "/api/v1/stats", &code, &body, &err)) {
        fprintf(stderr, "status: %s\n", err.c_str());
        return kExitStatusDown;
    }
    if (code != 200) { fprintf(stderr, "status: the endpoint answered HTTP %d\n", code); return kExitStatusDown; }
    if (rawJson) { printf("%s\n", body.c_str()); return kExitOk; }
    JsonValue root;
    std::string text;
    if (!parseJsonStrict(body, &root, &err) || !renderStatusText(root, &text, &err)) {
        fprintf(stderr, "status: %s\n", err.c_str());
        return kExitStatusDown;
    }
    printf("%s", text.c_str());
    return kExitOk;
}

/** Feeds the stats snapshot from the run loop's report callback. One miner process drives one device. */
class StatsObserver final : public RunObserver {
   public:
    StatsObserver(StatsPublisher *pub, const StatsDevice &base, const std::string &poolEndpoint, const std::string &poolPreset)
        : pub_(pub), base_(base), poolEndpoint_(poolEndpoint), poolPreset_(poolPreset) {}

    void onReport(const MinerStats &s, const GpuTelemetry &g) override {
        StatsDevice d = base_;
        const double toT = s.macsPerUnit > 0 ? s.macsPerUnit * 1e6 / 1e12 : 0.0;  // hashrate is in millions of work units per second
        d.rateTmacs = s.hashrate * toT;
        d.rateAvgTmacs = s.hashrateAvg * toT;
        d.attemptsPerSecond = s.attemptsPerSecond;
        d.powerWatts = -1;
        d.temperatureC = -1;
        if (g.valid) { d.powerWatts = g.powerMilliwatts / 1000.0; d.temperatureC = static_cast<int>(g.temperatureC); }
        last_ = statsOutcomesFromRun(s);
        d.outcomes = last_;
        carryInto(&d.outcomes, carry_);
        const double up = s.uptimeSeconds + carryUptime_;
        lastUptime_ = up;
        pub_->update([&](StatsSnapshot *snap) {
            snap->devices.assign(1, d);
            snap->uptimeSeconds = up;
            snap->poolEndpoint = poolEndpoint_;
            snap->poolPreset = poolPreset_;
            snap->poolState = "mining";
            if (snap->state == "starting") snap->state = "running";
        });
    }

    /** The user run ended for a fee window: fold its counters into the carried totals so the next user run adds to them.
     *  User counters therefore accumulate across user runs only; fee shares never reach this class. */
    void foldRun() {
        carryInto(&carry_, last_);
        carry_.pending = 0;
        carryUptime_ = lastUptime_;
        last_ = StatsOutcomes{};
    }

    static StatsOutcomes statsOutcomesFromRun(const MinerStats &s) {
        StatsOutcomes o;
        o.submitted = s.submitted;
        o.acknowledged = s.acknowledged;
        o.accepted = s.accepted;
        o.rejected = s.poolRejected;
        o.stale = s.stale;
        o.unknown = s.unknown;
        o.pending = s.pending;
        o.hostInvalid = s.hostInvalid;
        o.submitFailed = s.submitFailed;
        o.localStale = s.localStale;
        return o;
    }

    /** dst += src for every counter except pending (a live gauge), saturating. */
    static void carryInto(StatsOutcomes *dst, const StatsOutcomes &src) {
        auto add = [](uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; };
        dst->submitted = add(dst->submitted, src.submitted);
        dst->acknowledged = add(dst->acknowledged, src.acknowledged);
        dst->accepted = add(dst->accepted, src.accepted);
        dst->rejected = add(dst->rejected, src.rejected);
        dst->stale = add(dst->stale, src.stale);
        dst->unknown = add(dst->unknown, src.unknown);
        dst->guardRetired = add(dst->guardRetired, src.guardRetired);
        dst->hostInvalid = add(dst->hostInvalid, src.hostInvalid);
        dst->submitFailed = add(dst->submitFailed, src.submitFailed);
        dst->localStale = add(dst->localStale, src.localStale);
    }

   private:
    StatsPublisher *pub_;
    StatsDevice base_;
    std::string poolEndpoint_, poolPreset_;
    StatsOutcomes last_, carry_;
    double carryUptime_ = 0, lastUptime_ = 0;
};

/** Receives the FEE run's report intervals. Fee shares are kept apart from the device and total outcomes: they only ever reach
 *  stats.fee.outcomes. Counters accumulate across windows. */
class FeeStatsObserver final : public RunObserver {
   public:
    explicit FeeStatsObserver(StatsPublisher *pub) : pub_(pub) {}

    void onReport(const MinerStats &s, const GpuTelemetry &) override {
        last_ = StatsObserver::statsOutcomesFromRun(s);
        publish();
    }
    /** A window opened (true) or closed (false): fold the finished window's counters into the carried fee totals. */
    void boundary(bool opening) {
        StatsObserver::carryInto(&carry_, last_);
        carry_.pending = 0;
        if (!opening) {
            // Per-window counts for the operator and the gate; the totals keep accumulating in stats.donation.outcomes.
            fprintf(stderr, "donation window closed: %llu accepted, %llu rejected, %llu host_invalid, %llu submit_failed (totals: %llu accepted, %llu rejected)\n",
                    static_cast<unsigned long long>(last_.accepted), static_cast<unsigned long long>(last_.rejected),
                    static_cast<unsigned long long>(last_.hostInvalid), static_cast<unsigned long long>(last_.submitFailed),
                    static_cast<unsigned long long>(carry_.accepted), static_cast<unsigned long long>(carry_.rejected));
        }
        last_ = StatsOutcomes{};
        publish();
    }

   private:
    StatsPublisher *pub_;
    StatsOutcomes last_, carry_;
    void publish() {
        StatsOutcomes o = carry_;
        StatsObserver::carryInto(&o, last_);
        o.pending = last_.pending;
        pub_->update([&](StatsSnapshot *snap) { snap->fee.outcomes = o; });
    }
};

/**
 * Runs the shutdown machine on its own thread so the (blocking) run loop on the main thread cannot starve it.
 *   - `stopFlag` is the signal handler's flag; when it becomes non-zero the machine starts DRAINING;
 *   - the main thread calls runLoopFinished() when runMiner returns, which lets DRAINING complete;
 *   - if the drain deadline passes first the machine records the in-flight shares as unknown, persists, closes, and the
 *     driver terminates the process with kExitShutdownForced (the GPU is released by the OS), never replaying anything.
 */
class ShutdownDriver {
   public:
    ShutdownDriver(uint64_t drainMs, ShutdownHooks hooks, volatile sig_atomic_t *stopFlag, std::atomic<bool> *runLoopDone,
                   std::function<void(const char *)> stateSink, volatile sig_atomic_t *closeFlag = nullptr)
        : stopFlag_(stopFlag), closeFlag_(closeFlag), runLoopDone_(runLoopDone), stateSink_(std::move(stateSink)) {
        std::atomic<bool> *done = runLoopDone;
        hooks.drained = [done] { return done->load(); };
        machine_.reset(new ShutdownMachine(drainMs, std::move(hooks)));
    }
    ~ShutdownDriver() { join(); }

    void start() { thread_ = std::thread([this] { loop(); }); }
    void runLoopFinished() { runLoopDone_->store(true); }
    void requestStop(const std::string &why) { machine_->requestStop(why); }
    void join() { if (thread_.joinable()) thread_.join(); }
    ShutdownMachine &machine() { return *machine_; }

   private:
    volatile sig_atomic_t *stopFlag_;
    volatile sig_atomic_t *closeFlag_;   // console close/logoff/shutdown: skip the drain
    std::atomic<bool> *runLoopDone_;
    std::function<void(const char *)> stateSink_;
    std::unique_ptr<ShutdownMachine> machine_;
    std::thread thread_;
    ShutdownState lastShown_ = ShutdownState::Running;

    static const char *publicName(ShutdownState s) {
        switch (s) {
            case ShutdownState::Running: return "running";
            case ShutdownState::Draining: return "draining";
            case ShutdownState::Done: return "stopped";
            default: return "stopping";
        }
    }

    void loop() {
        while (!machine_->done()) {
            if (closeFlag_ && *closeFlag_ && !machine_->stopRequested()) {
                machine_->setDrainMs(0);
                machine_->requestStop("console-close");
            } else if (stopFlag_ && *stopFlag_) {
                machine_->requestStop("signal");
            }
            const ShutdownState s = machine_->step(steadyNowMs());
            if (s != lastShown_) { lastShown_ = s; if (stateSink_) stateSink_(publicName(s)); }
            if (machine_->done()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (machine_->forced() && !runLoopDone_->load()) {
            fflush(stdout);
            fflush(stderr);
            std::_Exit(kExitShutdownForced);
        }
    }
};

/** The record PERSIST leaves behind: why the run ended and what happened to in-flight shares. */
inline std::string shutdownRecordJson(bool forced, uint64_t unknownRecorded, const std::string &reason, const StatsSnapshot &snap) {
    return "{\"schema\":\"canaryminer-shutdown-v1\",\"reason\":" + jsonQuote(reason) + ",\"forced\":" + (forced ? "true" : "false") +
           ",\"in_flight_recorded_unknown\":" + std::to_string(unknownRecorded) + ",\"replayed\":0,\"final_outcomes\":" +
           outcomesJson(statsTotals(snap)) + "}\n";
}

}  // namespace om
