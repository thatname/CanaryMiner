// Fee wiring (H-022): the outer-loop lane switch, driven with an injected clock and a fake runMiner.
// Window at the offset, 99:1 over cycles, loop-top-only switching, fee failure -> debt, stop mid-window, corrupt state, no destination,
// wallet separation, D3 budget untouched, idle time, stats separation.

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../src/product/fee_session.h"
#include "../src/product/product_main.h"
#include "product_test_util.h"

using namespace om;

namespace {

constexpr uint64_t kSec = 1000;
constexpr uint64_t kMin = 60 * kSec;
const char *kUserWallet = "prl1USERWALLETSECRET";
const char *kUserWorker = "USERWORKER";
const char *kUserAccount = "USERACCOUNT";
const char *kDest = "prl1FEEDESTINATION";

class FixedRandom final : public FeeRandom {
   public:
    explicit FixedRandom(uint64_t v) : v_(v) {}
    uint64_t below(uint64_t n) override { return std::min(v_, n ? n - 1 : 0); }

   private:
    uint64_t v_;
};

enum class FeeBehaviour { Works, ConnectFails, NoWork, StopAfter20s };
enum class UserBehaviour { Mines, Idle, GuardStopEvery };

struct RunRecord {
    std::string wallet, worker, account;
    std::string poolHost;
    int poolPort = 0;
    bool proofCompression = false;
    PearlSourcePolicy policy;
    bool fee = false;
    uint64_t startMs = 0, endMs = 0;
    int exitCode = 0;
    uint64_t productiveIterations = 0;
    uint64_t iterationsAfterExitCode = 0;
};

// Simulated world: one tick = one loop iteration = one second.
struct Sim {
    uint64_t now = 0;
    bool stop = false;
    uint64_t sleptMs = 0;
    uint64_t stopAtMs = ~0ull;  // the operator stops the miner at this time
    FeeBehaviour feeBehaviour = FeeBehaviour::Works;
    UserBehaviour userBehaviour = UserBehaviour::Mines;
    uint64_t guardStopsLeft = 0;
    std::vector<RunRecord> runs;
    std::vector<std::string> log;
    std::vector<FeeStatus> published;
    int hookCallsAfterStopCode = 0;

    int run(const RunOptions &o) {
        RunRecord r;
        r.wallet = o.wallet;
        r.worker = o.worker;
        r.account = o.poolAccount;
        r.poolHost = o.poolHost;
        r.poolPort = o.poolPort;
        r.proofCompression = o.poolProofCompression;
        r.policy = o.pearlPolicy;
        r.fee = o.worker == kFeeWorker;
        r.startMs = now;
        int code = 0;
        bool searchDone = false;  // like runMiner: the first loop top sees no completed search
        if (r.fee && feeBehaviour == FeeBehaviour::ConnectFails) { r.exitCode = 1; r.endMs = now; runs.push_back(r); return 1; }
        uint64_t feeIterations = 0;
        for (;;) {
            if (stop || now >= stopAtMs) { stop = true; code = 0; break; }
            if (o.loopHook) {
                const int c = o.loopHook->atLoopTop(searchDone);
                if (c) { code = c; break; }
            }
            if (r.fee && feeBehaviour == FeeBehaviour::NoWork) { now += kSec; searchDone = false; continue; }
            if (!r.fee && userBehaviour == UserBehaviour::Idle) { now += kSec; searchDone = false; continue; }
            if (!r.fee && userBehaviour == UserBehaviour::GuardStopEvery && guardStopsLeft) {
                --guardStopsLeft;
                code = kRunGuardStop;
                break;
            }
            now += kSec;  // one search
            searchDone = true;
            ++r.productiveIterations;
            ++feeIterations;
            if (r.fee && feeBehaviour == FeeBehaviour::StopAfter20s && feeIterations == 20) { stop = true; }
        }
        r.exitCode = code;
        r.endMs = now;
        runs.push_back(r);
        return code;
    }

    FeeLoopDeps deps() {
        FeeLoopDeps d;
        d.nowMs = [this] { return now; };
        d.run = [this](const RunOptions &o) { return run(o); };
        d.stopped = [this] { return stop; };
        d.sleepMs = [this](uint64_t ms) { sleptMs += ms; now += ms; };
        d.log = [this](const std::string &l) { log.push_back(l); };
        d.publish = [this](const FeeStatus &s) { published.push_back(s); };
        return d;
    }
};

RunOptions userOptions() {
    RunOptions o;
    o.poolHost = "pool.example";
    o.poolPort = 7048;
    o.wallet = kUserWallet;
    o.worker = kUserWorker;
    o.poolAccount = kUserAccount;
    o.poolProofCompression = true;
    return o;
}

bool logHas(const Sim &s, const std::string &needle) {
    for (const auto &l : s.log)
        if (l.find(needle) != std::string::npos) return true;
    return false;
}

uint64_t feeRunMs(const Sim &s) {
    uint64_t t = 0;
    for (const auto &r : s.runs)
        if (r.fee) t += r.endMs - r.startMs;
    return t;
}

int feeRuns(const Sim &s) {
    int n = 0;
    for (const auto &r : s.runs) n += r.fee;
    return n;
}

struct Rig {
    Sim sim;
    std::string dir;
    FixedRandom rng;
    GuardRestartBudget budget;
    std::unique_ptr<FeeSession> session;
    FeeLoopDeps deps;
    explicit Rig(uint64_t offsetMs, const char *tag, const std::string &destination = kDest) : dir(makeScratchDir(tag)), rng(offsetMs) {
        deps = sim.deps();
        session.reset(new FeeSession(productFeePolicy(), &rng, FeeStateFile(dir), destination, deps));
        session->load();
    }
    int go() { return runOuterLoop(userOptions(), session.get(), deps, &budget, 0, 1); }
    ~Rig() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};

}  // namespace

int main() {
    // Production schedule is the default; the accelerated schedule is opt-in at build time only.
#ifndef CANARY_FEE_TEST_POLICY
    CHECK(std::string(feePolicyBuildLabel()) == "release");
    CHECK(productFeePolicy().cycleMs == 100 * kMin && productFeePolicy().windowMs == kMin &&
          productFeePolicy().retryBackoffMs == 10 * kMin);
#endif
    CHECK(feePoolEndpoint() == "pearl.f2pool.com:5500");
    CHECK(feePoolLogin() == "bucnazty90.canaryfee");

    // ---- window opens at the offset; the fee run uses the fee identity, the user run the user identity, never crossed
    {
        Rig rig(30 * kMin, "fee-offset");
        rig.sim.stopAtMs = 31 * kMin + 90 * kSec;  // stop shortly after the first window
        const int rc = rig.go();
        CHECK(rc == 0);
        CHECK(rig.sim.runs.size() == 3);  // user, fee, user
        if (rig.sim.runs.size() == 3) {
            const RunRecord &u1 = rig.sim.runs[0], &f = rig.sim.runs[1], &u2 = rig.sim.runs[2];
            CHECK(!u1.fee && f.fee && !u2.fee);
            CHECK(u1.exitCode == kRunFeeWindow && f.exitCode == kRunFeeWindowDone);
            CHECK(u1.endMs >= 30 * kMin && u1.endMs <= 30 * kMin + 3 * kSec);  // at the offset (to the iteration)
            CHECK(f.endMs - f.startMs >= kMin && f.endMs - f.startMs <= kMin + 2 * kSec);
            for (const RunRecord &r : rig.sim.runs) {
                if (r.fee) CHECK(r.wallet == kDest && r.worker == "canaryfee" && r.account == "bucnazty90" &&
                                 r.poolHost == "pearl.f2pool.com" && r.poolPort == 5500 && r.proofCompression &&
                                 r.policy.network == PearlNetwork::Mainnet &&
                                 r.policy.targetSemantics == PearlTargetSemantics::BaseBeforePenalty);
                else CHECK(r.wallet == kUserWallet && r.worker == kUserWorker && r.account == kUserAccount &&
                           r.poolHost == "pool.example" && r.poolPort == 7048 && r.proofCompression);
            }
            CHECK(u2.startMs == f.endMs);  // the user run resumes right after the window
        }
        // no D3 involvement
        CHECK(rig.sim.sleptMs == 0 && rig.budget.usedInWindow(rig.sim.now) == 0);
        // one log line per window open and close; the fee address may appear, the user wallet and worker never
        int opens = 0, closes = 0;
        for (const auto &l : rig.sim.log) {
            opens += l.find("donation window: opening") != std::string::npos;
            closes += l.find("donation window: closed") != std::string::npos;
            CHECK(l.find(kUserWallet) == std::string::npos && l.find(kUserWorker) == std::string::npos && l.find(kUserAccount) == std::string::npos);
        }
        CHECK(opens == 1 && closes == 1);
        CHECK(logHas(rig.sim, kDest) && logHas(rig.sim, "result ok"));
        CHECK(rig.session->windowsCompleted() == 1 && rig.session->poolFailures() == 0);
    }

    // The donation lane uses the fixed F2Pool endpoint and does not mutate the user's pool spec.
    {
        Rig rig(30 * kMin, "fee-fixed-pool");
        RunOptions user = userOptions();
        rig.sim.stopAtMs = 31 * kMin + 90 * kSec;
        CHECK(runOuterLoop(user, rig.session.get(), rig.deps, &rig.budget, 0, 1) == 0);
        CHECK(user.poolHost == "pool.example" && user.poolPort == 7048 && user.poolAccount == kUserAccount && user.poolProofCompression);
        CHECK(rig.sim.runs.size() == 3 && rig.sim.runs[1].fee);
        if (rig.sim.runs.size() == 3) {
            CHECK(rig.sim.runs[1].account == "bucnazty90");
            CHECK(rig.sim.runs[0].account == kUserAccount && rig.sim.runs[2].account == kUserAccount);
        }
    }

    // ---- 99:1 over three cycles: fee time is 1% of the productive time (setup is not productive, it is not in these runs)
    {
        Rig rig(10 * kMin, "fee-ratio");
        rig.sim.stopAtMs = 3 * 100 * kMin + kMin;
        CHECK(rig.go() == 0);
        const double fee = static_cast<double>(feeRunMs(rig.sim));
        const double total = static_cast<double>(rig.sim.now);
        CHECK(feeRuns(rig.sim) >= 3);
        CHECK(fee / total > 0.0095 && fee / total < 0.0108);
        CHECK(rig.session->windowsCompleted() >= 3);
    }

    // ---- the run ends only at a loop top: a window request leaves the user run with no further iteration
    {
        Rig rig(5 * kMin, "fee-looptop");
        rig.sim.stopAtMs = 7 * kMin + 30 * kSec;
        CHECK(rig.go() == 0);
        CHECK(!rig.sim.runs.empty() && rig.sim.runs[0].exitCode == kRunFeeWindow);
        // the sim runs one search per iteration and consults the hook only before each one (runMiner's contract)
        CHECK(rig.sim.runs[0].productiveIterations * kSec == rig.sim.runs[0].endMs - rig.sim.runs[0].startMs);
    }

    // ---- fee pool failure (connect/auth): straight back to the user, debt <= one window, D3 untouched, mining never stops
    {
        Rig rig(10 * kMin, "fee-fail");
        rig.sim.feeBehaviour = FeeBehaviour::ConnectFails;
        rig.sim.stopAtMs = 100 * kMin + 25 * kMin;  // across the cycle boundary
        CHECK(rig.go() == 0);
        CHECK(rig.session->poolFailures() >= 1);
        CHECK(logHas(rig.sim, "result failed"));
        CHECK(rig.session->scheduler().state().debtMs <= 60 * kSec);
        CHECK(rig.session->scheduler().state().owedMs <= 120 * kSec);
        CHECK(rig.sim.sleptMs == 0 && rig.budget.usedInWindow(rig.sim.now) == 0);
        // the user kept mining (nothing but fee failures interrupted it), and every window attempt used the fee identity
        uint64_t userIt = 0;
        for (const auto &r : rig.sim.runs) userIt += r.fee ? 0 : r.productiveIterations;
        CHECK(userIt * kSec >= 100 * kMin);
        CHECK(!logHas(rig.sim, "guard stop"));
        // retry is backed off: consecutive failed windows are at least the retry backoff apart
        uint64_t prevEnd = 0;
        bool backoffOk = true;
        for (const auto &r : rig.sim.runs) {
            if (!r.fee) continue;
            if (prevEnd && r.startMs < prevEnd + productFeePolicy().retryBackoffMs) backoffOk = false;
            prevEnd = r.endMs;
        }
        CHECK(backoffOk);
    }

    // ---- fee pool never provides work: the window is abandoned after 60 s and counted as a failure
    {
        Rig rig(10 * kMin, "fee-nowork");
        rig.sim.feeBehaviour = FeeBehaviour::NoWork;
        rig.sim.stopAtMs = 14 * kMin;
        CHECK(rig.go() == 0);
        CHECK(rig.session->poolFailures() == 1 && rig.session->windowsCompleted() == 0);
        CHECK(logHas(rig.sim, "did not provide work"));
        for (const auto &r : rig.sim.runs)
            if (r.fee) CHECK(r.endMs - r.startMs >= 60 * kSec && r.endMs - r.startMs <= 62 * kSec);
        CHECK(rig.session->scheduler().state().feeMs == 0);  // nothing credited for idle time
    }

    // ---- stop mid-window: clean exit, state persisted, the unmet part honored by the next start
    {
        Rig rig(10 * kMin, "fee-stop");
        rig.sim.feeBehaviour = FeeBehaviour::StopAfter20s;
        const int rc = rig.go();
        CHECK(rc == 0);
        CHECK(logHas(rig.sim, "interrupted by stop"));
        CHECK(rig.session->poolFailures() == 0);
        FeeState onDisk;
        CHECK(FeeStateFile(rig.dir).load(&onDisk) == FeeStateFile::LoadResult::Loaded);
        CHECK(onDisk.feeMs >= 19 * kSec && onDisk.feeMs <= 21 * kSec && onDisk.owedMs == kMin);
        // next start: same state directory
        Sim sim2;
        FixedRandom rng2(0);
        GuardRestartBudget budget2;
        FeeLoopDeps deps2 = sim2.deps();
        FeeSession next(productFeePolicy(), &rng2, FeeStateFile(rig.dir), kDest, deps2);
        next.load();
        CHECK(next.scheduler().state().feeMs == onDisk.feeMs && next.scheduler().state().userMs == onDisk.userMs);
        sim2.stopAtMs = 3 * kMin;
        CHECK(runOuterLoop(userOptions(), &next, deps2, &budget2, 0, 1) == 0);
        const uint64_t remainingMs = kMin - onDisk.feeMs;
        CHECK(feeRuns(sim2) == 1);
        CHECK(feeRunMs(sim2) >= remainingMs && feeRunMs(sim2) <= remainingMs + 2 * kSec);
        CHECK(next.scheduler().state().feeMs == kMin);
    }

    // ---- shutdown flush: a forced stop (console close, drain timeout) writes the donation state even with no change since the
    //      last save; a disabled fee writes nothing
    {
        const std::string dir = makeScratchDir("fee-flush");
        Sim sim;
        FixedRandom rng(0);
        FeeLoopDeps deps = sim.deps();
        FeeSession s(productFeePolicy(), &rng, FeeStateFile(dir), kDest, deps);
        s.load();
        std::filesystem::remove(FeeStateFile(dir).path());  // no file yet; the flush must create it
        CHECK(!std::filesystem::exists(FeeStateFile(dir).path()));
        s.flushForShutdown();
        FeeState onDisk;
        CHECK(std::filesystem::exists(FeeStateFile(dir).path()));
        CHECK(FeeStateFile(dir).load(&onDisk) == FeeStateFile::LoadResult::Loaded && onDisk == s.scheduler().state());

        const std::string off = makeScratchDir("fee-flush-off");
        FeeSession disabled(productFeePolicy(), &rng, FeeStateFile(off), std::string(), deps);
        disabled.flushForShutdown();
        CHECK(!std::filesystem::exists(FeeStateFile(off).path()));
    }

    // ---- corrupt state file: warning, fresh state, mining and the fee both go on
    {
        const std::string dir = makeScratchDir("fee-corrupt");
        { std::ofstream f(FeeStateFile(dir).path(), std::ios::binary); f << "{not json at all"; }
        Sim sim;
        FixedRandom rng(5 * kMin);
        GuardRestartBudget budget;
        FeeLoopDeps deps = sim.deps();
        FeeSession s(productFeePolicy(), &rng, FeeStateFile(dir), kDest, deps);
        s.load();
        CHECK(logHas(sim, "damaged"));
        CHECK(s.scheduler().state().initialized && s.scheduler().state().cycle == 0 && s.scheduler().state().feeMs == 0);
        sim.stopAtMs = 8 * kMin;
        CHECK(runOuterLoop(userOptions(), &s, deps, &budget, 0, 1) == 0);
        CHECK(feeRuns(sim) == 1);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    // ---- no destination compiled in: inert, no hook, no fee run, plain D3 loop
    {
        Rig rig(1 * kMin, "fee-inert", "");
        CHECK(!rig.session->enabled());
        rig.sim.stopAtMs = 10 * kMin;
        CHECK(rig.go() == 0);
        CHECK(rig.sim.runs.size() == 1 && !rig.sim.runs[0].fee);
        // null session behaves the same
        Sim sim;
        GuardRestartBudget budget;
        FeeLoopDeps deps = sim.deps();
        sim.stopAtMs = 10 * kMin;
        CHECK(runOuterLoop(userOptions(), nullptr, deps, &budget, 0, 1) == 0);
        CHECK(sim.runs.size() == 1 && !sim.runs[0].fee);
    }

    // ---- D3 budget: three guard stops restart, the fourth ends the session; fee windows in between neither use nor reset it
    {
        Rig rig(1 * kMin, "fee-d3");
        rig.sim.userBehaviour = UserBehaviour::GuardStopEvery;
        rig.sim.guardStopsLeft = 4;
        rig.sim.stopAtMs = 500 * kMin;
        const int rc = rig.go();
        CHECK(rc == 1);  // exitRuntime: budget spent
        CHECK(logHas(rig.sim, "restart budget spent"));
        CHECK(rig.sim.sleptMs == 3 * GuardRestartBudget::kDelayMs);
        CHECK(rig.budget.usedInWindow(rig.sim.now) == 3);
    }
    {
        Rig rig(1 * kMin, "fee-d3b");  // a fee window early in the run, then guard stops: windows are not counted against D3
        rig.sim.stopAtMs = 3 * kMin;
        CHECK(rig.go() == 0);
        CHECK(feeRuns(rig.sim) == 1 && rig.budget.usedInWindow(rig.sim.now) == 0 && rig.sim.sleptMs == 0);
    }

    // ---- idle / no-job time never accrues
    {
        Rig rig(10 * kMin, "fee-idle");
        rig.sim.userBehaviour = UserBehaviour::Idle;
        rig.sim.stopAtMs = 3600 * kSec;
        CHECK(rig.go() == 0);
        CHECK(rig.session->scheduler().state().userMs == 0 && rig.session->scheduler().state().feeMs == 0);
        CHECK(feeRuns(rig.sim) == 0);
    }

    // ---- stats separation: fee shares never reach the device or total outcomes; user counters accumulate across user runs only
    {
        StatsPublisher pub;
        StatsDevice base;
        StatsObserver user(&pub, base, "pool:1", "kryptex");
        FeeStatsObserver feeObs(&pub);
        GpuTelemetry g{};
        MinerStats u1;
        u1.submitted = 5; u1.acknowledged = 5; u1.accepted = 4; u1.stale = 1; u1.uptimeSeconds = 100;
        user.onReport(u1, g);
        user.foldRun();
        feeObs.boundary(true);
        MinerStats f1;
        f1.submitted = 3; f1.acknowledged = 3; f1.accepted = 2; f1.poolRejected = 1;
        feeObs.onReport(f1, g);
        StatsSnapshot snap = pub.snapshot();
        CHECK(snap.fee.outcomes.accepted == 2 && snap.fee.outcomes.rejected == 1 && snap.fee.outcomes.submitted == 3);
        CHECK(statsTotals(snap).accepted == 4 && statsTotals(snap).submitted == 5 && statsTotals(snap).rejected == 0);
        feeObs.boundary(false);
        MinerStats u2;
        u2.submitted = 2; u2.acknowledged = 2; u2.accepted = 2; u2.uptimeSeconds = 50;
        user.onReport(u2, g);
        MinerStats f2;
        f2.submitted = 1; f2.acknowledged = 1; f2.accepted = 1;
        feeObs.onReport(f2, g);
        snap = pub.snapshot();
        CHECK(snap.devices[0].outcomes.accepted == 6 && snap.devices[0].outcomes.submitted == 7 && snap.devices[0].outcomes.stale == 1);
        CHECK(snap.uptimeSeconds >= 149.9 && snap.uptimeSeconds <= 150.1);
        CHECK(snap.fee.outcomes.accepted == 3 && snap.fee.outcomes.submitted == 4 && snap.fee.outcomes.rejected == 1);
        JsonValue root;
        std::string err;
        CHECK(parseJsonStrict(statsToJson(snap), &root, &err));
        const JsonValue *fee = root.find("donation");
        CHECK(root.find("fee") == nullptr && fee && fee->find("donation_seconds") && fee->find("enabled") && fee->find("window_active") && fee->find("pool_failures") && fee->find("windows_completed") && fee->find("outcomes"));
        CHECK(root.find("outcomes")->find("accepted")->text == "6");
    }

    // ---- the published status follows the window
    {
        Rig rig(2 * kMin, "fee-status");
        rig.sim.stopAtMs = 5 * kMin;
        CHECK(rig.go() == 0);
        bool sawActive = false, sawInactiveAfter = false;
        for (const auto &s : rig.sim.published) {
            if (s.windowActive) sawActive = true;
            else if (sawActive) sawInactiveAfter = true;
        }
        CHECK(sawActive && sawInactiveAfter);
    }

    // ---- the donation is ON by default; --no-donation and "donation": false make it inert: no scheduler, no window, no state file
    {
        ProductConfig def;
        CHECK(!def.donation && !donationActive(def));  // default off in this fork
        ProductConfig fileOff;
        std::string err;
        CHECK(parseConfig("{\"schema_version\":1,\"donation\":false}", &fileOff, &err) && !fileOff.donation && !donationActive(fileOff));
        ProductConfig fileOn;
        CHECK(parseConfig("{\"schema_version\":1,\"donation\":true}", &fileOn, &err) && fileOn.donation);
        CHECK(!parseConfig("{\"schema_version\":1,\"donation\":\"no\"}", &fileOn, &err) && err.find("true or false") != std::string::npos);
        // the CLI flag wins over a config that says true
        CliOverrides flag;
        flag.noDonation = true;
        CHECK(applyOverrides(flag, &fileOn, &err) && !fileOn.donation && !donationActive(fileOn));
        // and without the flag a config that says false stays off
        CliOverrides none;
        CHECK(applyOverrides(none, &fileOff, &err) && !fileOff.donation);
        // parsed from argv
        const char *argv[] = {"canary-miner", "--no-donation"};
        CliParse parsed;
        CHECK(parseProductCli(2, argv, &parsed, &err) && parsed.ov.noDonation && parsed.rest.empty());
        const char *argv2[] = {"canary-miner"};
        CHECK(parseProductCli(1, argv2, &parsed, &err) && !parsed.ov.noDonation);
        CHECK(effectiveConfigText(fileOff, "test").find("donation: off") != std::string::npos &&
              effectiveConfigText(def, "test").find("donation: on") != std::string::npos);
        // inert run: no hook, no donation run, no state file written
        const std::string dir = makeScratchDir("fee-optout");
        Sim sim;
        GuardRestartBudget budget;
        FeeLoopDeps deps = sim.deps();
        sim.stopAtMs = 120 * kMin;
        std::unique_ptr<FeeSession> session;
        if (donationActive(fileOff)) session.reset(new FeeSession(productFeePolicy(), nullptr, FeeStateFile(dir), kDest, deps));
        CHECK(!session);
        CHECK(runOuterLoop(userOptions(), session.get(), deps, &budget, 0, 1) == 0);
        CHECK(sim.runs.size() == 1 && !sim.runs[0].fee && sim.runs[0].worker == kUserWorker);
        CHECK(!std::filesystem::exists(FeeStateFile(dir).path()) && !std::filesystem::exists(FeeStateFile(dir).backupPath()));
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    return finishTests("fee_session");
}
