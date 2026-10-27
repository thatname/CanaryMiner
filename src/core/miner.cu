// CanaryMiner - CUDA build entry point (NVIDIA).
//
// The run loop lives in run.cpp. This file is the product shell around it: subcommands (run, list-devices, selftest, status,
// version), the config file and CLI, the device allowlist, the redacting log sink, the stats endpoint and the shutdown state
// machine (src/product/). Research and harness options that predate the product CLI are still parsed below.

#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "run.h"
#include "pearl_build_profile.h"
#include "pearl_pool.h"
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
#include "pearl_guard_control_launch.h"
#endif
#include "pearl_accepted_quota.h"
#include "pearl_sustained_admission.h"
#include "pearl_integrated_pipeline.h"
#include "gpu_identity.h"
#include "../algos/pearl-pow/cuda_preparation_plan.h"
#include "../algos/pearl-pow/attempt_overlap.h"
#include "../algos/pearl-pow/a_root_fusion.h"
#include "../product/log_sink.h"
#include "../product/product_main.h"
#include "../product/selftest.h"
#include "../product/restart_budget.h"
#include "../product/fee_session.h"
#include "../product/supervisor.h"
#include "../product/gpu_tuning.h"
#include <chrono>
#include <thread>

// Defined by the linked CUDA backend, not inferred from this TU's flags.
namespace om { bool pearlCudaExperimentCompiled(const char *) noexcept;
bool pearlCudaShapeExperimentCompiled() noexcept; }

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_closeRequested = 0;  // console close/logoff/shutdown: the drain is skipped
static void onSignal(int) { g_stop = 1; }

#if defined(_WIN32)
static HANDLE g_shutdownDone = nullptr;  // signalled after the shutdown record and state are written
static BOOL WINAPI consoleHandler(DWORD type) {
    if (type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) {
        // Windows ends the process as soon as this handler returns, so block until the shutdown record and state are
        // written. Hard cap 4.5 s, under the system's 5 s grace period; then return regardless.
        g_closeRequested = 1;
        g_stop = 1;
        if (g_shutdownDone) WaitForSingleObject(g_shutdownDone, 4500);
        return TRUE;
    }
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        g_stop = 1;
        return TRUE;
    }
    return FALSE;
}
#endif

namespace {

// Identity of every CUDA device, as the driver reports it (the runtime API is a thin layer over the driver API).
bool queryDevices(std::vector<om::DeviceIdentity> *out) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    for (int d = 0; d < n; d++) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        om::DeviceIdentity id;
        id.index = d;
        id.name = p.name;
        id.uuid = om::nvidiaGpuUuid(reinterpret_cast<const unsigned char *>(p.uuid.bytes), sizeof(p.uuid.bytes));
        char pci[32];
        snprintf(pci, sizeof(pci), "%04x:%02x:%02x.0", static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID),
                 static_cast<unsigned>(p.pciDeviceID));
        id.pciBus = pci;
        id.ccMajor = p.major;
        id.ccMinor = p.minor;
        id.memBytes = p.totalGlobalMem;
        out->push_back(id);
    }
    return !out->empty();
}

void printDeviceTable(const std::vector<om::DeviceIdentity> &present) {
    for (const auto &d : present) {
        const om::DeviceVerdict v = om::checkDevice(d, om::compiledCc());
        printf("  [%d] %-26s %5.1f GB  cc %d.%d  %s  %s\n      uuid %s  pci %s\n", d.index, d.name.c_str(), d.memBytes / 1e9, d.ccMajor,
               d.ccMinor, v.allowed ? "supported" : "NOT SUPPORTED", v.allowed ? v.family.c_str() : v.reason.c_str(), d.uuid.c_str(),
               d.pciBus.c_str());
    }
}

}  // namespace

int main(int argc, char **argv) {
    using namespace om;

    CliParse cli;
    {
        std::string cliError;
        if (!parseProductCli(argc, argv, &cli, &cliError)) {
            fprintf(stderr, "%s%s", cliError.c_str(), "\n");
            return kExitUsage;
        }
    }
    if (cli.command == Command::Version) {
        printf("%s\n", versionText().c_str());
        return kExitOk;
    }
    ProductConfig cfg;
    {
        std::string cfgError;
        if (!loadProductConfig(cli.ov.configPath, &cfg, &cfgError) || !applyOverrides(cli.ov, &cfg, &cfgError) ||
            !validateConfig(cfg, false, &cfgError) || (cli.command == Command::Run && !checkPoolLogin(cfg, &cfgError))) {
            fprintf(stderr, "%s%s", cfgError.c_str(), "\n");
            return kExitConfig;
        }
    }
    GpuTuningPlatform tuningPlatform = currentGpuTuningPlatform();
    if (cli.command == Command::Status) return commandStatus(cfg, cli.ov.json);

    std::string algoName = defaultAlgorithmName;
    RunOptions opt;
    opt.backendLabel = "CUDA";
    bool pearlNetworkSeen = false, pearlSemanticsSeen = false, pearlAuthorizationSeen = false;
    int deviceIndex = 0;  // which CUDA GPU; one process per GPU for a rig
    bool formalSeen = false, formalNonceSeen = false;
    bool formalShapeSeen = false, formalKernelSeen = false;
    bool formalTileSeen = false;
    unsigned batchCount = 0, intervalCount = 0, plainCount = 0;
    unsigned benchHeightCount = 0, benchEpochSecondsCount = 0;
    unsigned mclkCount = 0, memOcCount = 0;
    std::string batchText, intervalText, benchHeightText;
    // H-032: a worker child is one GPU of a supervised run (the supervisor passes these; they are not in --help).
    bool workerChild = false, noDashboard = false;
    std::string workerIdText;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    std::string guardControlText;
    PearlGuardControlLaunch guardControl;
#endif

    // --pool/--wallet/--worker/--device(s)/--config/--api-*/--log-dir were consumed by the product parser above.
    opt.remoteOptionsSpecified = !cli.ov.pools.empty() || cli.ov.walletSet || cli.ov.workerSet;
    for (size_t i = 0; i < cli.rest.size(); i++) {
        const std::string a = cli.rest[i];
        if (a == "--pass") opt.remoteOptionsSpecified = true;
        auto next = [&]() -> std::string {
            return (i + 1 < cli.rest.size()) ? cli.rest[++i] : std::string();
        };
        if (a == "--algo") algoName = next();
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
        else if (a == "--pearl-guard-control") {
#if defined(_WIN32)
            fprintf(stderr, "--pearl-guard-control is not supported on Windows\n");
            return 1;
#endif
            if(!guardControlText.empty()) return 1;
            guardControlText=next();
            if(guardControlText.empty()) return 1;
        }
#endif
        else if (a == "--pearl-authorization-mode") {
            if (pearlAuthorizationSeen || !parsePearlAuthorizationModeOption(next(), &opt.pearlSoloAuthorization)) {
                fprintf(stderr, "--pearl-authorization-mode requires one standard|solo value\n");
                return 1;
            }
            pearlAuthorizationSeen = true;
        }
        else if (a == "--pearl-work-range") {
            if (opt.pearlWorkRange.enabled || !parsePearlWorkRange(next(), &opt.pearlWorkRange)) {
                fprintf(stderr, "--pearl-work-range requires one finite 0xFIRST:0xLAST (16 lowercase hex digits each)\n");
                return 1;
            }
        }
        else if (a == "--pearl-transcript") opt.pearlTranscript = next();
        else if (a == "--pearl-network") {
            if (pearlNetworkSeen || !parsePearlNetworkOption(next(), &opt.pearlPolicy.network)) {
                fprintf(stderr, "--pearl-network requires one mainnet|testnet|testnet2|regtest value\n");
                return 1;
            }
            pearlNetworkSeen = true;
        }
        else if (a == "--pearl-target-semantics") {
            if (pearlSemanticsSeen || !parsePearlTargetSemanticsOption(next(), &opt.pearlPolicy.targetSemantics)) {
                fprintf(stderr, "--pearl-target-semantics requires one base-before-penalty|final-adjusted-bound value\n");
                return 1;
            }
            pearlSemanticsSeen = true;
        }
        else if (a == "--pass") opt.password = next();
        else if (a == "--batch") {
            // The grid launches 256 threads per block and advances the nonce
            // counter by exactly this many, so it must be a multiple of 256
            // and non-zero, or batches overlap, skip, or retest forever.
            batchCount++;
            batchText = next();
            uint64_t b = strtoull(batchText.c_str(), nullptr, 10);
            if (b < 256) b = 256;
            opt.batch = b & ~UINT64_C(255);
        }
        else if (a == "--bench") opt.bench = true;
        else if (a == "--autotune") opt.autotune = true;
        else if (a == "--bench-formal") {
            if (formalSeen) {
                fprintf(stderr, "duplicate --bench-formal option\n");
                return 1;
            }
            formalSeen = true;
            opt.formalBenchmark.enabled = true;
        }
        else if (a == "--bench-start-nonce") {
            if (formalNonceSeen) {
                fprintf(stderr, "duplicate --bench-start-nonce option\n");
                return 1;
            }
            formalNonceSeen = true;
            const std::string value = next();
            if (!parseBenchmarkStartingNonce(
                    value, &opt.formalBenchmark.startingNonce)) {
                fprintf(stderr,
                        "--bench-start-nonce requires 0x followed by exactly "
                        "16 hexadecimal digits\n");
                return 1;
            }
        }
        else if (a == "--pearl-shape-id") {
            if (formalShapeSeen) {
                fprintf(stderr, "duplicate --pearl-shape-id option\n");
                return 1;
            }
            formalShapeSeen = true;
            opt.formalBenchmark.shapeId = next();
        }
        else if (a == "--pearl-kernel-id") {
            if (formalKernelSeen) {
                fprintf(stderr, "duplicate --pearl-kernel-id option\n");
                return 1;
            }
            formalKernelSeen = true;
            opt.formalBenchmark.kernelId = next();
        }
        else if (a == "--pearl-tile-id") {
            if (formalTileSeen) {
                fprintf(stderr, "duplicate --pearl-tile-id option\n");
                return 1;
            }
            formalTileSeen = true;
            opt.formalBenchmark.tileId = next();
        }
        else if (a == "--bench-height") {
            benchHeightCount++;
            benchHeightText = next();
            opt.benchEpoch = strtoull(benchHeightText.c_str(), nullptr, 10);
        }
        else if (a == "--bench-epoch-secs") {
            benchEpochSecondsCount++;
            opt.benchEpochSeconds = atoi(next().c_str());
        }
        else if (a == "--mclk-offset") {
            mclkCount++;
            opt.memOffsetMhz = atoi(next().c_str());
        }
        else if (a == "--mem-oc") {
            memOcCount++;
            opt.memOc = true;
        }
        else if (a == "--plain") {
            plainCount++;
            opt.plain = true;
        }
        else if (a == "--ascii") om::g_asciiOnly = true;
        else if (a == "--no-dashboard") noDashboard = true;
        else if (a == "--worker-child") workerChild = true;
        else if (a == "--worker-id") workerIdText = next();
        else if (a == "--interval") {
            intervalCount++;
            intervalText = next();
            opt.reportSeconds = atoi(intervalText.c_str());
        }
        else if (a == "--list-algos") {
            for (const auto &n : availableAlgorithms()) printf("%s\n", n.c_str());
            return 0;
        } else if (a == "--help" || a == "-h") {
            printf("%s", productHelpText().c_str());
            return 0;
        } else if (a == "--help-all") {
            printf(
                "CanaryMiner " CANARY_MINER_VERSION " (CUDA, NVIDIA Blackwell) - all options\n\n"
                "usage: canary-miner [run] [options]    mine (default)\n"
                "       canary-miner list-devices       GPU identity and whether it is supported\n"
                "       canary-miner selftest           offline kernel + proof self-test per device\n"
                "       canary-miner status [--json]    read the stats endpoint of a running miner\n"
                "       canary-miner version\n\n"
                "  --config FILE     config-v1 JSON file (options below override it)\n"
                "  --algo NAME       pearl-pow only (default pearl-pow)\n"
                "  --list-algos      list compiled-in algorithms and exit\n"
                "  --devices LIST    all | index | GPU-uuid | pci id, comma separated; one\n"
                "                    worker per selected GPU, supervised (alias --device)\n"
                "  --pool HOST:PORT  Pearl pool endpoint; requires --wallet and\n"
                "                    both explicit policy options below (a preset name such as\n"
                "                    kryptex or herominers carries them; --pool is repeatable)\n"
                "  --api-bind ADDR   stats endpoint address, loopback only (default 127.0.0.1)\n"
                "  --api-port N      stats endpoint port (default 8085; 0 disables)\n"
                "  --log-dir DIR     rotating redacted log files\n"
                "  --no-dashboard    plain timestamped lines instead of the live dashboard\n"
                "  --no-donation     keep the optional 1%% developer donation off (off by default;\n"
                "                    opt in with config key donation: true)\n"
                "  --pearl-network mainnet\n"
                "                    other parsed networks are unsupported live\n"
                "  --pearl-target-semantics base-before-penalty|final-adjusted-bound\n"
                "  --pearl-authorization-mode standard|solo (default standard)\n"
                "                    use the semantics verified for the pool\n"
                "  --wallet ADDR     public PRL payout address (alias --user)\n"
                "  --worker NAME     worker name (default canary)\n"
                "  --pearl-work-range 0xFIRST:0xLAST\n"
                "                    finite production Pearl range; 16 hex digits each\n"
                "  --pass P          pool protocol password field (default x);\n"
                "                    not wallet secrets or account credentials\n"
                "  --pearl-transcript FILE  test-only Pearl pool metadata log;\n"
                "                    redacts PRL addresses and never logs wallet/auth\n"
                "  --batch N         logical work-unit budget (default 4194304);\n"
                "                    not a nonce count or an accepted-share count\n"
                "  --interval N      seconds between readouts (default 5)\n"
                "  --plain           one-line JSON output, for logs/systemd\n"
                "  --ascii           plain ASCII frame (no box-drawing glyphs)\n"
                "  --mclk-offset N   optional memory transfer-rate offset in MHz;\n"
                "                    0 leaves it unchanged; may require privileges\n"
                "  --mem-oc          optional device-specific memory tuning\n"
                "                    Tuning changes device state and is unqualified;\n"
                "                    formal/qualification runs reject these controls\n"
                "  --bench           synthetic benchmark, no pool connection;\n"
                "                    does not establish accepted shares or earnings\n"
                "  --autotune        explicitly opt in to the Pearl startup tuner\n"
                "  --bench-formal    reproducible synthetic Pearl CUDA benchmark\n"
                "  --bench-start-nonce 0xNNNNNNNNNNNNNNNN\n"
                "  --pearl-shape-id ID --pearl-kernel-id ID --pearl-tile-id ID\n"
                "                    also requires --plain --batch 4194304\n"
                "                    --interval 5; rejects tuning/OC controls\n"
                "  --bench-height H  fixed synthetic fixture height 1851444 only\n"
                "\nSupported devices are listed by 'list-devices'.\n"
            );
            return 0;
        }
        else {
            // AN UNKNOWN ARGUMENT IS AN ERROR, NOT A SHRUG.
            //
            // This loop used to fall off the end and ignore anything it did
            // not recognise. A typo, a renamed flag, or a probe like
            // `--version` (which this miner does not have) therefore did not
            // fail - it started mining, on whatever GPU was there, with
            // whatever defaults the rest of the line implied. That is how a
            // one-line check turned into an unclaimed run on a card another
            // lane could have been measuring on.
            //
            // The same mistake is already documented ten lines above for
            // --algo, where swallowing an unknown algorithm silently mined
            // Ergo instead of Pearl. Fixing it there and not here left the
            // hole open for every other flag.
            fprintf(stderr,
                    "unknown option '%s'. Run with --help for the list.\n",
                    a.c_str());
            return 1;
        }
    }

    if (workerChild) {
        // Each GPU worker keeps its own donation state, shutdown record and log files under gpu-<index>.
        if (workerIdText.empty() || workerIdText.size() > 3 || workerIdText.find_first_not_of("0123456789") != std::string::npos) return 1;
        const std::filesystem::path sub = "gpu-" + workerIdText;
        cfg.stateDir = (std::filesystem::path(cfg.stateDir) / sub).string();
        if (!cfg.log.dir.empty()) cfg.log.dir = (std::filesystem::path(cfg.log.dir) / sub).string();
    }

    if (cli.command == Command::Run && !workerChild && cfg.tuning.ocResetOnly && !kGpuTuningShipped) {
        fprintf(stderr, "%s\n", gpuTuningUnsupportedMessage());
        return kExitOk;  // reset-only never starts mining
    }
    if (kGpuTuningShipped && cli.command == Command::Run && !workerChild && cfg.tuning.ocResetOnly && !tuningPlatform.windows && !tuningPlatform.wsl) {
        std::vector<DeviceIdentity> present;
        if (!queryDevices(&present)) {
            fprintf(stderr, "no CUDA device found\n");
            return kExitDeviceRefused;
        }
        DeviceSelection selection;
        if (!selectDevices(cfg.devices, present, compiledCc(), &selection)) {
            for (const auto &r : selection.refusals) fprintf(stderr, "device refused: %s\n", r.c_str());
            return kExitDeviceRefused;
        }
        std::vector<GpuTuningDevice> selected;
        for (const DeviceIdentity &d : selection.chosen) selected.push_back({d.index, d.uuid});
        GpuTuning reset(cfg.tuning, std::move(selected), cfg.stateDir, nullptr, tuningPlatform);
        std::vector<std::string> events;
        if (reset.resetOnly(&events)) {
            for (const std::string &line : events) fprintf(stderr, "%s\n", line.c_str());
            return kExitOk;
        }
        for (const std::string &line : events) fprintf(stderr, "%s\n", line.c_str());
        return kExitRuntime;  // reset-only must never fall through into a mining run when restoration fails
    }

    // ---- product layer -> run options (CLI already overrode the config file)
    if (cli.command == Command::Run && !opt.bench) {
        if (!cfg.pools.empty()) {
            opt.poolHost = cfg.pools[0].host;
            opt.poolPort = cfg.pools[0].port;
            opt.poolProofCompression = cfg.pools[0].compression == "auto" && poolPresetSpeaksV2(cfg.pools[0].preset);
            opt.poolAccount = cfg.pools[0].account;
            // explicit policy flags (research/harness) win over the pool entry's own
            if (!pearlNetworkSeen && cfg.pools[0].policyKnown &&
                !parsePearlNetworkOption(cfg.pools[0].network, &opt.pearlPolicy.network)) {
                fprintf(stderr, "pool network '%s' is not supported\n", cfg.pools[0].network.c_str());
                return kExitConfig;
            }
            if (!pearlSemanticsSeen && cfg.pools[0].policyKnown &&
                !parsePearlTargetSemanticsOption(cfg.pools[0].targetSemantics, &opt.pearlPolicy.targetSemantics)) {
                fprintf(stderr, "pool target_semantics '%s' is not supported\n", cfg.pools[0].targetSemantics.c_str());
                return kExitConfig;
            }
        }
        opt.wallet = cfg.wallet;
        opt.worker = cfg.worker;
    }

    if (cli.command == Command::ListDevices) {
        std::vector<DeviceIdentity> present;
        if (!queryDevices(&present)) {
            printf("no CUDA device found\n");
            return kExitDeviceRefused;
        }
        printf("%zu CUDA device(s); this build contains %s kernels\n", present.size(), buildArchLabel().c_str());
        printDeviceTable(present);
        return kExitOk;
    }

    if (cli.command == Command::Selftest) {
        std::vector<DeviceIdentity> present;
        if (!queryDevices(&present)) {
            fprintf(stderr, "no CUDA device found\n");
            return kExitDeviceRefused;
        }
        DeviceSelection sel;
        if (!selectDevices(cfg.devices, present, compiledCc(), &sel)) {
            for (const auto &r : sel.refusals) fprintf(stderr, "device refused: %s\n", r.c_str());
            return kExitDeviceRefused;
        }
        bool allPass = true;
        for (const DeviceIdentity &d : sel.chosen) {
            cudaSetDevice(d.index);
            applyProductFixedSelection();
            const SelftestReport report = runSelftest(d, compiledCc(), [] { return std::unique_ptr<Algorithm>(createAlgorithm("pearl-pow")); });
            printf("%s", report.text(d).c_str());
            if (cli.ov.json) printf("%s\n", report.json(d).c_str());
            allPass = allPass && report.ok();
            fflush(stdout);
        }
        return allPass ? kExitOk : kExitSelftestFailed;
    }

    if (!buildAllowsAlgorithm(algoName)) {
        fprintf(stderr, "this CanaryMiner build supports only pearl-pow\n");
        return 1;
    }

    if (!pearlWorkRangePolicy(opt.pearlWorkRange, algoName=="pearl-pow",
                              opt.bench || opt.formalBenchmark.enabled, !opt.poolHost.empty())) {
        fprintf(stderr, "--pearl-work-range requires production Pearl pool mode\n"); return 1;
    }
    const bool anyFormalOption = formalSeen || formalNonceSeen ||
                                 formalShapeSeen || formalKernelSeen ||
                                 formalTileSeen;
    if (anyFormalOption && !formalSeen) {
        fprintf(stderr,
                "formal benchmark identity options require --bench-formal\n");
        return 1;
    }
    if (formalSeen) {
        if (!opt.bench || algoName != "pearl-pow") {
            fprintf(stderr,
                    "--bench-formal requires CUDA --bench --algo pearl-pow\n");
            return 1;
        }
        if (!formalNonceSeen || !formalShapeSeen || !formalKernelSeen ||
            !formalTileSeen) {
            fprintf(stderr,
                    "--bench-formal requires --bench-start-nonce, "
                    "--pearl-shape-id, --pearl-kernel-id, and --pearl-tile-id\n");
            return 1;
        }
        if (!validFormalBenchmarkRequest(opt.formalBenchmark)) {
            fprintf(stderr,
                    "formal shape/kernel/tile IDs must match "
                    "[a-z0-9][a-z0-9._-]*\n");
            return 1;
        }
        uint64_t formalBatch = 0, formalInterval = 0, formalHeight = 0;
        if (plainCount != 1 || batchCount != 1 || intervalCount != 1) {
            fprintf(stderr,
                    "--bench-formal requires exactly one each of --plain, "
                    "--batch 4194304, and --interval 5\n");
            return 1;
        }
        if (batchText != "4194304" || intervalText != "5" ||
            !parseFormalUnsignedDecimal(batchText, &formalBatch) ||
            formalBatch != kPearlFormalBenchmarkBatch ||
            !parseFormalUnsignedDecimal(intervalText, &formalInterval) ||
            formalInterval != kPearlFormalBenchmarkReportSeconds) {
            fprintf(stderr,
                    "--bench-formal requires canonical --batch 4194304 and "
                    "--interval 5 decimal tokens\n");
            return 1;
        }
        if (benchHeightCount > 1 ||
            (benchHeightCount == 1 &&
             (benchHeightText != "1851444" ||
              !parseFormalUnsignedDecimal(benchHeightText, &formalHeight) ||
              formalHeight != kPearlBenchmarkEpoch))) {
            fprintf(stderr,
                    "--bench-formal accepts only the exact decimal fixture "
                    "height 1851444\n");
            return 1;
        }
        if (opt.benchEpoch != kPearlBenchmarkEpoch) {
            fprintf(stderr,
                    "--bench-formal uses the fixed pearl-synthetic-v1 epoch %llu\n",
                    static_cast<unsigned long long>(kPearlBenchmarkEpoch));
            return 1;
        }
        if (benchEpochSecondsCount != 0) {
            fprintf(stderr,
                    "--bench-formal rejects any explicit --bench-epoch-secs "
                    "because the "
                    "pearl-synthetic-v1 fixture has one fixed epoch\n");
            return 1;
        }
        if (mclkCount || memOcCount) {
            fprintf(stderr,
                    "--bench-formal rejects --mclk-offset and --mem-oc "
                    "even when their explicit value would match a default\n");
            return 1;
        }
        const char *forbidden[] = {
            "CANARY_PEARL_SHAPE", "CANARY_PEARL_TILE",
            "CANARY_PEARL_BENCH_ALL_HIT", "CANARY_PEARL_HARD_TARGET",
            "CANARY_PEARL_PROFILE_INTERVAL"};
        for (const char *name : forbidden) {
            if (getenv(name)) {
                fprintf(stderr,
                        "--bench-formal rejects inherited environment variable %s\n",
                        name);
                return 1;
            }
        }
    }

    // `--pool` with an empty value swallows whatever follows it, so a launcher
    // that expands a variable to nothing quietly mines to a host called
    // "--worker". Refuse it rather than spend a connection finding out.
    if (!opt.poolHost.empty() &&
        (opt.poolHost[0] == '-' || opt.poolPort <= 0 ||
         opt.poolPort > 65535)) {
        fprintf(stderr,
                "--pool needs HOST:PORT, got '%s:%d'. An empty value in a "
                "launcher script is the usual cause.\n",
                opt.poolHost.c_str(), opt.poolPort);
        return 1;
    }

    if (!opt.pearlTranscript.empty() &&
        (algoName != "pearl-pow" || opt.poolHost.empty())) {
        fprintf(stderr, "--pearl-transcript requires --algo pearl-pow with --pool.\n");
        return 1;
    }

    const char *productionMode=getenv("CANARY_PEARL_PRODUCTION_MODE");
    if (productionMode && !pearlProductionValue(productionMode)) return 1;
    opt.pearlProduction=productionMode!=nullptr;
    const char *staleContinuation=getenv("CANARY_PEARL_STALE_CONTINUATION");
    const bool personalStale=staleContinuation &&
        !std::strcmp(staleContinuation,"canary-personal-stale-continuation-v1");
    if (staleContinuation && (!personalStale || !opt.pearlProduction)) return 1;
    PearlSustainedPolicy sustainedPolicy;
    const char *sustainedMode = getenv("CANARY_PEARL_SUSTAINED_MODE");
    const char *sustainedSeconds = getenv("CANARY_PEARL_SUSTAINED_SECONDS");
    if (!pearlParseSustained(sustainedMode, sustainedSeconds, &sustainedPolicy)) {
        fprintf(stderr,"invalid sustained duration controls; refusing before device/source setup\n"); return 1;
    }
    opt.pearlSustainedSeconds = sustainedPolicy.seconds;
    opt.pearlSustainedPublic = sustainedPolicy.publicRun;
    const char *acceptedQuota = getenv("CANARY_PEARL_ACCEPTED_QUOTA");
    if (pearlAuthorizationSeen && algoName != "pearl-pow") {
        fprintf(stderr, "--pearl-authorization-mode requires --algo pearl-pow\n");
        return 1;
    }
    // Guarded/private research owns its existing authorization rewrite. Never
    // combine it with normal native SOLO decoration or alter its default path.
    bool authorizationResearch = getenv("CANARY_PEARL_NATIVE_EVIDENCE") != nullptr ||
        getenv("CANARY_PEARL_PRIVATE_EXECUTION") != nullptr ||
        getenv("CANARY_PEARL_PUBLIC_EXECUTION") != nullptr ||
        getenv("CANARY_PEARL_LOCAL_RETIREMENTS") != nullptr ||
        getenv("CANARY_PEARL_ACCEPTED_QUOTA") != nullptr ||
        getenv("CANARY_PEARL_SUSTAINED_MODE") != nullptr;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    authorizationResearch = authorizationResearch || !guardControlText.empty();
#endif
    if (!pearlSoloAuthorizationAllowed(opt.pearlSoloAuthorization, algoName.c_str(),
            opt.bench, !opt.poolHost.empty(), authorizationResearch)) {
        fprintf(stderr, "SOLO authorization requires a normal Pearl pool run without research guards\n");
        return 1;
    }
    const char *nativeMode = getenv("CANARY_PEARL_NATIVE_EVIDENCE");
    if (!pearlParseAcceptedQuota(acceptedQuota, &opt.pearlAcceptedQuota)) {
        fprintf(stderr, "invalid accepted quota; refusing before device/source setup\n");
        return 1;
    }
    opt.pearlPublicExecution=pearlParsePublicExecution(getenv("CANARY_PEARL_NATIVE_EVIDENCE"),getenv("CANARY_PEARL_PUBLIC_EXECUTION"));
    const bool publicExecution=pearlPublicEnabled(opt.pearlPublicExecution);
    if(!pearlPublicEnvironment(opt.pearlPublicExecution))return 1;
    opt.pearlPrivateExecution=pearlParsePrivateExecution(getenv("CANARY_PEARL_PRIVATE_EXECUTION"),getenv("CANARY_PEARL_PRIVATE_EXECUTION_MODE"));
    const bool privateExecution=pearlPrivateEnabled(opt.pearlPrivateExecution);
    if (!pearlPrivateEnvironment(opt.pearlPrivateExecution)) return 1;
    opt.pearlShapeExperiment = pearlParseShapeExperiment(
        getenv("CANARY_PEARL_SHAPE_EXPERIMENT"), getenv("CANARY_PEARL_SHAPE"), getenv("CANARY_PEARL_TILE"));
    const bool shapeExperiment = pearlShapeExperimentEnabled(opt.pearlShapeExperiment);
    if (opt.pearlShapeExperiment == PearlShapeExperiment::Invalid ||
        (shapeExperiment && (!pearlShapeExperimentCompiled() || !pearlCudaShapeExperimentCompiled() ||
         !pearlGenericNativeEvidenceMode(nativeMode) ||
         (!opt.pearlAcceptedQuota && !opt.pearlSustainedSeconds && !opt.pearlProduction)))) {
        fprintf(stderr,"invalid shape experiment; refusing before device/source setup\n"); return 1;
    }
    const char *evidenceBatch = (pearlGenericNativeEvidenceMode(nativeMode) || pearlPrivateGeneric(opt.pearlPrivateExecution)) ? "256" : "1048576";
    if (((sustainedPolicy.seconds || opt.pearlProduction) && (batchCount != 1 || batchText != evidenceBatch)) ||
        !pearlSustainedAdmission(opt, algoName.c_str(), sustainedMode, sustainedSeconds,
            getenv("CANARY_PEARL_ACCEPTED_QUOTA"), getenv("CANARY_PEARL_LOCAL_RETIREMENTS"),
            getenv("CANARY_PEARL_EVIDENCE_RUN_ID"), getenv("CANARY_PEARL_EXPERIMENT"),
            (getenv("CANARY_PEARL_SHAPE") && !shapeExperiment) || !pearlShapeExperimentTileAllowed(opt.pearlShapeExperiment,getenv("CANARY_PEARL_TILE")) ||
            getenv("CANARY_PEARL_PROFILE_INTERVAL") || getenv("CANARY_PEARL_A_ROOT_FUSION") ||
            (getenv("CANARY_PEARL_ATTEMPT_OVERLAP") && !privateExecution && !publicExecution) || getenv("CANARY_PEARL_HARD_TARGET") ||
            getenv("CANARY_PEARL_BENCH_ALL_HIT"), nativeMode)) {
        fprintf(stderr,"invalid sustained admission; refusing before device/source setup\n"); return 1;
    }
    if ((acceptedQuota && (batchCount != 1 || batchText != evidenceBatch)) ||
        !pearlAcceptedQuotaAdmission(opt, algoName.c_str(), acceptedQuota,
            getenv("CANARY_PEARL_LOCAL_RETIREMENTS"), getenv("CANARY_PEARL_EVIDENCE_RUN_ID"),
            getenv("CANARY_PEARL_EXPERIMENT"),
            (getenv("CANARY_PEARL_SHAPE") && !shapeExperiment) || !pearlShapeExperimentTileAllowed(opt.pearlShapeExperiment,getenv("CANARY_PEARL_TILE")) ||
            getenv("CANARY_PEARL_PROFILE_INTERVAL") || getenv("CANARY_PEARL_A_ROOT_FUSION") ||
            (getenv("CANARY_PEARL_ATTEMPT_OVERLAP") && !privateExecution && !publicExecution) || getenv("CANARY_PEARL_HARD_TARGET") ||
            getenv("CANARY_PEARL_BENCH_ALL_HIT"), nativeMode)) {
        fprintf(stderr, "invalid accepted quota; refusing before device/source setup\n");
        return 1;
    }
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    if(!guardControlText.empty()) {
        PearlPoolSteadyClock guardClock;
        if(algoName!="pearl-pow" || opt.bench || opt.poolHost!="127.0.0.1" ||
           (!opt.pearlAcceptedQuota && !opt.pearlSustainedSeconds && !opt.pearlProduction) ||
           !guardControl.admit(guardControlText,getenv("CANARY_PEARL_EVIDENCE_RUN_ID"),
                               opt.pearlTranscript,guardClock.nowMs(),personalStale ? guard_retirement_v2::Profile::PersonalProduction : opt.pearlProduction ? guard_retirement_v2::Profile::Production : sustainedPolicy.public24h ?
                                   guard_retirement_v2::Profile::Public24h : guard_retirement_v2::Profile::Legacy)) {
            fprintf(stderr,"invalid explicit guard control admission\n"); return 1;
        }
        opt.pearlGuardRetirementSession=guardControl.session();
    }
#endif
    bool shapeOwnedGuard = false;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    shapeOwnedGuard = opt.pearlGuardRetirementSession != nullptr;
#endif
    if (opt.pearlProduction && !shapeOwnedGuard) return 1;
    if(publicExecution) {
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
        if(!pearlPublicPolicy(opt.pearlPublicExecution,shapeOwnedGuard && opt.poolHost=="127.0.0.1",
            shapeOwnedGuard?opt.pearlGuardRetirementSession->cap():0,opt.batch,opt.pearlAcceptedQuota,opt.pearlSustainedSeconds,opt.pearlSustainedPublic))return 1;
#else
        return 1;
#endif
    }
    if (privateExecution && (!shapeOwnedGuard || opt.pearlSustainedPublic ||
            (!opt.pearlAcceptedQuota && !opt.pearlSustainedSeconds && !opt.pearlProduction))) return 1;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    if (privateExecution && opt.pearlGuardRetirementSession->cap()!=20) return 1;
#endif
    if (shapeExperiment && !pearlShapeExperimentAdmission(opt.pearlShapeExperiment,
            pearlGenericNativeEvidenceMode(nativeMode),
            opt.pearlAcceptedQuota || opt.pearlSustainedSeconds || opt.pearlProduction, shapeOwnedGuard, false)) {
        fprintf(stderr,"shape experiment requires an owned bounded guard\n"); return 1;
    }
    const char *integratedPipeline = getenv("CANARY_PEARL_INTEGRATED_PIPELINE");
    if (!pearlIntegratedPipelineAdmission(pearlParseIntegratedPipeline(integratedPipeline),
            pearlIntegratedPipelineContext(opt, algoName.c_str(),
                pearlCudaIntegratedPipelineCompiled(), shapeOwnedGuard))) {
        fprintf(stderr,"invalid or non-isolated integrated Pearl pipeline; refusing before device/source setup\n");
        return 1;
    }
    const char *rootFusion = getenv("CANARY_PEARL_A_ROOT_FUSION");
    if (rootFusion && (algoName != "pearl-pow" ||
        !pearl::aRootFusionAdmission(pearl::parseARootFusion(rootFusion),
            pearlCudaARootFusionCompiled(), opt.formalBenchmark.enabled,
            getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr, opt.bench,
            opt.poolHost, opt.poolPort,
            pearl::parseCudaFixedExperiment(getenv("CANARY_PEARL_EXPERIMENT")) ==
                pearl::CudaFixedExperimentSelection::Wm8Runtime))) {
        fprintf(stderr, "invalid or non-isolated A root fusion; refusing before device/source setup\n");
        return 1;
    }
    const char *overlap = getenv("CANARY_PEARL_ATTEMPT_OVERLAP");
    if (overlap && (algoName != "pearl-pow" ||
        !pearl::attemptOverlapAdmission(pearl::parseAttemptOverlap(overlap),
            pearlCudaAttemptOverlapCompiled(), opt.formalBenchmark.enabled,
            getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr, opt.bench,
            opt.poolHost, opt.poolPort))) {
        fprintf(stderr, "invalid or non-isolated Pearl overlap; refusing before device/source setup\n");
        return 1;
    }
    const char *experiment = getenv("CANARY_PEARL_EXPERIMENT");
    // Includes default-off TMA WM8: capability comes from the linked backend,
    // not this CLI TU's defines. Its exact selector remains loopback-only and
    // rejects overlap/root/profiling through the shared typed admission below.
    if (experiment && (algoName != "pearl-pow" ||
        !pearl::cudaFixedExperimentConfigurationValid(
            pearl::parseCudaFixedExperiment(experiment), pearlCudaExperimentCompiled(experiment),
            opt.formalBenchmark.enabled, opt.bench, opt.poolHost, opt.poolPort,
            getenv("CANARY_PEARL_SHAPE") != nullptr, getenv("CANARY_PEARL_TILE") != nullptr,
            getenv("CANARY_PEARL_ATTEMPT_OVERLAP") != nullptr,
            getenv("CANARY_PEARL_A_ROOT_FUSION") != nullptr,
            getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr))) {
        fprintf(stderr, "invalid or non-isolated Pearl experiment; refusing before device/source setup\n");
        return 1;
    }
    const char *admission = pearlRuntimeAdmissionProblem(
        opt, algoName.c_str(), PearlRuntimeCapability::CanonicalCudaPoolV1,
        getenv("CANARY_PEARL_BENCH_ALL_HIT") != nullptr,
        getenv("CANARY_PEARL_HARD_TARGET") != nullptr);
    if (admission) {
        fprintf(stderr, "%s; refusing before device/source setup\n", admission);
        return 1;
    }
    if (algoName == "pearl-pow" && !opt.bench && !pearlLoginByAccount(opt)) {
        const std::string problem = pearlWalletProblem(opt.wallet);
        if (!problem.empty()) {
            fprintf(stderr, "invalid Pearl wallet: %s; refusing before device/source setup\n", problem.c_str());
            return 1;
        }
    }
    // A plain mining run is supervised: one worker process per selected GPU (H-032). Research, benchmark and evidence modes are not.
    bool supervised = cli.command == Command::Run && !opt.bench && !opt.formalBenchmark.enabled && !opt.pearlProduction &&
                      !opt.pearlAcceptedQuota && !opt.pearlSustainedSeconds && !privateExecution && !publicExecution && !workerChild;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    supervised = supervised && opt.pearlGuardRetirementSession == nullptr;
#endif
    // ---- devices: identity, allowlist, one device per process (nothing has touched a kernel yet)
    DeviceIdentity chosenDevice;
    std::vector<DeviceIdentity> chosenAll;
    {
        std::vector<DeviceIdentity> present;
        if (!queryDevices(&present)) {
            fprintf(stderr, "no CUDA device found\n");
            return kExitDeviceRefused;
        }
        DeviceSelection sel;
        if (!selectDevices(cfg.devices, present, compiledCc(), &sel)) {
            for (const auto &r : sel.refusals) fprintf(stderr, "device refused: %s\n", r.c_str());
            fprintf(stderr, "no usable supported device; run 'list-devices' for the identity of each GPU\n");
            return kExitDeviceRefused;
        }
        if (sel.chosen.size() != 1 && !supervised) {
            fprintf(stderr, "%zu supported devices selected; this mode drives one device per process. Pass --devices N\n", sel.chosen.size());
            return kExitUsage;
        }
        chosenAll = sel.chosen;
        chosenDevice = sel.chosen[0];
        deviceIndex = chosenDevice.index;
    }

    if (!workerChild && cfg.tuning.requested() && (!kGpuTuningShipped || tuningPlatform.windows || tuningPlatform.wsl))
        fprintf(stderr, "%s\n", gpuTuningUnsupportedMessage());
    else if (!workerChild && cfg.tuning.requested() && !supervised && !cfg.tuning.ocResetOnly)
        fprintf(stderr, "GPU tuning requires a standard supervised mining run; mining continues untuned\n");

    if (!platformInit()) {
        fprintf(stderr, "network init failed\n");
        return 1;
    }
    signal(SIGINT, onSignal);
#if defined(SIGTERM)
    signal(SIGTERM, onSignal);
#endif
#if defined(SIGPIPE)
    signal(SIGPIPE, SIG_IGN);  // a pool disconnecting mid-send must not kill us
#endif
#if defined(_WIN32)
    g_shutdownDone = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif

    if (supervised) {
        // The supervisor creates no CUDA context: it starts one worker per GPU (each a copy of this program) and watches them.
        SupervisorOptions so;
        std::unique_ptr<GpuTuning> gpuTuning;
        so.exe = selfExecutablePath();
        if (so.exe.empty()) {
            fprintf(stderr, "cannot locate this program to start the GPU workers\n");
            return kExitRuntime;
        }
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--gpu-id") { ++i; continue; }
            so.userArgs.push_back(argv[i]);
        }
        for (const DeviceIdentity &d : chosenAll) {
            WorkerSeed seed;
            seed.index = d.index;
            seed.uuid = d.uuid;
            seed.name = d.name;
            seed.pciBus = d.pciBus;
            seed.family = checkDevice(d, compiledCc()).family;
            seed.ccMajor = d.ccMajor;
            seed.ccMinor = d.ccMinor;
            so.seeds.push_back(seed);
        }
        so.version = CANARY_MINER_VERSION;
        so.archLabel = buildArchLabel();
        so.poolEndpoint = cfg.pools.empty() ? std::string() : cfg.pools[0].endpoint();
        so.poolPreset = cfg.pools.empty() ? std::string() : cfg.pools[0].preset;
        so.worker = cfg.worker;
        so.apiBind = cfg.api.bind;
        so.apiEnabled = cfg.api.enabled;
        so.apiPort = cfg.api.port;
        // Worker i serves its stats on the loopback port after the public one (or after 8085 when the public endpoint is off).
        int base = (cfg.api.enabled ? cfg.api.port : 8085) + 1;
        if (base + static_cast<int>(chosenAll.size()) > 65535) base = 18085;
        so.workerPortBase = base;
        so.donationOn = donationActive(cfg);
        so.donationDestinationConfigured = feeDestinationConfigured();
        so.drainMs = static_cast<uint64_t>(cfg.drainSeconds) * 1000u;
        // Never printed: the wallet, the donation account and its payout destination (H-031 Amendment 2).
        so.secrets = {cfg.wallet, std::string(CANARY_FEE_POOL_ACCOUNT), feeDestination()};
        so.dashboard = stdoutIsTty() && !noDashboard;
        so.stop = &g_stop;
        std::vector<GpuTuningDevice> tuningDevices;
        for (const DeviceIdentity &d : chosenAll) tuningDevices.push_back({d.index, d.uuid});
        if (kGpuTuningShipped && gpuTuningNeedsSupervisor(cfg.tuning, cfg.stateDir, tuningPlatform)) {
            gpuTuning.reset(new GpuTuning(cfg.tuning, std::move(tuningDevices), cfg.stateDir, nullptr, tuningPlatform));
            so.tuning = gpuTuning.get();
        }
        if (!so.dashboard) {
            printf("CanaryMiner %s: %zu GPU worker(s), pool %s, worker %s, donation %s\n", CANARY_MINER_VERSION, so.seeds.size(),
                   so.poolEndpoint.empty() ? "none" : so.poolEndpoint.c_str(), so.worker.c_str(),
                   donationActive(cfg) ? "on (1% developer donation, mined to the developer's pool)" : "off");
            printf("Allow about 5 minutes for the hashrate to stabilize.\n");
            fflush(stdout);
        }
        Supervisor supervisor(std::move(so));
        const int rc = supervisor.run();
        platformShutdown();
        return rc;
    }

    // Pick the GPU before anything allocates on it. Run one process per GPU
    // (each with its own --device and --mem-oc) to drive a multi-GPU rig.
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    if (deviceIndex < 0 || deviceIndex >= devCount) {
        fprintf(stderr, "--device %d out of range (%d CUDA GPU%s, 0-%d)\n",
                deviceIndex, devCount, devCount == 1 ? "" : "s", devCount - 1);
        return 1;
    }
    cudaSetDevice(deviceIndex);
    if (workerChild) {
        // The supervisor asks this worker to stop by closing its stdin: end-of-file sets the same flag as a signal.
        std::thread([] {
            char buf[256];
            while (fread(buf, 1, sizeof(buf), stdin) > 0) {
            }
            g_stop = 1;
        }).detach();
    }

    // An ordinary product run uses the live miner's fixed qualified selection instead of the auto-tuner (research modes keep theirs).
    if (cli.command == Command::Run && !opt.bench && !opt.autotune && !opt.formalBenchmark.enabled && !opt.pearlProduction && !opt.pearlAcceptedQuota &&
        !opt.pearlSustainedSeconds && !privateExecution && !publicExecution) {
        applyProductFixedSelection();
        opt.productFixedSelection = true;
    }
    std::unique_ptr<Algorithm> algo(createAlgorithm(algoName));
    if (!algo) {
        fprintf(stderr, "unknown algorithm '%s' (try --list-algos)\n", algoName.c_str());
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceIndex) != cudaSuccess) {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }

    // After the device is chosen, because an algorithm may compile for the
    // card it is actually going to run on.
    char arch[32];
    snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);

    const std::string gpuUuid = nvidiaGpuUuid(
        reinterpret_cast<const unsigned char *>(prop.uuid.bytes),
        sizeof(prop.uuid.bytes));
    if (gpuUuid.empty()) {
        fprintf(stderr, "cannot format selected CUDA device UUID\n");
        return 1;
    }

#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    if(opt.pearlGuardRetirementSession) {
        std::printf("{\"schema\":\"pearl-native-device-v1\",\"event\":\"pearl_native_device\","
                    "\"run_id\":\"%s\",\"device_index\":%u,\"gpu_uuid\":\"%s\"}\n",
                    getenv("CANARY_PEARL_EVIDENCE_RUN_ID"),static_cast<unsigned>(deviceIndex),gpuUuid.c_str());
        if(std::fflush(stdout)!=0 || std::ferror(stdout)) return 1;
    }
#endif

    // ---- product services: only for an ordinary mining run (never for the research/evidence modes, whose stdout is a
    // machine-readable protocol of its own, nor for benchmarks)
    bool plainRun = cli.command == Command::Run && !opt.bench && !opt.formalBenchmark.enabled && !opt.pearlProduction &&
                    !opt.pearlAcceptedQuota && !opt.pearlSustainedSeconds && !privateExecution && !publicExecution;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    plainRun = plainRun && opt.pearlGuardRetirementSession == nullptr;
#endif
    Redactor redactor;
    StatsPublisher statsPublisher;
    StatsServer statsServer;
    std::unique_ptr<LogSink> logSink;
    std::unique_ptr<OutputTee> outputTee;
    std::unique_ptr<StatsObserver> statsObserver;
    std::unique_ptr<ShutdownDriver> shutdownDriver;
    std::atomic<bool> runLoopDone{false};
    std::atomic<om::FeeSession *> feeForShutdown{nullptr};  // set once the fee session exists; the shutdown persist step flushes it
    if (plainRun) {
        redactor.addSecret(cfg.wallet, true);
        redactor.addSecret(CANARY_FEE_POOL_ACCOUNT);  // H-031 Amendment 2: the donation account and payout never reach a log
        redactor.addSecret(feeDestination());
        if (cfg.log.redactWorker) redactor.addSecret(cfg.worker);
        if (!cfg.log.dir.empty()) {
            LogSinkConfig lc;
            lc.dir = cfg.log.dir;
            lc.maxFileBytes = cfg.log.maxFileBytes;
            lc.maxFiles = cfg.log.maxFiles;
            logSink.reset(new LogSink(lc, &redactor));
            std::string logError;
            if (!logSink->open(&logError)) {
                fprintf(stderr, "log: %s\n", logError.c_str());
                return kExitConfig;
            }
            outputTee.reset(new OutputTee(logSink.get(), &redactor));
            if (!outputTee->start()) {
                fprintf(stderr, "log: cannot route output through the log sink\n");
                return kExitRuntime;
            }
        }
        if (!workerChild) {
            printf("%s", effectiveConfigText(cfg, cli.ov.configPath.empty() ? "command line" : "config file, command line overrides").c_str());
            fflush(stdout);
        }

        StatsDevice base;
        base.index = chosenDevice.index;
        base.uuid = chosenDevice.uuid;
        base.pciBus = chosenDevice.pciBus;
        base.name = chosenDevice.name;
        base.family = checkDevice(chosenDevice, compiledCc()).family;
        base.ccMajor = chosenDevice.ccMajor;
        base.ccMinor = chosenDevice.ccMinor;
        const std::string poolEndpoint = cfg.pools.empty() ? std::string() : cfg.pools[0].endpoint();
        const std::string poolPreset = cfg.pools.empty() ? std::string() : cfg.pools[0].preset;
        statsPublisher.update([&](StatsSnapshot *snap) {
            snap->version = CANARY_MINER_VERSION;
            snap->poolEndpoint = poolEndpoint;
            snap->poolPreset = poolPreset;
            snap->poolState = "connecting";
            snap->devices.assign(1, base);
            snap->fee.destinationConfigured = feeDestinationConfigured();
            snap->fee.enabled = cfg.donation;
            if (!cfg.donation) { snap->fee.state = "off"; snap->fee.note = "developer donation is off (--no-donation or donation: false)"; }
        });
        statsObserver.reset(new StatsObserver(&statsPublisher, base, poolEndpoint, poolPreset));
        opt.observer = statsObserver.get();
        if (cfg.api.enabled) {
            std::string apiError;
            if (!statsServer.start(cfg.api.bind, cfg.api.port, [&statsPublisher] { return statsToJson(statsPublisher.snapshot()); },
                                   versionToJson(CANARY_MINER_VERSION, buildArchLabel()), &apiError))
                fprintf(stderr, "stats endpoint disabled: %s\n", apiError.c_str());
        }

        ShutdownHooks hooks;
        hooks.inFlight = [&statsPublisher]() -> uint64_t {
            const StatsSnapshot s = statsPublisher.snapshot();
            return statsTotals(s).pending;
        };
        hooks.markUnknown = [&statsPublisher](uint64_t n) {
            statsPublisher.update([n](StatsSnapshot *s) {
                if (s->devices.empty()) return;
                StatsOutcomes &o = s->devices[0].outcomes;
                o.unknown = saturatingCounterAdd(o.unknown, n);
                o.pending = o.pending > n ? o.pending - n : 0;
            });
        };
        hooks.persist = [&cfg, &statsPublisher, &feeForShutdown](bool forced, uint64_t unknown, const std::string &reason) {
            // The donation state goes first: forced exits (console close, a drain timeout) never reach the fee session's finish().
            if (om::FeeSession *fee = feeForShutdown.load()) fee->flushForShutdown();
            if (cfg.stateDir.empty()) return true;
            std::string writeError;
            return writeFileAtomic((std::filesystem::path(cfg.stateDir) / "shutdown-record.json").string(),
                                   shutdownRecordJson(forced, unknown, reason, statsPublisher.snapshot()), &writeError);
        };
        hooks.close = [&statsServer]() { statsServer.stop(); return true; };
        hooks.release = [&]() {
            if (!runLoopDone.load()) return true;  // forced path: the process exits and the OS releases the device
            cudaSetDevice(deviceIndex);
            algo.reset();
            return true;
        };
        shutdownDriver.reset(new ShutdownDriver(static_cast<uint64_t>(cfg.drainSeconds) * 1000u, std::move(hooks), &g_stop, &runLoopDone,
                                                [&statsPublisher](const char *state) {
                                                    statsPublisher.update([state](StatsSnapshot *s) { s->state = state; });
                                                }, &g_closeRequested));
        shutdownDriver->start();
    }

    // H-019 D3 (live v28u): a guard stop restarts this GPU in a fresh run after 10 s, at most 3 per rolling hour.
    // Transport losses reconnect inside runMiner and never reach this budget. An operator stop is a clean exit.
    // H-022: the donation scheduler also lives in this outer loop. A donation window ends the user's runMiner at a loop top, a second runMiner
    // mines the donation destination, then the user run resumes; none of that touches the D3 budget.
    om::GuardRestartBudget guardBudget;
    om::FeeLoopDeps feeDeps;
    feeDeps.nowMs = [] { return steadyNowMs(); };
    feeDeps.run = [&](const RunOptions &o) {
        return runMiner(algo.get(), o, prop.name, prop.totalGlobalMem / 1e9, arch, &g_stop, gpuUuid.c_str(), static_cast<unsigned>(deviceIndex));
    };
    feeDeps.stopped = [] { return g_stop != 0; };
    feeDeps.sleepMs = [](uint64_t ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
    feeDeps.log = [](const std::string &line) { fprintf(stderr, "%s\n", line.c_str()); };
    std::unique_ptr<FeeStatsObserver> feeObserver;
    std::unique_ptr<om::FeeRandom> feeRandom;
    std::unique_ptr<om::FeeSession> feeSession;
    if (plainRun && donationActive(cfg)) {
        feeObserver.reset(new FeeStatsObserver(&statsPublisher));
        feeDeps.feeObserver = feeObserver.get();
        feeDeps.feeWindowBoundary = [&](bool opening) { feeObserver->boundary(opening); };
        feeDeps.userRunSuspended = [&] { if (statsObserver) statsObserver->foldRun(); };
        feeDeps.publish = [&](const om::FeeStatus &f) {
            statsPublisher.update([&](StatsSnapshot *snap) {
                snap->fee.state = f.windowActive ? "donation" : "user";
                snap->fee.windowActive = f.windowActive;
                snap->fee.cycle = f.cycle;
                snap->fee.userSeconds = f.userMs / 1000;
                snap->fee.feeSeconds = f.feeMs / 1000;
                snap->fee.debtSeconds = f.debtMs / 1000;
                snap->fee.poolFailures = f.poolFailures;
                snap->fee.windowsCompleted = f.windowsCompleted;
                snap->fee.pool = feePoolEndpoint();
                snap->fee.login = "developer";  // H-031 Amendment 2: the donation account is never presented
                snap->fee.note = "developer donation: 1% (99:1 per 100 productive minutes)";
            });
        };
        feeRandom.reset(new om::SystemFeeRandom());
        feeSession.reset(new om::FeeSession(om::productFeePolicy(), feeRandom.get(),
                                            om::FeeStateFile(!cfg.log.dir.empty() ? cfg.log.dir : cfg.stateDir), feeDestination(), feeDeps));
        feeSession->load();
        feeForShutdown.store(feeSession.get());
        fprintf(stderr, "donation-policy: %s\n", feePolicyBuildLabel());
        fprintf(stderr, "developer donation: 1%% (99:1 per 100 productive minutes), mined to the developer's pool; disable with --no-donation\n");
    } else if (plainRun) {
        fprintf(stderr, "developer donation: off\n");
    }
    int rc = om::runOuterLoop(opt, feeSession.get(), feeDeps, &guardBudget, kExitOk, kExitRuntime);
    bool forcedShutdown = false;
    if (shutdownDriver) {
        shutdownDriver->runLoopFinished();
        if (!shutdownDriver->machine().stopRequested()) shutdownDriver->requestStop(rc == 0 ? "run-complete" : "run-error");
        shutdownDriver->join();
        forcedShutdown = shutdownDriver->machine().forced();
        if (outputTee) outputTee->stop();
#if defined(_WIN32)
        if (g_shutdownDone) SetEvent(g_shutdownDone);  // release a console-close handler that is waiting
#endif
    }
    if (opt.pearlSustainedSeconds || opt.pearlProduction) algo.reset();
    platformShutdown();
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
    if(!guardControl.close()) return 1;
#endif
    if (forcedShutdown) return kExitShutdownForced;
    return opt.pearlSustainedSeconds && g_stop ? 1 : rc;
}
