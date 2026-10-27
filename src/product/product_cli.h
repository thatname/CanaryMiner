// Product command line: subcommands, the standard miner options, and the options that override the config file.
//
//   canary-miner [run] [options]      mine (default: no subcommand needed)
//   canary-miner list-devices         identity of every CUDA device and whether it is supported
//   canary-miner selftest             offline kernel + proof self-test on each selected device (no pool, no network)
//   canary-miner status [--json]      read the local stats endpoint of a running miner
//   canary-miner version              version, build architecture, donation policy
//
//   -o, --pool SPEC       HOST:PORT | stratum+tcp://HOST:PORT | kryptex | herominers | f2pool (repeatable; replaces the config's pools)
//   -u, --wallet ADDR     prl1 payout address, optionally ADDR.WORKER (alias --user); F2Pool also takes ACCOUNT.WORKER
//       --worker NAME     rig name; overrides the WORKER part of --wallet
//   -p, --password P      pool password field (default x; passed to the pool protocol)
//       --gpu-id LIST     GPUs to use (alias of --devices)
//       --devices LIST    comma separated: all | index | GPU-uuid | PCI bus id   (alias --device N)
//       --api-enable      turn the stats endpoint on (port 21550 unless --api-port is given)
//       --api-port N      stats endpoint port (0 disables the endpoint)
//       --api-bind ADDR   loopback only
//       --no-donation     turn the 1% developer donation off
//       --config FILE     config-v1 JSON file
//       --log-dir DIR     rotating redacted log files
//       --json            (status) print the raw JSON
//   -h, --help            the grouped help text (productHelpText); the legacy option loop in miner.cu also answers it
//
// Accepted and ignored, for harnesses written for other miners: --algorithm pearl|pearlhash, --disable-cpu.
// GPU tuning flags are consumed here; this release ignores them with a notice (kGpuTuningShipped in gpu_tuning.h).
//
// Everything this parser does not recognise is returned in `rest` untouched for the legacy option loop in miner.cu
// (policy flags such as --pearl-network, benchmark flags, ...), so existing harness command lines keep working.
// CLI values override the config file; nothing here reads a file or the network.

#pragma once

#include <climits>
#include <string>
#include <vector>

#include "config.h"

namespace om {

enum class Command { Run, ListDevices, Selftest, Status, Version };

/** Process exit codes of the product commands (the research option paths keep their historical 1). */
enum ExitCode : int {
    kExitOk = 0,
    kExitRuntime = 1,         // mining stopped on an error
    kExitUsage = 2,           // bad command line
    kExitConfig = 3,          // config file unreadable, invalid or incomplete
    kExitDeviceRefused = 4,   // no usable supported device, or a requested device is refused
    kExitSelftestFailed = 5,  // a selftest step failed
    kExitStatusDown = 6,      // `status` could not reach a running miner
    kExitShutdownForced = 7,  // the drain deadline passed; shares in flight were recorded as unknown
};

/** The stats port `--api-enable` uses when no `--api-port` is given. The config file default stays 8085. */
constexpr int kApiEnablePort = 21550;

struct CliOverrides {
    std::string configPath;
    std::vector<std::string> pools;
    bool walletSet = false;
    std::string wallet;  // as typed: ADDR or ADDR.WORKER; split by applyOverrides
    bool workerSet = false;
    std::string worker;
    bool devicesSet = false;
    std::vector<std::string> devices;
    bool apiEnable = false;
    bool apiBindSet = false;
    std::string apiBind;
    bool apiPortSet = false;
    int apiPort = 0;
    bool logDirSet = false;
    std::string logDir;
    bool json = false;
    bool noDonation = false;  // --no-donation: switch the developer donation off for this run
    GpuTuningConfig tuning;
};

struct CliParse {
    Command command = Command::Run;
    bool commandExplicit = false;
    bool help = false;  // -h / --help was given (it is also left in `rest` for the legacy loop)
    CliOverrides ov;
    std::vector<std::string> rest;  // untouched legacy arguments (argv[0] excluded)
};

inline std::vector<std::string> splitCommaList(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

/** Options of other miners that are refused: overclocking belongs to the vendor tools, not to this miner. */
inline bool isOverclockOption(const std::string &a) {
    static const char *const kPrefixes[] = {"--gpu-cclock", "--gpu-coffset", "--gpu-mclock", "--gpu-moffset", "--gpu-plimit", "--gpu-fan", "--oc-"};
    for (const char *p : kPrefixes)
        if (a.rfind(p, 0) == 0) return true;
    return false;
}

/** Returns false with *error set on a malformed product option. Does not validate values beyond their shape. */
inline bool parseProductCli(int argc, const char *const *argv, CliParse *out, std::string *error) {
    CliParse r;
    int i = 1;
    if (i < argc && argv[i][0] != '-') {
        const std::string c = argv[i];
        if (c == "run") r.command = Command::Run;
        else if (c == "list-devices") r.command = Command::ListDevices;
        else if (c == "selftest") r.command = Command::Selftest;
        else if (c == "status") r.command = Command::Status;
        else if (c == "version") r.command = Command::Version;
        else {
            if (error) *error = "unknown command '" + c + "' (run, list-devices, selftest, status, version; or start with an option, e.g. --pool)";
            return false;
        }
        r.commandExplicit = true;
        ++i;
    }
    auto need = [&](int &k, const char *opt, std::string *value) {
        if (k + 1 >= argc) {
            if (error) *error = std::string(opt) + " needs a value";
            return false;
        }
        *value = argv[++k];
        return true;
    };
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        std::string v;
        auto tuningValue = [&](const char *name, std::string *value) {
            const std::string option(name);
            if (a == option) {
                if (!need(i, name, value)) return false;
                if (value->empty()) { if (error) *error = option + " needs a value"; return false; }
                return true;
            }
            const std::string equals = option + "=";
            if (a.rfind(equals, 0) != 0) return false;
            *value = a.substr(equals.size());
            if (value->empty()) {
                if (error) *error = option + " needs a value";
                return false;
            }
            return true;
        };
        if (a == "--config") {
            if (!need(i, "--config", &v)) return false;
            if (!r.ov.configPath.empty()) {
                if (error) *error = "--config given twice";
                return false;
            }
            r.ov.configPath = v;
        } else if (a == "--pool" || a == "-o") {
            if (!need(i, a.c_str(), &v)) return false;
            if (v.empty()) {
                if (error) *error = "--pool needs a preset name or HOST:PORT; an empty value in a launcher script is the usual cause";
                return false;
            }
            r.ov.pools.push_back(v);
        } else if (a == "--wallet" || a == "--user" || a == "-u") {
            if (!need(i, a.c_str(), &v)) return false;
            if (v.empty()) {
                if (error) *error = a + " needs your PRL address (prl1...), optionally followed by .WORKER";
                return false;
            }
            r.ov.walletSet = true;
            r.ov.wallet = v;
        } else if (a == "--worker") {
            if (!need(i, "--worker", &v)) return false;
            r.ov.workerSet = true;
            r.ov.worker = v;
        } else if (a == "--password" || a == "-p") {
            // The legacy loop in miner.cu owns --pass; the value travels there unchanged.
            if (!need(i, a.c_str(), &v)) return false;
            r.rest.push_back("--pass");
            r.rest.push_back(v);
        } else if (a == "--devices" || a == "--device" || a == "--gpu-id") {
            if (!need(i, a.c_str(), &v)) return false;
            for (const auto &d : splitCommaList(v)) r.ov.devices.push_back(d);
            if (r.ov.devices.empty()) {
                if (error) *error = a + " needs at least one device";
                return false;
            }
            r.ov.devicesSet = true;
        } else if (a == "--api-enable") {
            r.ov.apiEnable = true;
        } else if (a == "--api-bind") {
            if (!need(i, "--api-bind", &v)) return false;
            r.ov.apiBindSet = true;
            r.ov.apiBind = v;
        } else if (a == "--api-port") {
            if (!need(i, "--api-port", &v)) return false;
            if (v.empty() || v.size() > 5) {
                if (error) *error = "--api-port needs a port number (0 disables the endpoint)";
                return false;
            }
            int p = 0;
            for (char c : v) {
                if (c < '0' || c > '9') {
                    if (error) *error = "--api-port needs a port number (0 disables the endpoint)";
                    return false;
                }
                p = p * 10 + (c - '0');
            }
            if (p > 65535) {
                if (error) *error = "--api-port must be 0..65535";
                return false;
            }
            r.ov.apiPortSet = true;
            r.ov.apiPort = p;
        } else if (a == "--log-dir") {
            if (!need(i, "--log-dir", &v)) return false;
            r.ov.logDirSet = true;
            r.ov.logDir = v;
        } else if (a == "--json") {
            r.ov.json = true;
        } else if (a == "--no-donation") {
            r.ov.noDonation = true;
        } else if (a == "--gpu-cclock" || a.rfind("--gpu-cclock=", 0) == 0) {
            if (r.ov.tuning.gpuCclockSet) { if (error) *error = "--gpu-cclock given twice"; return false; }
            if (!tuningValue("--gpu-cclock", &v)) return false;
            r.ov.tuning.gpuCclockSet = true;
            r.ov.tuning.gpuCclock = v;
        } else if (a == "--gpu-mclock" || a.rfind("--gpu-mclock=", 0) == 0) {
            if (r.ov.tuning.gpuMclockSet) { if (error) *error = "--gpu-mclock given twice"; return false; }
            if (!tuningValue("--gpu-mclock", &v)) return false;
            r.ov.tuning.gpuMclockSet = true;
            r.ov.tuning.gpuMclock = v;
        } else if (a == "--gpu-coffset" || a.rfind("--gpu-coffset=", 0) == 0) {
            if (r.ov.tuning.gpuCoffsetSet) { if (error) *error = "--gpu-coffset given twice"; return false; }
            if (!tuningValue("--gpu-coffset", &v)) return false;
            r.ov.tuning.gpuCoffsetSet = true;
            r.ov.tuning.gpuCoffset = v;
        } else if (a == "--gpu-moffset" || a.rfind("--gpu-moffset=", 0) == 0) {
            if (r.ov.tuning.gpuMoffsetSet) { if (error) *error = "--gpu-moffset given twice"; return false; }
            if (!tuningValue("--gpu-moffset", &v)) return false;
            r.ov.tuning.gpuMoffsetSet = true;
            r.ov.tuning.gpuMoffset = v;
        } else if (a == "--gpu-plimit" || a.rfind("--gpu-plimit=", 0) == 0) {
            if (r.ov.tuning.gpuPlimitSet) { if (error) *error = "--gpu-plimit given twice"; return false; }
            if (!tuningValue("--gpu-plimit", &v)) return false;
            r.ov.tuning.gpuPlimitSet = true;
            r.ov.tuning.gpuPlimit = v;
        } else if (a == "--gpu-fan" || a.rfind("--gpu-fan=", 0) == 0) {
            if (r.ov.tuning.gpuFanSet) { if (error) *error = "--gpu-fan given twice"; return false; }
            if (!tuningValue("--gpu-fan", &v)) return false;
            r.ov.tuning.gpuFanSet = true;
            r.ov.tuning.gpuFan = v;
        } else if (a == "--oc-delay" || a.rfind("--oc-delay=", 0) == 0) {
            if (r.ov.tuning.ocDelaySet) { if (error) *error = "--oc-delay given twice"; return false; }
            if (!tuningValue("--oc-delay", &v)) return false;
            uint64_t seconds = 0;
            for (char c : v) {
                if (c < '0' || c > '9') { if (error) *error = "--oc-delay needs whole seconds (0..4294967295)"; return false; }
                const unsigned digit = static_cast<unsigned>(c - '0');
                if (seconds > (static_cast<uint64_t>(UINT_MAX) - digit) / 10u) {
                    if (error) *error = "--oc-delay needs whole seconds (0..4294967295)";
                    return false;
                }
                seconds = seconds * 10u + digit;
            }
            r.ov.tuning.ocDelaySet = true;
            r.ov.tuning.ocDelaySeconds = seconds;
        } else if (a == "--oc-reset-only") {
            if (r.ov.tuning.ocResetOnly) { if (error) *error = "--oc-reset-only given twice"; return false; }
            r.ov.tuning.ocResetOnly = true;
        } else if (a == "--algorithm") {
            // Accepted for harnesses written for other miners; this build mines Pearl only.
            if (!need(i, "--algorithm", &v)) return false;
            if (v != "pearl" && v != "pearlhash") {
                if (error) *error = "--algorithm supports pearl (or pearlhash, the same algorithm); this build mines Pearl only";
                return false;
            }
        } else if (a == "--disable-cpu") {
            // Accepted and ignored: this miner is GPU only.
        } else if (isOverclockOption(a)) {
            if (error) *error = "unknown GPU tuning flag '" + a + "'";
            return false;
        } else if (a == "--list-devices" && !r.commandExplicit) {
            r.command = Command::ListDevices;
            r.commandExplicit = true;
        } else if (a == "--version" && !r.commandExplicit) {
            r.command = Command::Version;
            r.commandExplicit = true;
        } else {
            if (a == "--help" || a == "-h") r.help = true;
            r.rest.push_back(a);  // --help too: the legacy loop in miner.cu still answers it until it calls productHelpText()
        }
    }
    *out = r;
    return true;
}

/** Account name accepted by F2Pool's login: letters, digits and '_', at most 64 (the same rule as a config pool entry). */
inline bool validAccountName(const std::string &s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
    return true;
}

/** A Pearl payout address (mainnet or testnet prefix); anything else before a dot is treated as an F2Pool account. */
inline bool looksLikePrlAddress(const std::string &s) { return s.rfind("prl1", 0) == 0 || s.rfind("tprl1", 0) == 0; }

/**
 * Layer the CLI over a config (file or defaults). Pool specs are parsed here so errors name the offending value.
 * --wallet ADDR.WORKER: the split happens on the first dot; --worker wins over the split part. On an F2Pool pool, a
 * name that is not a prl1 address is the F2Pool account (the login becomes ACCOUNT.WORKER, as on every other miner).
 */
inline bool applyOverrides(const CliOverrides &ov, ProductConfig *cfg, std::string *error) {
    if (!ov.pools.empty()) {
        std::vector<PoolEntry> pools;
        for (const auto &spec : ov.pools) {
            PoolEntry e;
            if (!parsePoolSpec(spec, &e, error)) return false;
            pools.push_back(e);
        }
        cfg->pools = pools;
    }
    if (ov.walletSet) {
        const size_t dot = ov.wallet.find('.');
        const std::string head = dot == std::string::npos ? ov.wallet : ov.wallet.substr(0, dot);
        if (dot != std::string::npos) {
            const std::string tail = ov.wallet.substr(dot + 1);
            if (head.empty() || tail.empty() || tail.find('.') != std::string::npos) {
                if (error) *error = "--wallet must be ADDRESS or ADDRESS.WORKER (one dot)";
                return false;
            }
            if (!ov.workerSet) cfg->worker = tail;
        }
        const bool f2pool = !cfg->pools.empty() && cfg->pools[0].preset == "f2pool";
        if (!looksLikePrlAddress(head) && f2pool) {
            if (!validAccountName(head)) {
                if (error) *error = "the F2Pool account may contain only letters, digits and '_' (max 64); a Pearl address starts with prl1";
                return false;
            }
            cfg->pools[0].account = head;
            cfg->wallet.clear();
        } else {
            // Any other value is passed through unchanged: miner.cu's own address check (after the policy checks) reports it.
            cfg->wallet = head;
            if (!cfg->pools.empty()) cfg->pools[0].account.clear();
        }
    }
    if (ov.workerSet) cfg->worker = ov.worker;
    if (ov.devicesSet) cfg->devices = ov.devices;
    if (ov.apiEnable) {
        if (ov.apiPortSet && ov.apiPort == 0) {
            if (error) *error = "--api-enable conflicts with --api-port 0";
            return false;
        }
        cfg->api.enabled = true;
        if (!ov.apiPortSet) cfg->api.port = kApiEnablePort;
    }
    if (ov.apiBindSet) cfg->api.bind = ov.apiBind;
    if (ov.apiPortSet) {
        cfg->api.enabled = ov.apiPort != 0;
        if (ov.apiPort != 0) cfg->api.port = ov.apiPort;
    }
    if (ov.logDirSet) cfg->log.dir = ov.logDir;
    if (ov.noDonation) cfg->donation = false;  // the flag only ever turns it off; it wins over "donation": true
    if (ov.tuning.gpuCclockSet) { cfg->tuning.gpuCclockSet = true; cfg->tuning.gpuCclock = ov.tuning.gpuCclock; }
    if (ov.tuning.gpuMclockSet) { cfg->tuning.gpuMclockSet = true; cfg->tuning.gpuMclock = ov.tuning.gpuMclock; }
    if (ov.tuning.gpuCoffsetSet) { cfg->tuning.gpuCoffsetSet = true; cfg->tuning.gpuCoffset = ov.tuning.gpuCoffset; }
    if (ov.tuning.gpuMoffsetSet) { cfg->tuning.gpuMoffsetSet = true; cfg->tuning.gpuMoffset = ov.tuning.gpuMoffset; }
    if (ov.tuning.gpuPlimitSet) { cfg->tuning.gpuPlimitSet = true; cfg->tuning.gpuPlimit = ov.tuning.gpuPlimit; }
    if (ov.tuning.gpuFanSet) { cfg->tuning.gpuFanSet = true; cfg->tuning.gpuFan = ov.tuning.gpuFan; }
    if (ov.tuning.ocDelaySet) { cfg->tuning.ocDelaySet = true; cfg->tuning.ocDelaySeconds = ov.tuning.ocDelaySeconds; }
    if (ov.tuning.ocResetOnly) cfg->tuning.ocResetOnly = true;
    return true;
}

/**
 * The grouped help text. Names no developer account and no payout address: the donation is described only as
 * "mined on F2Pool to the developer".
 */
inline std::string productHelpText() {
    return
        "CanaryMiner: GPU miner for Pearl (PRL), NVIDIA Blackwell (RTX 50-series) only\n"
        "\n"
        "usage: canary-miner [run] [options]    mine (no subcommand needed)\n"
        "       canary-miner list-devices       GPU identity and whether it is supported\n"
        "       canary-miner selftest           offline kernel + proof self-test per device\n"
        "       canary-miner status [--json]    read the stats endpoint of a running miner\n"
        "       canary-miner version\n"
        "\n"
        "Pool\n"
        "  -o, --pool SPEC       HOST:PORT, stratum+tcp://HOST:PORT, or a preset: f2pool, kryptex,\n"
        "                        herominers. Known pool hosts are recognised automatically.\n"
        "                        stratum+ssl:// (TLS) is not supported yet.\n"
        "  -u, --wallet ADDR     your PRL payout address (prl1...). Add .WORKER to name the rig:\n"
        "                        ADDR.rig1 (the part after the first dot is the worker).\n"
        "                        F2Pool needs ACCOUNT.WORKER: your F2Pool account name, not a prl1 address.\n"
        "      --worker NAME     rig name; overrides the WORKER part of --wallet (default canary)\n"
        "  -p, --password P      pool password field (default x)\n"
        "\n"
        "Devices\n"
         "      --gpu-id LIST     GPUs to use: all | index | GPU-uuid | PCI bus id, comma separated\n"
         "                        (alias --devices)\n"
         "\n"
         "GPU tuning: not supported in this release. These flags are accepted and ignored with a notice:\n"
         "      --gpu-cclock --gpu-mclock --gpu-coffset --gpu-moffset --gpu-plimit --gpu-fan --oc-delay --oc-reset-only\n"
         "                        Config keys tuning.gpu_cclock and the rest are ignored the same way.\n"
         "                        Set clocks, power and fans with nvidia-smi, MSI Afterburner or a similar tool.\n"
         "\n"
        "API (stats endpoint, loopback only)\n"
        "      --api-enable      turn the stats endpoint on (port 21550 unless --api-port is given)\n"
        "      --api-port N      stats endpoint port (default 8085 from the config; 0 disables)\n"
        "      --api-bind ADDR   address to listen on (default 127.0.0.1; loopback only)\n"
        "\n"
        "Donation\n"
        "      --no-donation     keep the 1% developer donation off (the default in this build).\n"
        "                        Opt in with the config key \"donation\": true. The flag wins over the config.\n"
        "\n"
        "Other\n"
        "      --config FILE     config-v1 JSON file; command-line options override it\n"
        "      --log-dir DIR     rotating redacted log files\n"
        "      --json            (status) raw JSON\n"
        "  -h, --help            this text\n"
        "      --help-all        every option, including research and benchmark modes\n"
        "  Accepted and ignored: --algorithm pearl, --disable-cpu.\n"
        "\n"
        "Examples\n"
        "  Windows:  canary-miner.exe --pool stratum+tcp://prl-us.kryptex.network:7048 --wallet YOUR_PRL_ADDRESS.rig1\n"
        "  F2Pool:   canary-miner.exe --pool f2pool --wallet YOUR_F2POOL_ACCOUNT.rig1\n"
        "  Linux:    ./canary-miner --pool stratum+tcp://prl-us.kryptex.network:7048 --wallet YOUR_PRL_ADDRESS.rig1\n";
}

}  // namespace om
