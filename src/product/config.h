// Versioned product configuration (config-v1).
//
// A config file is one JSON object. `schema_version` is required and must be 1; any other value is refused with a
// migration message (this build never guesses at a newer or older layout). Unknown keys are errors, at every level:
// a misspelled key must not silently fall back to a default. CLI options override the file (see product_cli.h).
// config-v1.schema.json documents the same layout for editors and the Python schema test.
//
//   {
//     "schema_version": 1,
//     "wallet": "<PRL payout address>",
//     "worker": "rig_1",
//     "pools": ["kryptex", {"url": "pool.example.net:3333", "network": "mainnet", "target_semantics": "base-before-penalty"}],
//     "devices": ["all"],
//     "api":      {"enabled": true, "bind": "127.0.0.1", "port": 8085},
//     "log":      {"dir": "logs", "max_file_bytes": 10485760, "max_files": 5, "redact_worker": false},
//     "state_dir": "state",
//     "shutdown": {"drain_seconds": 10}
//   }

#pragma once

#include <stdint.h>

#include <cctype>
#include <climits>
#include <initializer_list>
#include <string>
#include <vector>

#include "json_dom.h"
#include "redact.h"

namespace om {

constexpr unsigned kConfigSchemaVersion = 1;

struct PoolEntry {
    std::string preset;  // "kryptex", "herominers" or empty for a custom endpoint
    std::string host;
    int port = 0;
    std::string network = "mainnet";
    std::string targetSemantics = "base-before-penalty";
    // H-019 task 3 (H-002): "auto" offers the gzip v2 dialect at login and uses it only when the pool answers type v2;
    // "off" never sends the offer. `account` is the v2 login (empty: the payout address).
    std::string compression = "auto";
    std::string account;
    // true when network/target_semantics come from a preset or from the config file (the user wrote that entry); false for a
    // bare HOST:PORT on the command line, where the policy must still be given explicitly (--pearl-network/--pearl-target-semantics)
    bool policyKnown = false;

    std::string endpoint() const { return host + ":" + std::to_string(port); }
};

struct ApiConfig {
    bool enabled = true;
    std::string bind = "127.0.0.1";
    int port = 8085;
};

struct LogConfig {
    std::string dir;  // empty: no file sink
    uint64_t maxFileBytes = 10ull * 1024 * 1024;
    unsigned maxFiles = 5;
    bool redactWorker = false;
};

/** Raw GPU tuning values. Lists use comma-separated values in selected-device order; syntax/ranges are checked at runtime. */
struct GpuTuningConfig {
    bool gpuCclockSet = false;
    std::string gpuCclock;
    bool gpuMclockSet = false;
    std::string gpuMclock;
    bool gpuCoffsetSet = false;
    std::string gpuCoffset;
    bool gpuMoffsetSet = false;
    std::string gpuMoffset;
    bool gpuPlimitSet = false;
    std::string gpuPlimit;
    bool gpuFanSet = false;
    std::string gpuFan;
    bool ocDelaySet = false;
    uint64_t ocDelaySeconds = 0;
    bool ocResetOnly = false;

    bool requested() const {
        return gpuCclockSet || gpuMclockSet || gpuCoffsetSet || gpuMoffsetSet || gpuPlimitSet || gpuFanSet || ocDelaySet || ocResetOnly;
    }
};

struct ProductConfig {
    unsigned schemaVersion = kConfigSchemaVersion;
    std::string wallet;
    std::string worker = "canary";
    std::vector<PoolEntry> pools;
    std::vector<std::string> devices;  // empty = every supported device
    ApiConfig api;
    LogConfig log;
    std::string stateDir = "state";
    unsigned drainSeconds = 10;
    bool donation = false;  // developer donation (1%, 99:1); off by default in this fork, opt in with "donation": true
    GpuTuningConfig tuning;
};

struct PoolPreset {
    const char *name;
    const char *host;
    int port;
};

/** Presets whose pool speaks the v2 gzip dialect. A custom HOST:PORT never offers it. */
inline bool poolPresetSpeaksV2(const std::string &name) {
    return name == "kryptex" || name == "f2pool";
}

inline const PoolPreset *poolPresets(size_t *count) {
    static const PoolPreset kPresets[] = {
        {"kryptex", "prl-us.kryptex.network", 7048},
        {"herominers", "us2.pearl.herominers.com", 1200},
        {"f2pool", "pearl.f2pool.com", 5500},
    };
    *count = sizeof(kPresets) / sizeof(kPresets[0]);
    return kPresets;
}

/** "HOST:PORT" -> parts. Rejects empty hosts, a leading '-', non-numeric or out-of-range ports, and whitespace. */
inline bool parseEndpoint(const std::string &text, std::string *host, int *port) {
    const size_t c = text.rfind(':');
    if (c == std::string::npos || c == 0 || c + 1 >= text.size()) return false;
    const std::string h = text.substr(0, c);
    if (h[0] == '-') return false;
    for (char ch : h)
        if (ch <= ' ' || ch == '/' || ch == '@') return false;
    long p = 0;
    for (size_t i = c + 1; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') return false;
        p = p * 10 + (text[i] - '0');
        if (p > 65535) return false;
    }
    if (p < 1) return false;
    *host = h;
    *port = static_cast<int>(p);
    return true;
}

/** True when `host` is `domain` or a subdomain of it ("us2.pearl.herominers.com" matches "herominers.com"). */
inline bool hostInDomain(const std::string &host, const char *domain) {
    const std::string d = domain;
    if (host.size() < d.size()) return false;
    if (host.compare(host.size() - d.size(), d.size(), d) != 0) return false;
    return host.size() == d.size() || host[host.size() - d.size() - 1] == '.';
}

/**
 * A pool given as a preset name, HOST:PORT, or stratum+tcp://HOST:PORT (CLI --pool and string entries in the config).
 * A host of a known pool gets that pool's preset name, so its quirks apply (v2 gzip login for kryptex and f2pool). The
 * endpoint stays the one the user typed. stratum+ssl:// is refused until TLS exists.
 */
inline bool parsePoolSpec(const std::string &spec, PoolEntry *out, std::string *error) {
    size_t n = 0;
    const PoolPreset *presets = poolPresets(&n);
    PoolEntry e;
    for (size_t i = 0; i < n; ++i) {
        if (spec == presets[i].name) {
            e.preset = presets[i].name;
            e.host = presets[i].host;
            e.port = presets[i].port;
            e.policyKnown = true;
            *out = e;
            return true;
        }
    }
    std::string endpoint = spec;
    if (endpoint.rfind("stratum+ssl://", 0) == 0) {
        if (error) *error = "TLS not supported yet: stratum+ssl:// needs TLS, which this build does not have; use stratum+tcp:// or HOST:PORT";
        return false;
    }
    if (endpoint.rfind("stratum+tcp://", 0) == 0) endpoint = endpoint.substr(std::string("stratum+tcp://").size());
    if (!parseEndpoint(endpoint, &e.host, &e.port)) {
        if (error) *error = "pool '" + spec + "' is neither a preset (kryptex, herominers, f2pool), HOST:PORT, nor stratum+tcp://HOST:PORT";
        return false;
    }
    if (hostInDomain(e.host, "kryptex.network")) e.preset = "kryptex";
    else if (hostInDomain(e.host, "herominers.com")) e.preset = "herominers";
    else if (hostInDomain(e.host, "f2pool.com")) e.preset = "f2pool";
    e.policyKnown = !e.preset.empty();
    *out = e;
    return true;
}

inline bool validNetwork(const std::string &s) { return s == "mainnet"; }  // other networks are unsupported live

inline bool validSemantics(const std::string &s) { return s == "base-before-penalty" || s == "final-adjusted-bound"; }

namespace config_detail {

inline bool fail(std::string *error, const std::string &what) {
    if (error) *error = what;
    return false;
}

inline bool onlyKeys(const JsonValue &obj, const char *where, std::initializer_list<const char *> allowed, std::string *error) {
    for (const auto &m : obj.members) {
        bool ok = false;
        for (const char *a : allowed) ok = ok || m.first == a;
        if (!ok) return fail(error, std::string("unknown key '") + m.first + "' in " + where);
    }
    return true;
}

inline bool getString(const JsonValue &obj, const char *key, const char *where, std::string *out, std::string *error) {
    const JsonValue *v = obj.find(key);
    if (!v) return true;
    if (!v->isString()) return fail(error, std::string("'") + key + "' in " + where + " must be a string");
    *out = v->text;
    return true;
}

inline bool getBool(const JsonValue &obj, const char *key, const char *where, bool *out, std::string *error) {
    const JsonValue *v = obj.find(key);
    if (!v) return true;
    if (!v->isBool()) return fail(error, std::string("'") + key + "' in " + where + " must be true or false");
    *out = v->boolean;
    return true;
}

inline bool getUint(const JsonValue &obj, const char *key, const char *where, uint64_t lo, uint64_t hi, uint64_t *out,
                    std::string *error) {
    const JsonValue *v = obj.find(key);
    if (!v) return true;
    uint64_t n = 0;
    if (!v->asUint64(&n)) return fail(error, std::string("'") + key + "' in " + where + " must be a whole number");
    if (n < lo || n > hi)
        return fail(error, std::string("'") + key + "' in " + where + " must be between " + std::to_string(lo) + " and " +
                               std::to_string(hi));
    *out = n;
    return true;
}

}  // namespace config_detail

/** Parse and validate config text. On failure *error holds one clear line and *cfg is untouched. */
inline bool parseConfig(const std::string &text, ProductConfig *cfg, std::string *error) {
    using namespace config_detail;
    JsonValue root;
    std::string jerr;
    if (!parseJsonStrict(text, &root, &jerr)) return fail(error, "config is not valid JSON: " + jerr);
    if (!root.isObject()) return fail(error, "config must be a JSON object");
    const JsonValue *sv = root.find("schema_version");
    if (!sv) return fail(error, "config has no schema_version; add \"schema_version\": 1 (config-v1)");
    uint64_t version = 0;
    if (!sv->asUint64(&version))
        return fail(error, "schema_version must be a whole number; this build reads config-v1 (schema_version 1)");
    if (version != kConfigSchemaVersion)
        return fail(error, "config schema_version " + std::to_string(version) + " is not supported; this build reads schema_version " +
                               std::to_string(kConfigSchemaVersion) +
                               " only. There is no automatic migration: edit the file to the config-v1 layout (see config.example.json)");
    if (!onlyKeys(root, "the config", {"schema_version", "wallet", "worker", "pools", "devices", "api", "log", "state_dir", "shutdown", "donation", "tuning"}, error))
        return false;

    ProductConfig out;
    if (!getString(root, "wallet", "the config", &out.wallet, error)) return false;
    if (!getString(root, "worker", "the config", &out.worker, error)) return false;
    if (!getString(root, "state_dir", "the config", &out.stateDir, error)) return false;
    if (!getBool(root, "donation", "the config", &out.donation, error)) return false;

    if (const JsonValue *tuning = root.find("tuning")) {
        if (!tuning->isObject()) return fail(error, "'tuning' must be an object");
        if (!onlyKeys(*tuning, "tuning", {"gpu_cclock", "gpu_mclock", "gpu_coffset", "gpu_moffset", "gpu_plimit", "gpu_fan", "oc_delay", "oc_reset_only"}, error))
            return false;
        out.tuning.gpuCclockSet = tuning->find("gpu_cclock") != nullptr;
        out.tuning.gpuMclockSet = tuning->find("gpu_mclock") != nullptr;
        out.tuning.gpuCoffsetSet = tuning->find("gpu_coffset") != nullptr;
        out.tuning.gpuMoffsetSet = tuning->find("gpu_moffset") != nullptr;
        out.tuning.gpuPlimitSet = tuning->find("gpu_plimit") != nullptr;
        out.tuning.gpuFanSet = tuning->find("gpu_fan") != nullptr;
        if (!getString(*tuning, "gpu_cclock", "tuning", &out.tuning.gpuCclock, error) ||
            !getString(*tuning, "gpu_mclock", "tuning", &out.tuning.gpuMclock, error) ||
            !getString(*tuning, "gpu_coffset", "tuning", &out.tuning.gpuCoffset, error) ||
            !getString(*tuning, "gpu_moffset", "tuning", &out.tuning.gpuMoffset, error) ||
            !getString(*tuning, "gpu_plimit", "tuning", &out.tuning.gpuPlimit, error) ||
            !getString(*tuning, "gpu_fan", "tuning", &out.tuning.gpuFan, error))
            return false;
        uint64_t delay = 0;
        if (!getUint(*tuning, "oc_delay", "tuning", 0, UINT_MAX, &delay, error) ||
            !getBool(*tuning, "oc_reset_only", "tuning", &out.tuning.ocResetOnly, error))
            return false;
        out.tuning.ocDelaySet = tuning->find("oc_delay") != nullptr;
        out.tuning.ocDelaySeconds = delay;
    }

    if (const JsonValue *pools = root.find("pools")) {
        if (!pools->isArray() || pools->items.empty()) return fail(error, "'pools' must be a non-empty array");
        if (pools->items.size() > 8) return fail(error, "'pools' lists more than 8 entries");
        for (const JsonValue &p : pools->items) {
            PoolEntry e;
            std::string perr;
            if (p.isString()) {
                if (!parsePoolSpec(p.text, &e, &perr)) return fail(error, perr);
            } else if (p.isObject()) {
                if (!onlyKeys(p, "a pool entry", {"url", "network", "target_semantics", "compression", "account"}, error)) return false;
                std::string url, network = "mainnet", semantics = "base-before-penalty";
                if (!getString(p, "url", "a pool entry", &url, error)) return false;
                if (url.empty()) return fail(error, "a pool entry needs a 'url' (preset name or HOST:PORT)");
                if (!parsePoolSpec(url, &e, &perr)) return fail(error, perr);
                if (!getString(p, "network", "a pool entry", &network, error)) return false;
                if (!getString(p, "target_semantics", "a pool entry", &semantics, error)) return false;
                std::string compression = "auto", account;
                if (!getString(p, "compression", "a pool entry", &compression, error)) return false;
                if (compression != "auto" && compression != "off")
                    return fail(error, "compression '" + compression + "' must be auto or off");
                if (!getString(p, "account", "a pool entry", &account, error)) return false;
                if (account.size() > 64)
                    return fail(error, "account is longer than 64 characters");
                for (char c : account)
                    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
                        return fail(error, "account may contain only letters, digits and '_'");
                e.network = network;
                e.targetSemantics = semantics;
                e.compression = compression;
                e.account = account;
            } else {
                return fail(error, "each pool must be a preset name, HOST:PORT, or an object");
            }
            e.policyKnown = true;
            if (!validNetwork(e.network))
                return fail(error, "network '" + e.network + "' is not supported; live mining supports mainnet only");
            if (!validSemantics(e.targetSemantics))
                return fail(error, "target_semantics '" + e.targetSemantics + "' must be base-before-penalty or final-adjusted-bound");
            out.pools.push_back(e);
        }
    }

    if (const JsonValue *dev = root.find("devices")) {
        if (!dev->isArray()) return fail(error, "'devices' must be an array");
        if (dev->items.size() > 16) return fail(error, "'devices' lists more than 16 entries");
        for (const JsonValue &d : dev->items) {
            if (d.isString()) out.devices.push_back(d.text);
            else if (d.isNumber()) {
                uint64_t i = 0;
                if (!d.asUint64(&i) || i > 15) return fail(error, "a device index must be a whole number between 0 and 15");
                out.devices.push_back(std::to_string(i));
            } else {
                return fail(error, "each device must be an index, a UUID/PCI string, or \"all\"");
            }
        }
    }

    if (const JsonValue *api = root.find("api")) {
        if (!api->isObject()) return fail(error, "'api' must be an object");
        if (!onlyKeys(*api, "api", {"enabled", "bind", "port"}, error)) return false;
        uint64_t port = static_cast<uint64_t>(out.api.port);
        if (!getBool(*api, "enabled", "api", &out.api.enabled, error) || !getString(*api, "bind", "api", &out.api.bind, error) ||
            !getUint(*api, "port", "api", 1, 65535, &port, error))
            return false;
        out.api.port = static_cast<int>(port);
    }

    if (const JsonValue *log = root.find("log")) {
        if (!log->isObject()) return fail(error, "'log' must be an object");
        if (!onlyKeys(*log, "log", {"dir", "max_file_bytes", "max_files", "redact_worker"}, error)) return false;
        uint64_t maxBytes = out.log.maxFileBytes, maxFiles = out.log.maxFiles;
        if (!getString(*log, "dir", "log", &out.log.dir, error) || !getUint(*log, "max_file_bytes", "log", 4096, 1ull << 32, &maxBytes, error) ||
            !getUint(*log, "max_files", "log", 1, 100, &maxFiles, error) || !getBool(*log, "redact_worker", "log", &out.log.redactWorker, error))
            return false;
        out.log.maxFileBytes = maxBytes;
        out.log.maxFiles = static_cast<unsigned>(maxFiles);
    }

    if (const JsonValue *sd = root.find("shutdown")) {
        if (!sd->isObject()) return fail(error, "'shutdown' must be an object");
        if (!onlyKeys(*sd, "shutdown", {"drain_seconds"}, error)) return false;
        uint64_t drain = out.drainSeconds;
        if (!getUint(*sd, "drain_seconds", "shutdown", 1, 600, &drain, error)) return false;
        out.drainSeconds = static_cast<unsigned>(drain);
    }
    *cfg = out;
    return true;
}

/** Cross-field checks that apply to the final (file + CLI) configuration. `forRun` demands what mining needs. */
/** F2Pool logs in with the F2Pool account name only; the PRL payout address is set inside the F2Pool account. */
inline bool checkPoolLogin(const ProductConfig &cfg, std::string *error) {
    if (!cfg.pools.empty() && cfg.pools[0].preset == "f2pool" && cfg.pools[0].account.empty())
        return config_detail::fail(error, "F2Pool needs your F2Pool account name, not a PRL address: use --wallet YOUR_F2POOL_ACCOUNT.rig1 "
                                          "(set your PRL payout address in your F2Pool account settings), or use another pool such as kryptex");
    return true;
}

inline bool validateConfig(const ProductConfig &cfg, bool forRun, std::string *error) {
    using config_detail::fail;
    if (cfg.worker.empty() || cfg.worker.size() > 64) return fail(error, "worker must be 1 to 64 characters");
    for (char c : cfg.worker)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
            return fail(error, "worker may contain letters, digits, '_', '-' and '.' only");
    if (cfg.api.bind != "127.0.0.1" && cfg.api.bind != "::1" && cfg.api.bind != "localhost")
        return fail(error, "api.bind must be a loopback address (127.0.0.1, ::1 or localhost): the stats endpoint is local only");
    if (cfg.drainSeconds < 1 || cfg.drainSeconds > 600) return fail(error, "shutdown.drain_seconds must be 1..600");
    if (forRun) {
        if (!checkPoolLogin(cfg, error)) return false;
        if (cfg.wallet.empty()) return fail(error, "no wallet: set \"wallet\" in the config or pass --wallet");
        if (cfg.pools.empty()) return fail(error, "no pool: set \"pools\" in the config or pass --pool (kryptex, herominers, f2pool or HOST:PORT)");
    }
    return true;
}

/** The effective configuration with secrets hidden, one "key: value" per line, for the startup banner and `status`. */
inline std::string effectiveConfigText(const ProductConfig &cfg, const std::string &source) {
    std::string t;
    t += "effective configuration (" + source + ")\n";
    t += "  schema_version: " + std::to_string(cfg.schemaVersion) + "\n";
    t += "  wallet:   " + (cfg.wallet.empty() ? std::string("<unset>") : "prl1..." + Redactor::tail4(cfg.wallet)) + "\n";
    t += "  worker:   " + (cfg.log.redactWorker ? std::string("<redacted>") : cfg.worker) + "\n";
    for (size_t i = 0; i < cfg.pools.size(); ++i) {
        const PoolEntry &p = cfg.pools[i];
        t += "  pool " + std::to_string(i + 1) + ":   " + p.endpoint() + (p.preset.empty() ? "" : " (" + p.preset + ")") + " network=" +
             p.network + " target_semantics=" + p.targetSemantics + "\n";
    }
    if (cfg.pools.size() > 1) t += "  note:     failover is not implemented in this build; only pool 1 is used\n";
    t += "  devices:  ";
    if (cfg.devices.empty()) t += "all supported";
    for (size_t i = 0; i < cfg.devices.size(); ++i) t += (i ? ", " : "") + cfg.devices[i];
    t += "\n";
    t += std::string("  api:      ") + (cfg.api.enabled ? cfg.api.bind + ":" + std::to_string(cfg.api.port) : "disabled") + "\n";
    t += "  log:      " + (cfg.log.dir.empty() ? std::string("console only") : cfg.log.dir + " (" + std::to_string(cfg.log.maxFiles) + " x " +
                                                                                  std::to_string(cfg.log.maxFileBytes) + " bytes)") + "\n";
    t += "  state:    " + cfg.stateDir + "\n";
    t += "  drain:    " + std::to_string(cfg.drainSeconds) + " s\n";
    t += std::string("  donation: ") + (cfg.donation ? "on (1%)" : "off") + "\n";
    return t;
}

}  // namespace om
