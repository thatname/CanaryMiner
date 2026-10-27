// Device identity and the allowlist.
//
// The allowlist is the Blackwell (SM120) SKU list: RTX 5070, RTX 5080, matched on the exact CUDA device name AND the
// compute capability. Anything else (an Ada or Ampere card, a 5090, an unknown laptop part) is refused with a reason
// before any kernel is loaded. Entries are "observed-private", not "qualified"; widening
// the list is a code change that needs qualification evidence, never a config option.
//
// This header is pure host logic (no CUDA headers): the caller fills DeviceIdentity from cudaDeviceProp (the runtime API
// is a thin layer over the driver API and reports the same UUID, PCI location and compute capability).

#pragma once

#include <stdint.h>

#include <string>
#include <vector>

namespace om {

struct DeviceIdentity {
    int index = -1;
    std::string name;
    std::string uuid;    // "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    std::string pciBus;  // "dddd:bb:dd.0" (domain:bus:device.function, lower case)
    int ccMajor = 0;
    int ccMinor = 0;
    uint64_t memBytes = 0;
};

struct DeviceSku {
    const char *name;  // exact CUDA device name
    int ccMajor;
    int ccMinor;
    const char *family;  // "SM120"
};

inline const DeviceSku *deviceAllowlist(size_t *count) {
    static const DeviceSku kSkus[] = {
#if defined(CANARY_PEARL_AMPERE_BUILD)
        {"NVIDIA GeForce RTX 3080", 8, 6, "SM86"},
#else
        {"NVIDIA GeForce RTX 5070", 12, 0, "SM120"},
        {"NVIDIA GeForce RTX 5080", 12, 0, "SM120"},
#endif
    };
    *count = sizeof(kSkus) / sizeof(kSkus[0]);
    return kSkus;
}

struct DeviceVerdict {
    bool allowed = false;
    std::string family;  // "SM120" when allowed
    std::string reason;  // empty when allowed
};

/** Decide whether `d` may run. `compiledCc` is the native's architecture (120), or 0 to skip that check. */
inline DeviceVerdict checkDevice(const DeviceIdentity &d, int compiledCc = 0) {
    DeviceVerdict v;
    size_t n = 0;
    const DeviceSku *skus = deviceAllowlist(&n);
    const DeviceSku *hit = nullptr;
    for (size_t i = 0; i < n; ++i)
        if (d.name == skus[i].name) hit = &skus[i];
    if (!hit) {
        v.reason = "CanaryMiner supports NVIDIA Blackwell (RTX 50-series, SM120) only; '" + d.name + "' (compute capability " + std::to_string(d.ccMajor) + "." +
                   std::to_string(d.ccMinor) + ") is refused";
        return v;
    }
    if (d.ccMajor != hit->ccMajor || d.ccMinor != hit->ccMinor) {
        v.reason = "'" + d.name + "' reports compute capability " + std::to_string(d.ccMajor) + "." +
                   std::to_string(d.ccMinor) + ", expected " + std::to_string(hit->ccMajor) + "." +
                   std::to_string(hit->ccMinor);
        return v;
    }
    if (d.uuid.empty() || d.uuid.compare(0, 4, "GPU-") != 0) {
        v.reason = "device UUID is missing or malformed";
        return v;
    }
    if (compiledCc != 0 && compiledCc != hit->ccMajor * 10 + hit->ccMinor) {
        v.reason = std::string("this build contains SM") + std::to_string(compiledCc) + " kernels; '" + d.name + "' needs the " +
                   hit->family + " build";
        return v;
    }
    v.allowed = true;
    v.family = hit->family;
    return v;
}

/** One entry of --devices / config "devices": "all", an index, a UUID ("GPU-..."), or a PCI bus id. */
inline bool deviceSpecMatches(const std::string &spec, const DeviceIdentity &d) {
    if (spec.empty()) return false;
    if (spec.compare(0, 4, "GPU-") == 0) return spec == d.uuid;
    bool digits = true;
    for (char c : spec) digits = digits && c >= '0' && c <= '9';
    if (digits && spec.size() <= 4) return std::to_string(d.index) == spec;
    return spec == d.pciBus;
}

struct DeviceSelection {
    std::vector<DeviceIdentity> chosen;
    std::vector<std::string> refusals;  // one line per refused or unmatched request
};

/**
 * Resolve the requested specs against what the machine has.
 *   - "all" (or an empty list) selects every allowlisted device and silently skips the others (they are listed in
 *     `refusals` so callers can print them);
 *   - an explicit spec that matches nothing, or matches a device that is not allowed, is a refusal.
 * Returns true only when at least one device was chosen and no explicit request was refused.
 */
inline bool selectDevices(const std::vector<std::string> &specs, const std::vector<DeviceIdentity> &present, int compiledCc,
                          DeviceSelection *out) {
    out->chosen.clear();
    out->refusals.clear();
    bool explicitRefusal = false;
    const bool all = specs.empty() || (specs.size() == 1 && specs[0] == "all");
    if (all) {
        for (const auto &d : present) {
            const DeviceVerdict v = checkDevice(d, compiledCc);
            if (v.allowed) out->chosen.push_back(d);
            else out->refusals.push_back("device " + std::to_string(d.index) + ": " + v.reason);
        }
    } else {
        for (const auto &spec : specs) {
            bool matched = false;
            for (const auto &d : present) {
                if (!deviceSpecMatches(spec, d)) continue;
                matched = true;
                const DeviceVerdict v = checkDevice(d, compiledCc);
                if (!v.allowed) {
                    out->refusals.push_back("device " + std::to_string(d.index) + ": " + v.reason);
                    explicitRefusal = true;
                    continue;
                }
                bool dup = false;
                for (const auto &c : out->chosen) dup = dup || c.index == d.index;
                if (!dup) out->chosen.push_back(d);
            }
            if (!matched) {
                out->refusals.push_back("no device matches '" + spec + "'");
                explicitRefusal = true;
            }
        }
    }
    if (out->chosen.empty()) {
        if (out->refusals.empty()) out->refusals.push_back("no CUDA device found");
        return false;
    }
    return !explicitRefusal;
}

}  // namespace om
