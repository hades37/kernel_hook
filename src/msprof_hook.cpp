/**
 * msprof_hook - LD_PRELOAD hook for CANN msprof report interfaces.
 *
 * It intercepts the host-side profiling report entry points exported by libprofapi.so
 * and records operator name / type / task / tensor parameters into CSV files.
 *
 * Hooked symbols:
 *   MsprofGetHashId / MsprofStr2Id          -> build hash -> string dictionary
 *   MsprofReportApi                          -> NODE.launch (op name hash + time range)
 *   MsprofReportEvent                        -> events
 *   MsprofReportCompactInfo                  -> NODE.node_basic_info (op name/type/blockDim/...)
 *   MsprofReportAdditionalInfo               -> NODE.tensor_info (input/output tensors)
 *   MsprofReportBatchAdditionalInfo          -> batch of the above
 *   MsprofFinalize                           -> flush CSV
 *
 * Env:
 *   MSPROF_HOOK_OUT=<dir>   output directory (default: ./msprof_hook_out)
 *   MSPROF_HOOK_VERBOSE=1   also print decoded records to stderr
 */

#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "profiling/aprof_pub.h"

namespace {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
constexpr uint16_t kMagic = 0x5A5AU;

std::string Hex(uint64_t v)
{
    char buf[32] = {0};
    (void)snprintf(buf, sizeof(buf), "0x%" PRIx64, v);
    return buf;
}

std::string CsvEscape(const std::string &s)
{
    if (s.find_first_of(",\"\n\r") == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

const char *FormatName(uint32_t format)
{
    switch (format & 0xFFU) {
        case 0: return "NCHW";
        case 1: return "NHWC";
        case 2: return "ND";
        case 3: return "NC1HWC0";
        case 4: return "FRACTAL_Z";
        case 5: return "NC1HWC0_C04";
        case 6: return "HWCN";
        case 7: return "NDHWC";
        case 8: return "NCDHW";
        case 9: return "DHWCN";
        case 10: return "NDC1HWC0";
        case 11: return "FRACTAL_Z_3D";
        case 12: return "NC";
        case 13: return "NCL";
        case 14: return "NCHW_ALIGN";
        case 29: return "FRACTAL_NZ";
        default: return "UNKNOWN";
    }
}

const char *DataTypeName(uint32_t dtype)
{
    switch (dtype) {
        case 0: return "FLOAT";
        case 1: return "FLOAT16";
        case 2: return "INT8";
        case 3: return "INT32";
        case 4: return "UINT8";
        case 5: return "INT16";
        case 6: return "UINT16";
        case 7: return "UINT32";
        case 8: return "INT64";
        case 9: return "UINT64";
        case 10: return "DOUBLE";
        case 11: return "BOOL";
        case 12: return "STRING";
        case 16: return "BF16";
        case 27: return "INT4";
        case 29: return "UINT1";
        default: return "UNKNOWN";
    }
}

const char *TaskTypeName(uint32_t t)
{
    switch (t) {
        case 0: return "AI_CORE";
        case 1: return "AI_CPU";
        case 2: return "AI_VECTOR_CORE";
        case 3: return "WRITE_BACK";
        case 4: return "MIX_AIC";
        case 5: return "MIX_AIV";
        case 6: return "FFTS_PLUS";
        case 7: return "DSA";
        case 8: return "DVPP";
        case 9: return "HCCL";
        case 10: return "FUSION";
        default: return "UNKNOWN";
    }
}

std::string ShapeToString(const uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN])
{
    std::string s;
    for (uint32_t i = 0; i < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++i) {
        if (shape[i] == 0) {
            continue;
        }
        if (!s.empty()) {
            s += ",";
        }
        s += std::to_string(shape[i]);
    }
    return s.empty() ? "scalar" : s;
}

// ---------------------------------------------------------------------------
// data model
// ---------------------------------------------------------------------------
struct TensorDesc {
    uint32_t tensorType = 0;  // 0: input, 1: output
    uint32_t format = 0;
    uint32_t dataType = 0;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN] = {0};
};

struct OpKey {
    uint32_t tid = 0;
    uint64_t opHash = 0;
    uint64_t timestamp = 0;
    OpKey() = default;
    OpKey(uint32_t t, uint64_t h, uint64_t ts) : tid(t), opHash(h), timestamp(ts) {}
    bool operator<(const OpKey &o) const
    {
        return std::tie(tid, opHash, timestamp) < std::tie(o.tid, o.opHash, o.timestamp);
    }
};

struct OpRecord {
    uint64_t opHash = 0;
    uint64_t opTypeHash = 0;
    uint32_t taskType = 0;
    uint32_t blockDim = 0;
    uint32_t opFlag = 0;
    bool hasBasic = false;
    std::vector<TensorDesc> tensors;
};

struct LaunchRecord {
    uint32_t tid = 0;
    uint64_t opHash = 0;
    uint64_t beginTime = 0;
    uint64_t endTime = 0;
};

struct HookState {
    std::mutex mtx;
    std::unordered_map<uint64_t, std::string> hash2str;
    std::unordered_set<uint64_t> rawTargetHashes;
    std::map<OpKey, OpRecord> ops;
    std::vector<LaunchRecord> launches;
    std::string outDir;
    std::string rawTargetOp;
    FILE *rawReports = nullptr;
    uint64_t rawTargetHash = 0;
    uint64_t rawEventSeq = 0;
    bool verbose = false;
    bool rawAuditInitialized = false;
    bool rawAuditEnabled = false;
    bool rawAuditFailed = false;
    bool rawTargetHashSet = false;
    bool inited = false;
    bool flushed = false;
};

HookState &State()
{
    // MsprofHookFini runs during DSO teardown, after ordinary C++ static
    // destructors may already have run. Keep the state alive until process
    // reclamation so the final flush never touches destroyed containers.
    static HookState *state = new HookState();
    return *state;
}

void Mkdirs(const std::string &path)
{
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur == "/" || cur.empty()) {
                continue;
            }
            (void)mkdir(cur.c_str(), 0755);
        }
    }
}

void EnsureInit(HookState &st)
{
    if (st.inited) {
        return;
    }
    st.inited = true;
    const char *out = std::getenv("MSPROF_HOOK_OUT");
    st.outDir = (out != nullptr) ? out : "./msprof_hook_out";
    st.verbose = (std::getenv("MSPROF_HOOK_VERBOSE") != nullptr);
    Mkdirs(st.outDir);
    (void)fprintf(stderr, "[msprof_hook] enabled, output dir = %s\n", st.outDir.c_str());
}

// ---------------------------------------------------------------------------
// opt-in raw report audit
// ---------------------------------------------------------------------------
std::string JsonQuoted(const std::string &value)
{
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (unsigned char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20U) {
                    result += "\\u00";
                    result.push_back(hex[(c >> 4U) & 0x0FU]);
                    result.push_back(hex[c & 0x0FU]);
                } else {
                    result.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    result.push_back('"');
    return result;
}

void JsonNumber(std::string &json, const char *key, uint64_t value)
{
    json += JsonQuoted(key);
    json += ":";
    json += std::to_string(value);
}

void JsonBool(std::string &json, const char *key, bool value)
{
    json += JsonQuoted(key);
    json += value ? ":true" : ":false";
}

void JsonString(std::string &json, const char *key, const std::string &value)
{
    json += JsonQuoted(key);
    json += ":";
    json += JsonQuoted(value);
}

void JsonNull(std::string &json, const char *key)
{
    json += JsonQuoted(key);
    json += ":null";
}

void JsonComma(std::string &json)
{
    json += ",";
}

bool ParseHashValue(const char *text, uint64_t *value)
{
    if (text == nullptr || text[0] == '\0' || value == nullptr || text[0] == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const int base = (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) ? 0 : 10;
    const unsigned long long parsed = strtoull(text, &end, base);
    if (errno != 0 || end == text || end == nullptr || *end != '\0') {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

void DisableRawAuditLocked(HookState &st)
{
    if (st.rawReports != nullptr) {
        (void)fclose(st.rawReports);
        st.rawReports = nullptr;
    }
    st.rawAuditFailed = true;
}

void EnsureRawAuditLocked(HookState &st) noexcept
{
    if (st.rawAuditInitialized) {
        return;
    }
    st.rawAuditInitialized = true;
    try {
        const char *targetOp = std::getenv("MSPROF_HOOK_RAW_OP");
        if (targetOp != nullptr && targetOp[0] != '\0') {
            st.rawTargetOp = targetOp;
        }
        const char *targetHash = std::getenv("MSPROF_HOOK_RAW_OP_HASH");
        st.rawTargetHashSet = ParseHashValue(targetHash, &st.rawTargetHash);
        st.rawAuditEnabled = !st.rawTargetOp.empty() || st.rawTargetHashSet;
        if (!st.rawAuditEnabled) {
            return;
        }
        const std::string path = st.outDir + "/raw_reports.jsonl";
        st.rawReports = fopen(path.c_str(), "w");
        if (st.rawReports == nullptr) {
            st.rawAuditFailed = true;
            (void)fprintf(stderr, "[msprof_hook] raw audit disabled: cannot open %s\n", path.c_str());
        }
    } catch (...) {
        DisableRawAuditLocked(st);
    }
}

bool RawTargetMatchesLocked(const HookState &st, uint64_t opHash)
{
    if (!st.rawAuditEnabled) {
        return false;
    }
    if (st.rawTargetHashSet && opHash == st.rawTargetHash) {
        return true;
    }
    if (st.rawTargetHashes.find(opHash) != st.rawTargetHashes.end()) {
        return true;
    }
    const auto name = st.hash2str.find(opHash);
    return !st.rawTargetOp.empty() && name != st.hash2str.end() && name->second == st.rawTargetOp;
}

const std::string *RawOpNameLocked(const HookState &st, uint64_t opHash)
{
    const auto it = st.hash2str.find(opHash);
    return it == st.hash2str.end() ? nullptr : &it->second;
}

void RawCommonFields(std::string &json, HookState &st, const char *event,
                     const char *entrypoint,
                     bool isBatch, int64_t batchIndex, int64_t batchCount,
                     uint64_t inputLength, uint64_t recordLength,
                     bool hasNonPersistentFlag, uint32_t nonPersistentFlag,
                     bool hasThreadId, uint32_t threadId,
                     bool hasTimestamp, uint64_t timestamp,
                     bool hasMagic, uint16_t magic,
                     bool hasLevel, uint16_t level,
                     bool hasType, uint32_t type,
                     uint64_t opHash)
{
    json = "{";
    JsonNumber(json, "event_seq", ++st.rawEventSeq);
    JsonComma(json); JsonString(json, "event", event);
    JsonComma(json); JsonString(json, "entrypoint", entrypoint);
    JsonComma(json); JsonBool(json, "is_batch", isBatch);
    JsonComma(json);
    if (batchIndex >= 0) JsonNumber(json, "batch_index", static_cast<uint64_t>(batchIndex));
    else JsonNull(json, "batch_index");
    JsonComma(json);
    if (batchCount >= 0) JsonNumber(json, "batch_count", static_cast<uint64_t>(batchCount));
    else JsonNull(json, "batch_count");
    JsonComma(json); JsonNumber(json, "input_length", inputLength);
    JsonComma(json); JsonNumber(json, "record_length", recordLength);
    JsonComma(json);
    if (hasNonPersistentFlag) JsonNumber(json, "non_persistent_flag", nonPersistentFlag);
    else JsonNull(json, "non_persistent_flag");
    JsonComma(json);
    if (hasThreadId) JsonNumber(json, "thread_id", threadId);
    else JsonNull(json, "thread_id");
    JsonComma(json);
    if (hasTimestamp) JsonNumber(json, "timestamp", timestamp);
    else JsonNull(json, "timestamp");
    JsonComma(json);
    if (hasMagic) JsonNumber(json, "magic", magic);
    else JsonNull(json, "magic");
    JsonComma(json);
    if (hasLevel) JsonNumber(json, "level", level);
    else JsonNull(json, "level");
    JsonComma(json);
    if (hasType) JsonNumber(json, "type", type);
    else JsonNull(json, "type");
    JsonComma(json); JsonNumber(json, "op_hash", opHash);
    JsonComma(json);
    const std::string *opName = RawOpNameLocked(st, opHash);
    if (opName != nullptr) JsonString(json, "op_name", *opName);
    else JsonNull(json, "op_name");
}

void WriteRawLineLocked(HookState &st, const std::string &json) noexcept
{
    if (st.rawReports == nullptr || st.rawAuditFailed) {
        return;
    }
    if (fwrite(json.data(), 1, json.size(), st.rawReports) != json.size() ||
        fputc('\n', st.rawReports) == EOF) {
        DisableRawAuditLocked(st);
    }
}

void RawHashRegistrationLocked(HookState &st, const char *entrypoint,
                               uint64_t id, const char *s, size_t len) noexcept
{
    if (!st.rawAuditEnabled || st.rawReports == nullptr || s == nullptr || len == 0) {
        return;
    }
    try {
        const bool nameMatches = !st.rawTargetOp.empty() && len == st.rawTargetOp.size() &&
            memcmp(s, st.rawTargetOp.data(), len) == 0;
        const bool hashMatches = st.rawTargetHashSet && id == st.rawTargetHash;
        if (!nameMatches && !hashMatches) {
            return;
        }
        if (nameMatches) {
            st.rawTargetHashes.insert(id);
        }
        std::string json;
        RawCommonFields(json, st, "hash_registration", entrypoint, false, -1, -1,
                        static_cast<uint64_t>(len), static_cast<uint64_t>(len),
                        false, 0, false, 0, false, 0, false, 0, false, 0, false, 0, id);
        JsonComma(json); JsonNumber(json, "string_length", static_cast<uint64_t>(len));
        JsonComma(json); JsonString(json, "registered_string", std::string(s, len));
        json += "}";
        WriteRawLineLocked(st, json);
    } catch (...) {
        DisableRawAuditLocked(st);
    }
}

void RawApiLocked(HookState &st, uint32_t nonPersistentFlag, const MsprofApi *api) noexcept
{
    if (api == nullptr || !RawTargetMatchesLocked(st, api->itemId) || st.rawReports == nullptr) {
        return;
    }
    try {
        std::string json;
        RawCommonFields(json, st, "api", "MsprofReportApi", false, -1, -1,
                        sizeof(*api), sizeof(*api), true, nonPersistentFlag,
                        true, api->threadId, false, 0, true, api->magicNumber,
                        true, api->level, true, api->type, api->itemId);
        JsonComma(json); JsonNumber(json, "reserve", api->reserve);
        JsonComma(json); JsonNumber(json, "begin_time", api->beginTime);
        JsonComma(json); JsonNumber(json, "end_time", api->endTime);
        JsonComma(json); JsonNumber(json, "item_id", api->itemId);
        json += "}";
        WriteRawLineLocked(st, json);
    } catch (...) {
        DisableRawAuditLocked(st);
    }
}

bool ReadU32(const uint8_t *data, size_t size, size_t offset, uint32_t *value)
{
    if (data == nullptr || value == nullptr || offset > size || size - offset < sizeof(*value)) {
        return false;
    }
    memcpy(value, data + offset, sizeof(*value));
    return true;
}

bool ReadU64(const uint8_t *data, size_t size, size_t offset, uint64_t *value)
{
    if (data == nullptr || value == nullptr || offset > size || size - offset < sizeof(*value)) {
        return false;
    }
    memcpy(value, data + offset, sizeof(*value));
    return true;
}

void RawCompactLocked(HookState &st, uint32_t nonPersistentFlag,
                      const void *data, uint32_t length) noexcept
{
    const size_t payloadOffset = offsetof(MsprofCompactInfo, data);
    if (data == nullptr || length < payloadOffset + sizeof(uint64_t) || st.rawReports == nullptr) {
        return;
    }
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    const uint8_t *payload = bytes + payloadOffset;
    const size_t payloadBytes = std::min<size_t>(
        static_cast<size_t>(length) - payloadOffset,
        sizeof(static_cast<const MsprofCompactInfo *>(data)->data.info));
    uint64_t opHash = 0;
    if (!ReadU64(payload, payloadBytes, offsetof(MsprofNodeBasicInfo, opName), &opHash) ||
        !RawTargetMatchesLocked(st, opHash)) {
        return;
    }
    try {
        uint16_t magic = 0;
        uint16_t level = 0;
        uint32_t type = 0;
        uint32_t threadId = 0;
        uint32_t dataLen = 0;
        uint64_t timestamp = 0;
        memcpy(&magic, bytes + offsetof(MsprofCompactInfo, magicNumber), sizeof(magic));
        memcpy(&level, bytes + offsetof(MsprofCompactInfo, level), sizeof(level));
        memcpy(&type, bytes + offsetof(MsprofCompactInfo, type), sizeof(type));
        memcpy(&threadId, bytes + offsetof(MsprofCompactInfo, threadId), sizeof(threadId));
        memcpy(&dataLen, bytes + offsetof(MsprofCompactInfo, dataLen), sizeof(dataLen));
        memcpy(&timestamp, bytes + offsetof(MsprofCompactInfo, timeStamp), sizeof(timestamp));

        uint32_t taskType = 0, blockDim = 0, opFlag = 0;
        uint64_t opTypeHash = 0;
        const bool taskAvailable = ReadU32(payload, payloadBytes,
            offsetof(MsprofNodeBasicInfo, taskType), &taskType);
        const bool typeHashAvailable = ReadU64(payload, payloadBytes,
            offsetof(MsprofNodeBasicInfo, opType), &opTypeHash);
        const bool blockAvailable = ReadU32(payload, payloadBytes,
            offsetof(MsprofNodeBasicInfo, blockDim), &blockDim);
        const bool flagAvailable = ReadU32(payload, payloadBytes,
            offsetof(MsprofNodeBasicInfo, opFlag), &opFlag);

        std::string json;
        RawCommonFields(json, st, "compact_basic", "MsprofReportCompactInfo", false, -1, -1,
                        length, length, true, nonPersistentFlag, true, threadId,
                        true, timestamp, true, magic, true, level, true, type, opHash);
        JsonComma(json); JsonNumber(json, "header_length", payloadOffset);
        JsonComma(json); JsonNumber(json, "data_len", dataLen);
        JsonComma(json);
        if (typeHashAvailable) JsonNumber(json, "op_type_hash", opTypeHash);
        else JsonNull(json, "op_type_hash");
        JsonComma(json);
        if (taskAvailable) JsonNumber(json, "task_type", taskType);
        else JsonNull(json, "task_type");
        JsonComma(json);
        if (blockAvailable) JsonNumber(json, "block_dim", blockDim);
        else JsonNull(json, "block_dim");
        JsonComma(json);
        if (flagAvailable) JsonNumber(json, "op_flag", opFlag);
        else JsonNull(json, "op_flag");

        const char *rejectReason = nullptr;
        if (length < sizeof(MsprofCompactInfo)) rejectReason = "input_length_short";
        else if (magic != kMagic) rejectReason = "magic_mismatch";
        else if (level != MSPROF_REPORT_NODE_LEVEL) rejectReason = "level_mismatch";
        else if (type != MSPROF_REPORT_NODE_BASIC_INFO_TYPE) rejectReason = "type_mismatch";
        JsonComma(json); JsonBool(json, "legacy_accept", rejectReason == nullptr);
        JsonComma(json);
        if (rejectReason != nullptr) JsonString(json, "legacy_reject_reason", rejectReason);
        else JsonNull(json, "legacy_reject_reason");
        JsonComma(json); JsonBool(json, "truncated", payloadBytes < sizeof(MsprofNodeBasicInfo));
        JsonComma(json);
        if (payloadBytes < sizeof(MsprofNodeBasicInfo)) JsonString(json, "truncation_reason", "payload_short");
        else JsonNull(json, "truncation_reason");
        json += "}";
        WriteRawLineLocked(st, json);
    } catch (...) {
        DisableRawAuditLocked(st);
    }
}

void RawAdditionalLocked(HookState &st, uint32_t nonPersistentFlag,
                         const void *data, uint32_t recordLength,
                         uint32_t inputLength, bool isBatch,
                         uint32_t batchIndex, uint32_t batchCount) noexcept
{
    const size_t payloadOffset = offsetof(MsprofAdditionalInfo, data);
    if (data == nullptr || recordLength < payloadOffset + sizeof(uint64_t) || st.rawReports == nullptr) {
        return;
    }
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    const uint8_t *payload = bytes + payloadOffset;
    const size_t payloadBytes = std::min<size_t>(
        static_cast<size_t>(recordLength) - payloadOffset,
        MSPROF_ADDTIONAL_INFO_DATA_LENGTH);
    uint64_t opHash = 0;
    if (!ReadU64(payload, payloadBytes, offsetof(MsprofTensorInfo, opName), &opHash) ||
        !RawTargetMatchesLocked(st, opHash)) {
        return;
    }
    try {
        uint16_t magic = 0;
        uint16_t level = 0;
        uint32_t type = 0;
        uint32_t threadId = 0;
        uint32_t dataLen = 0;
        uint64_t timestamp = 0;
        memcpy(&magic, bytes + offsetof(MsprofAdditionalInfo, magicNumber), sizeof(magic));
        memcpy(&level, bytes + offsetof(MsprofAdditionalInfo, level), sizeof(level));
        memcpy(&type, bytes + offsetof(MsprofAdditionalInfo, type), sizeof(type));
        memcpy(&threadId, bytes + offsetof(MsprofAdditionalInfo, threadId), sizeof(threadId));
        memcpy(&dataLen, bytes + offsetof(MsprofAdditionalInfo, dataLen), sizeof(dataLen));
        memcpy(&timestamp, bytes + offsetof(MsprofAdditionalInfo, timeStamp), sizeof(timestamp));

        uint32_t tensorNum = 0;
        const size_t tensorNumOffset = offsetof(MsprofTensorInfo, tensorNum);
        const bool tensorNumAvailable = ReadU32(payload, payloadBytes, tensorNumOffset, &tensorNum);
        const size_t slotsOffset = offsetof(MsprofTensorInfo, tensorData);
        const size_t slotBytes = sizeof(MsrofTensorData);
        const size_t slotAvailableBytes = payloadBytes > slotsOffset ? payloadBytes - slotsOffset : 0;
        const size_t availableSlots = std::min<size_t>(
            MSPROF_GE_TENSOR_DATA_NUM, slotAvailableBytes / slotBytes);
        const size_t decodedCount = tensorNumAvailable
            ? std::min<size_t>(tensorNum, availableSlots) : 0;
        const bool truncated = payloadBytes < sizeof(MsprofTensorInfo) ||
            (tensorNumAvailable && tensorNum > availableSlots);
        const char *truncationReason = nullptr;
        if (payloadBytes < sizeof(MsprofTensorInfo)) truncationReason = "payload_short";
        else if (tensorNumAvailable && tensorNum > availableSlots) truncationReason = "tensor_slots_short";

        const bool outerAccepted = recordLength >= sizeof(MsprofAdditionalInfo);
        const char *rejectReason = nullptr;
        if (!outerAccepted) rejectReason = "input_length_short";
        else if (magic != kMagic) rejectReason = "magic_mismatch";
        else if (level != MSPROF_REPORT_NODE_LEVEL) rejectReason = "level_mismatch";
        else if (type != MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) rejectReason = "type_mismatch";
        else if (dataLen < sizeof(MsprofTensorInfo)) rejectReason = "data_len_short";

        std::string json;
        RawCommonFields(json, st, "additional_tensor_info",
                        isBatch ? "MsprofReportBatchAdditionalInfo" : "MsprofReportAdditionalInfo", isBatch,
                        isBatch ? static_cast<int64_t>(batchIndex) : -1,
                        isBatch ? static_cast<int64_t>(batchCount) : -1,
                        inputLength, recordLength, true, nonPersistentFlag,
                        true, threadId, true, timestamp, true, magic, true, level,
                        true, type, opHash);
        JsonComma(json); JsonNumber(json, "header_length", payloadOffset);
        JsonComma(json); JsonNumber(json, "data_len", dataLen);
        JsonComma(json); JsonNumber(json, "actual_payload_bytes", payloadBytes);
        JsonComma(json);
        if (tensorNumAvailable) JsonNumber(json, "tensor_num", tensorNum);
        else JsonNull(json, "tensor_num");
        JsonComma(json); JsonNumber(json, "tensor_capacity", MSPROF_GE_TENSOR_DATA_NUM);
        JsonComma(json); JsonNumber(json, "decoded_tensor_count", decodedCount);
        JsonComma(json); JsonBool(json, "legacy_accept", rejectReason == nullptr);
        JsonComma(json);
        if (rejectReason != nullptr) JsonString(json, "legacy_reject_reason", rejectReason);
        else JsonNull(json, "legacy_reject_reason");
        JsonComma(json); JsonBool(json, "truncated", truncated);
        JsonComma(json);
        if (truncationReason != nullptr) JsonString(json, "truncation_reason", truncationReason);
        else JsonNull(json, "truncation_reason");
        JsonComma(json); json += JsonQuoted("raw_tensor_slots"); json += ":[";
        for (uint32_t i = 0; i < MSPROF_GE_TENSOR_DATA_NUM; ++i) {
            if (i != 0) json += ",";
            json += "{";
            const size_t slotOffset = slotsOffset + static_cast<size_t>(i) * slotBytes;
            uint32_t tensorType = 0, format = 0, dtype = 0;
            const bool tensorTypeAvailable = ReadU32(payload, payloadBytes,
                slotOffset + offsetof(MsrofTensorData, tensorType), &tensorType);
            const bool formatAvailable = ReadU32(payload, payloadBytes,
                slotOffset + offsetof(MsrofTensorData, format), &format);
            const bool dtypeAvailable = ReadU32(payload, payloadBytes,
                slotOffset + offsetof(MsrofTensorData, dataType), &dtype);
            if (tensorTypeAvailable) JsonNumber(json, "tensor_type", tensorType);
            else JsonNull(json, "tensor_type");
            JsonComma(json);
            if (formatAvailable) JsonNumber(json, "format", format);
            else JsonNull(json, "format");
            JsonComma(json);
            if (dtypeAvailable) JsonNumber(json, "dtype", dtype);
            else JsonNull(json, "dtype");
            JsonComma(json); json += JsonQuoted("shape_raw"); json += ":[";
            for (uint32_t d = 0; d < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++d) {
                if (d != 0) json += ",";
                uint32_t dimension = 0;
                const size_t dimensionOffset = slotOffset + offsetof(MsrofTensorData, shape) +
                    static_cast<size_t>(d) * sizeof(uint32_t);
                if (ReadU32(payload, payloadBytes, dimensionOffset, &dimension)) {
                    json += std::to_string(dimension);
                } else {
                    json += "null";
                }
            }
            json += "]}";
        }
        json += "]}";
        WriteRawLineLocked(st, json);
    } catch (...) {
        DisableRawAuditLocked(st);
    }
}

void FlushRawLocked(HookState &st) noexcept
{
    if (st.rawReports == nullptr) {
        return;
    }
    if (fflush(st.rawReports) != 0) {
        DisableRawAuditLocked(st);
        return;
    }
    (void)fclose(st.rawReports);
    st.rawReports = nullptr;
}

// ---------------------------------------------------------------------------
// CSV flush
// ---------------------------------------------------------------------------
void WriteFile(const std::string &path, const std::string &content)
{
    FILE *fp = fopen(path.c_str(), "w");
    if (fp == nullptr) {
        (void)fprintf(stderr, "[msprof_hook] open %s failed\n", path.c_str());
        return;
    }
    (void)fwrite(content.data(), 1, content.size(), fp);
    (void)fclose(fp);
}

void FlushLocked(HookState &st)
{
    if (st.flushed || st.outDir.empty()) {
        FlushRawLocked(st);
        return;
    }
    // Do not create/overwrite CSV from processes that never saw a report call
    // (e.g. bash/tee wrappers when the hook is inherited via LD_PRELOAD).
    if (st.hash2str.empty() && st.ops.empty() && st.launches.empty()) {
        FlushRawLocked(st);
        return;
    }
    st.flushed = true;

    // hash_dic.csv
    std::string hashCsv = "hash,string\n";
    for (const auto &kv : st.hash2str) {
        hashCsv += std::to_string(kv.first) + "," + CsvEscape(kv.second) + "\n";
    }
    WriteFile(st.outDir + "/hash_dic.csv", hashCsv);

    // launch.csv
    std::string launchCsv = "thread_id,op_name,begin_ns,end_ns,duration_ns\n";
    for (const auto &l : st.launches) {
        std::string name;
        auto it = st.hash2str.find(l.opHash);
        name = (it == st.hash2str.end()) ? Hex(l.opHash) : it->second;
        uint64_t dur = (l.endTime >= l.beginTime) ? (l.endTime - l.beginTime) : 0;
        launchCsv += std::to_string(l.tid) + "," + CsvEscape(name) + "," +
                     std::to_string(l.beginTime) + "," + std::to_string(l.endTime) + "," +
                     std::to_string(dur) + "\n";
    }
    WriteFile(st.outDir + "/launch.csv", launchCsv);

    // ops.csv
    std::string opsCsv =
        "thread_id,timestamp,op_name,op_type,task_type,task_type_name,block_dim,op_flag,"
        "input_num,input_formats,input_dtypes,input_shapes,"
        "output_num,output_formats,output_dtypes,output_shapes\n";
    for (const auto &kv : st.ops) {
        const OpRecord &op = kv.second;
        auto nameIt = st.hash2str.find(op.opHash);
        auto typeIt = st.hash2str.find(op.opTypeHash);
        const std::string opName = (nameIt == st.hash2str.end()) ? Hex(op.opHash) : nameIt->second;
        const std::string opType = (typeIt == st.hash2str.end()) ? Hex(op.opTypeHash) : typeIt->second;

        std::vector<std::string> inFmt, inDtype, inShape, outFmt, outDtype, outShape;
        for (const auto &t : op.tensors) {
            if (t.tensorType == MSPROF_GE_TENSOR_TYPE_INPUT) {
                inFmt.emplace_back(FormatName(t.format));
                inDtype.emplace_back(DataTypeName(t.dataType));
                inShape.emplace_back(ShapeToString(t.shape));
            } else {
                outFmt.emplace_back(FormatName(t.format));
                outDtype.emplace_back(DataTypeName(t.dataType));
                outShape.emplace_back(ShapeToString(t.shape));
            }
        }
        auto join = [](const std::vector<std::string> &v) {
            std::string s;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i != 0) {
                    s += ";";
                }
                s += v[i];
            }
            return s.empty() ? std::string("N/A") : s;
        };
        opsCsv += std::to_string(kv.first.tid) + "," + std::to_string(kv.first.timestamp) + "," +
                  CsvEscape(opName) + "," + CsvEscape(opType) + "," +
                  std::to_string(op.taskType) + "," + TaskTypeName(op.taskType) + "," +
                  std::to_string(op.blockDim) + "," +
                  std::to_string(op.opFlag) + "," +
                  std::to_string(inFmt.size()) + "," + join(inFmt) + "," + join(inDtype) + "," +
                  CsvEscape(join(inShape)) + "," +
                  std::to_string(outFmt.size()) + "," + join(outFmt) + "," + join(outDtype) + "," +
                  CsvEscape(join(outShape)) + "\n";
    }
    WriteFile(st.outDir + "/ops.csv", opsCsv);

    (void)fprintf(stderr, "[msprof_hook] flushed: %zu hashes, %zu ops, %zu launches -> %s\n",
        st.hash2str.size(), st.ops.size(), st.launches.size(), st.outDir.c_str());
    FlushRawLocked(st);
}

// ---------------------------------------------------------------------------
// record handlers (must hold lock)
// ---------------------------------------------------------------------------
void OnHashLocked(HookState &st, const char *entrypoint, uint64_t id, const char *s, size_t len)
{
    if (s == nullptr || len == 0) {
        return;
    }
    if (st.hash2str.find(id) == st.hash2str.end()) {
        st.hash2str.emplace(id, std::string(s, len));
    }
    RawHashRegistrationLocked(st, entrypoint, id, s, len);
}

OpRecord &GetOrCreateOp(HookState &st, uint32_t tid, uint64_t opHash, uint64_t ts)
{
    OpKey key{tid, opHash, ts};
    auto it = st.ops.find(key);
    if (it == st.ops.end()) {
        OpRecord rec;
        rec.opHash = opHash;
        it = st.ops.emplace(key, rec).first;
    }
    return it->second;
}

void OnApiLocked(HookState &st, const MsprofApi *api)
{
    if (api == nullptr || api->magicNumber != kMagic) {
        return;
    }
    if (api->level == MSPROF_REPORT_NODE_LEVEL && api->type == MSPROF_REPORT_NODE_LAUNCH_TYPE) {
        LaunchRecord l;
        l.tid = api->threadId;
        l.opHash = api->itemId;
        l.beginTime = api->beginTime;
        l.endTime = api->endTime;
        st.launches.push_back(l);
        if (st.verbose) {
            (void)fprintf(stderr, "[msprof_hook] launch tid=%u op=%s begin=%" PRIu64 " end=%" PRIu64 "\n",
                l.tid, Hex(l.opHash).c_str(), l.beginTime, l.endTime);
        }
    }
}

void OnEventLocked(HookState &st, const MsprofEvent *event)
{
    if (event == nullptr || event->magicNumber != kMagic) {
        return;
    }
    if (st.verbose) {
        (void)fprintf(stderr, "[msprof_hook] event level=%u type=%u item=%s ts=%" PRIu64 "\n",
            event->level, event->type, Hex(event->itemId).c_str(), event->timeStamp);
    }
}

void OnCompactLocked(HookState &st, const void *data, uint32_t len)
{
    if (data == nullptr || len < sizeof(MsprofCompactInfo)) {
        return;
    }
    const auto *info = static_cast<const MsprofCompactInfo *>(data);
    if (info->magicNumber != kMagic) {
        return;
    }
    if (info->level != MSPROF_REPORT_NODE_LEVEL || info->type != MSPROF_REPORT_NODE_BASIC_INFO_TYPE) {
        return;
    }
    const MsprofNodeBasicInfo &nb = info->data.nodeBasicInfo;
    OpRecord &op = GetOrCreateOp(st, info->threadId, nb.opName, info->timeStamp);
    op.opTypeHash = nb.opType;
    op.taskType = nb.taskType;
    op.blockDim = nb.blockDim;
    op.opFlag = nb.opFlag;
    op.hasBasic = true;
    if (st.verbose) {
        (void)fprintf(stderr, "[msprof_hook] node_basic_info tid=%u op=%s opType=%s taskType=%u blockDim=%u\n",
            info->threadId, Hex(nb.opName).c_str(), Hex(nb.opType).c_str(), nb.taskType, nb.blockDim);
    }
}

void OnAdditionalLocked(HookState &st, const void *data, uint32_t len)
{
    if (data == nullptr || len < sizeof(MsprofAdditionalInfo)) {
        return;
    }
    const auto *info = static_cast<const MsprofAdditionalInfo *>(data);
    if (info->magicNumber != kMagic) {
        return;
    }
    if (info->level != MSPROF_REPORT_NODE_LEVEL || info->type != MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) {
        return;
    }
    if (info->dataLen < sizeof(MsprofTensorInfo)) {
        return;
    }
    const auto *ti = reinterpret_cast<const MsprofTensorInfo *>(info->data);
    OpRecord &op = GetOrCreateOp(st, info->threadId, ti->opName, info->timeStamp);
    uint32_t num = std::min<uint32_t>(ti->tensorNum, MSPROF_GE_TENSOR_DATA_NUM);
    for (uint32_t i = 0; i < num; ++i) {
        TensorDesc desc;
        desc.tensorType = ti->tensorData[i].tensorType;
        desc.format = ti->tensorData[i].format;
        desc.dataType = ti->tensorData[i].dataType;
        (void)memcpy(desc.shape, ti->tensorData[i].shape, sizeof(desc.shape));
        op.tensors.push_back(desc);
    }
    if (st.verbose) {
        (void)fprintf(stderr, "[msprof_hook] tensor_info tid=%u op=%s tensorNum=%u\n",
            info->threadId, Hex(ti->opName).c_str(), ti->tensorNum);
    }
}

// ---------------------------------------------------------------------------
// symbol resolution
// ---------------------------------------------------------------------------
template <typename T>
T Resolve(const char *name)
{
    return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}

}  // namespace

// ===========================================================================
// hooks
// ===========================================================================
extern "C" uint64_t MsprofGetHashId(const char *hashInfo, size_t length)
{
    using Fn = uint64_t (*)(const char *, size_t);
    static Fn real = Resolve<Fn>("MsprofGetHashId");
    uint64_t id = (real != nullptr) ? real(hashInfo, length) : 0;
    HookState &st = State();
    EnsureInit(st);
    std::lock_guard<std::mutex> lk(st.mtx);
    EnsureRawAuditLocked(st);
    OnHashLocked(st, "MsprofGetHashId", id, hashInfo, length);
    return id;
}

extern "C" uint64_t MsprofStr2Id(const char *hashInfo, size_t length)
{
    using Fn = uint64_t (*)(const char *, size_t);
    static Fn real = Resolve<Fn>("MsprofStr2Id");
    uint64_t id = (real != nullptr) ? real(hashInfo, length) : 0;
    HookState &st = State();
    EnsureInit(st);
    std::lock_guard<std::mutex> lk(st.mtx);
    EnsureRawAuditLocked(st);
    OnHashLocked(st, "MsprofStr2Id", id, hashInfo, length);
    return id;
}

extern "C" int32_t MsprofReportApi(uint32_t nonPersistantFlag, const MsprofApi *api)
{
    using Fn = int32_t (*)(uint32_t, const MsprofApi *);
    static Fn real = Resolve<Fn>("MsprofReportApi");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        EnsureRawAuditLocked(st);
        RawApiLocked(st, nonPersistantFlag, api);
        OnApiLocked(st, api);
    }
    return (real != nullptr) ? real(nonPersistantFlag, api) : 0;
}

extern "C" int32_t MsprofReportEvent(uint32_t nonPersistantFlag, const MsprofEvent *event)
{
    using Fn = int32_t (*)(uint32_t, const MsprofEvent *);
    static Fn real = Resolve<Fn>("MsprofReportEvent");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        OnEventLocked(st, event);
    }
    return (real != nullptr) ? real(nonPersistantFlag, event) : 0;
}

extern "C" int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length)
{
    using Fn = int32_t (*)(uint32_t, const VOID_PTR, uint32_t);
    static Fn real = Resolve<Fn>("MsprofReportCompactInfo");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        EnsureRawAuditLocked(st);
        RawCompactLocked(st, nonPersistantFlag, data, length);
        OnCompactLocked(st, data, length);
    }
    return (real != nullptr) ? real(nonPersistantFlag, data, length) : 0;
}

extern "C" int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length)
{
    using Fn = int32_t (*)(uint32_t, const VOID_PTR, uint32_t);
    static Fn real = Resolve<Fn>("MsprofReportAdditionalInfo");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        EnsureRawAuditLocked(st);
        RawAdditionalLocked(st, nonPersistantFlag, data, length, length, false, 0, 0);
        OnAdditionalLocked(st, data, length);
    }
    return (real != nullptr) ? real(nonPersistantFlag, data, length) : 0;
}

extern "C" int32_t MsprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length)
{
    using Fn = int32_t (*)(uint32_t, const VOID_PTR, uint32_t);
    static Fn real = Resolve<Fn>("MsprofReportBatchAdditionalInfo");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        EnsureRawAuditLocked(st);
        const auto *base = static_cast<const uint8_t *>(data);
        const uint32_t count = (sizeof(MsprofAdditionalInfo) == 0) ? 0 :
            (length / static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
        if (base != nullptr) {
            for (uint32_t i = 0; i < count; ++i) {
                const void *entry = base + static_cast<size_t>(i) * sizeof(MsprofAdditionalInfo);
                RawAdditionalLocked(st, nonPersistantFlag, entry,
                    static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)), length,
                    true, i, count);
                OnAdditionalLocked(st, entry, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
            }
        }
    }
    return (real != nullptr) ? real(nonPersistantFlag, data, length) : 0;
}

extern "C" int32_t MsprofFinalize(void)
{
    using Fn = int32_t (*)(void);
    static Fn real = Resolve<Fn>("MsprofFinalize");
    HookState &st = State();
    EnsureInit(st);
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        FlushLocked(st);
    }
    return (real != nullptr) ? real() : 0;
}

// final safety flush on unload/exit
__attribute__((destructor)) static void MsprofHookFini()
{
    HookState &st = State();
    std::lock_guard<std::mutex> lk(st.mtx);
    FlushLocked(st);
}
