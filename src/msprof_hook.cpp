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
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
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
    std::map<OpKey, OpRecord> ops;
    std::vector<LaunchRecord> launches;
    std::string outDir;
    bool verbose = false;
    bool inited = false;
    bool flushed = false;
};

HookState &State()
{
    static HookState state;
    return state;
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
        return;
    }
    // Do not create/overwrite CSV from processes that never saw a report call
    // (e.g. bash/tee wrappers when the hook is inherited via LD_PRELOAD).
    if (st.hash2str.empty() && st.ops.empty() && st.launches.empty()) {
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
}

// ---------------------------------------------------------------------------
// record handlers (must hold lock)
// ---------------------------------------------------------------------------
void OnHashLocked(HookState &st, uint64_t id, const char *s, size_t len)
{
    if (s == nullptr || len == 0) {
        return;
    }
    if (st.hash2str.find(id) == st.hash2str.end()) {
        st.hash2str.emplace(id, std::string(s, len));
    }
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
    OnHashLocked(st, id, hashInfo, length);
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
    OnHashLocked(st, id, hashInfo, length);
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
        const auto *base = static_cast<const uint8_t *>(data);
        const uint32_t count = (sizeof(MsprofAdditionalInfo) == 0) ? 0 :
            (length / static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
        for (uint32_t i = 0; i < count; ++i) {
            OnAdditionalLocked(st, base + i * sizeof(MsprofAdditionalInfo),
                static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
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
