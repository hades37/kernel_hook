#include "profiling/aprof_pub.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr int kMaxCalls = 32;
constexpr uint32_t kMaxBytes = 4096;

struct CallRecord {
    uint32_t kind = 0;
    uint32_t flags = 0;
    const void *original = nullptr;
    uint32_t length = 0;
    uint8_t bytes[kMaxBytes] = {};
};

CallRecord g_calls[kMaxCalls];
int g_call_count = 0;
int g_hash_call_count = 0;

void SaveCall(uint32_t kind, uint32_t flags, const void *data, uint32_t length)
{
    if (g_call_count >= kMaxCalls) {
        return;
    }
    CallRecord &record = g_calls[g_call_count++];
    record.kind = kind;
    record.flags = flags;
    record.original = data;
    record.length = length;
    const uint32_t copy_length = (length < kMaxBytes) ? length : kMaxBytes;
    if (data != nullptr && copy_length > 0) {
        std::memcpy(record.bytes, data, copy_length);
    }
}

uint64_t HashValue(const char *value, size_t length)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; value != nullptr && i < length; ++i) {
        hash ^= static_cast<unsigned char>(value[i]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
}  // namespace

extern "C" uint64_t MsprofGetHashId(const char *value, size_t length)
{
    ++g_hash_call_count;
    return HashValue(value, length);
}

extern "C" uint64_t MsprofStr2Id(const char *value, size_t length)
{
    ++g_hash_call_count;
    return HashValue(value, length);
}

extern "C" int32_t MsprofReportApi(uint32_t flags, const MsprofApi *api)
{
    SaveCall(1, flags, api, api == nullptr ? 0 : static_cast<uint32_t>(sizeof(*api)));
    return 101;
}

extern "C" int32_t MsprofReportEvent(uint32_t flags, const MsprofEvent *event)
{
    SaveCall(2, flags, event, event == nullptr ? 0 : static_cast<uint32_t>(sizeof(*event)));
    return 102;
}

extern "C" int32_t MsprofReportCompactInfo(uint32_t flags, const VOID_PTR data, uint32_t length)
{
    SaveCall(3, flags, data, length);
    return 103;
}

extern "C" int32_t MsprofReportAdditionalInfo(uint32_t flags, const VOID_PTR data, uint32_t length)
{
    SaveCall(4, flags, data, length);
    return 104;
}

extern "C" int32_t MsprofReportBatchAdditionalInfo(uint32_t flags, const VOID_PTR data, uint32_t length)
{
    SaveCall(5, flags, data, length);
    return 105;
}

extern "C" int32_t MsprofFinalize(void)
{
    return 106;
}

extern "C" void FakeMsprofReset(void)
{
    std::memset(g_calls, 0, sizeof(g_calls));
    g_call_count = 0;
    g_hash_call_count = 0;
}

extern "C" int32_t FakeMsprofCallCount(void)
{
    return g_call_count;
}

extern "C" int32_t FakeMsprofHashCallCount(void)
{
    return g_hash_call_count;
}

extern "C" int32_t FakeMsprofGetCall(int32_t index, uint32_t *kind, uint32_t *flags,
                                     const void **original, uint32_t *length,
                                     void *copy, uint32_t copy_capacity)
{
    if (index < 0 || index >= g_call_count || kind == nullptr || flags == nullptr ||
        original == nullptr || length == nullptr) {
        return 0;
    }
    const CallRecord &record = g_calls[index];
    *kind = record.kind;
    *flags = record.flags;
    *original = record.original;
    *length = record.length;
    if (copy != nullptr) {
        if (record.length > copy_capacity || record.length > kMaxBytes) {
            return 0;
        }
        if (record.length > 0) {
            std::memcpy(copy, record.bytes, record.length);
        }
    }
    return 1;
}
