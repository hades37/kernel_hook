#include "profiling/aprof_pub.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {
std::atomic<int> g_calls{0};

uint64_t HashValue(const char *value, size_t length)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; value != nullptr && i < length; ++i) {
        hash ^= static_cast<unsigned char>(value[i]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
}

extern "C" uint64_t MsprofGetHashId(const char *value, size_t length)
{
    ++g_calls;
    return HashValue(value, length);
}

extern "C" uint64_t MsprofStr2Id(const char *value, size_t length)
{
    ++g_calls;
    return HashValue(value, length);
}

extern "C" int32_t MsprofReportApi(uint32_t, const MsprofApi *)
{
    ++g_calls;
    return 101;
}

extern "C" int32_t MsprofReportCompactInfo(uint32_t, const VOID_PTR, uint32_t)
{
    ++g_calls;
    return 102;
}

extern "C" int32_t MsprofReportAdditionalInfo(uint32_t, const VOID_PTR, uint32_t)
{
    ++g_calls;
    return 103;
}

extern "C" int32_t FakeMsprofCallCount()
{
    return g_calls.load();
}
