#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "profiling/aprof_pub.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

extern "C" void FakeMsprofReset(void);
extern "C" int32_t FakeMsprofCallCount(void);
extern "C" int32_t FakeMsprofHashCallCount(void);
extern "C" int32_t FakeMsprofGetCall(int32_t, uint32_t *, uint32_t *, const void **,
                                     uint32_t *, void *, uint32_t);

namespace {
constexpr char kTargetOp[] = "kda_gate_chunk_cumsum_vector_kernel";
constexpr char kOtherOp[] = "RawReportOtherOp";
constexpr char kTargetType[] = "RawReportTargetType";
constexpr uint32_t kThreadId = 731;
constexpr uint32_t kOtherThreadId = 732;

enum CallKind : uint32_t {
    kApiCall = 1,
    kEventCall = 2,
    kCompactCall = 3,
    kAdditionalCall = 4,
    kBatchAdditionalCall = 5,
};

int Fail(const char *message)
{
    std::fprintf(stderr, "raw_report_contract_probe: %s\n", message);
    return 1;
}

uint64_t Hash(const char *name)
{
    return MsprofGetHashId(name, std::strlen(name));
}

bool VerifyForward(int index, uint32_t kind, uint32_t flags, const void *original,
                   uint32_t length)
{
    uint32_t actual_kind = 0;
    uint32_t actual_flags = 0;
    uint32_t actual_length = 0;
    const void *actual_original = nullptr;
    uint8_t copied[4096] = {};
    if (!FakeMsprofGetCall(index, &actual_kind, &actual_flags, &actual_original,
                           &actual_length, copied, sizeof(copied))) {
        return false;
    }
    return actual_kind == kind && actual_flags == flags && actual_original == original &&
           actual_length == length &&
           (length == 0 || std::memcmp(copied, original, length) == 0);
}

void FillAdditional(MsprofAdditionalInfo *record, uint64_t op_hash, uint32_t thread_id,
                    uint64_t timestamp, uint32_t tensor_num, uint32_t data_len)
{
    std::memset(record, 0, sizeof(*record));
    record->magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    record->level = MSPROF_REPORT_NODE_LEVEL;
    record->type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    record->threadId = thread_id;
    record->dataLen = data_len;
    record->timeStamp = timestamp;
    auto *tensor_info = reinterpret_cast<MsprofTensorInfo *>(record->data);
    tensor_info->opName = op_hash;
    tensor_info->tensorNum = tensor_num;
    for (uint32_t i = 0; i < tensor_num && i < MSPROF_GE_TENSOR_DATA_NUM; ++i) {
        tensor_info->tensorData[i].tensorType = (i % 2 == 0)
            ? MSPROF_GE_TENSOR_TYPE_INPUT : MSPROF_GE_TENSOR_TYPE_OUTPUT;
        tensor_info->tensorData[i].format = UINT32_C(0x12340000) + i;
        tensor_info->tensorData[i].dataType = UINT32_C(0xABCDEF00) + i;
        for (uint32_t d = 0; d < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++d) {
            tensor_info->tensorData[i].shape[d] = (d % 2 == 0) ? (1000U * i + d + 1) : 0;
        }
    }
}

int32_t ReportApi(uint64_t op_hash, uint32_t flags, uint32_t thread_id,
                  uint64_t begin, uint64_t end, MsprofApi *record)
{
    std::memset(record, 0, sizeof(*record));
    record->magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    record->level = MSPROF_REPORT_NODE_LEVEL;
    record->type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    record->threadId = thread_id;
    record->reserve = UINT32_C(0xA1B2C3D4);
    record->beginTime = begin;
    record->endTime = end;
    record->itemId = op_hash;
    return MsprofReportApi(flags, record);
}

int32_t ReportCompact(uint64_t op_hash, uint64_t type_hash, uint32_t flags,
                      uint32_t thread_id, uint64_t timestamp, MsprofCompactInfo *record)
{
    std::memset(record, 0, sizeof(*record));
    record->magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    record->level = MSPROF_REPORT_NODE_LEVEL;
    record->type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    record->threadId = thread_id;
    record->dataLen = sizeof(MsprofNodeBasicInfo);
    record->timeStamp = timestamp;
    record->data.nodeBasicInfo.opName = op_hash;
    record->data.nodeBasicInfo.opType = type_hash;
    record->data.nodeBasicInfo.taskType = 2;
    record->data.nodeBasicInfo.blockDim = 48;
    record->data.nodeBasicInfo.opFlag = UINT32_C(0xF00D1234);
    return MsprofReportCompactInfo(flags, record, sizeof(*record));
}

int32_t ReportAdditional(uint64_t op_hash, uint32_t flags, uint32_t thread_id,
                         uint64_t timestamp, uint32_t tensor_num, uint32_t data_len,
                         MsprofAdditionalInfo *record)
{
    FillAdditional(record, op_hash, thread_id, timestamp, tensor_num, data_len);
    return MsprofReportAdditionalInfo(flags, record, sizeof(*record));
}
}  // namespace

int main()
{
    static_assert(sizeof(MsprofTensorInfo) <= MSPROF_ADDTIONAL_INFO_DATA_LENGTH,
                  "MsprofTensorInfo must fit in the public additional-info buffer");
    FakeMsprofReset();

    const uint64_t target_hash = Hash(kTargetOp);
    const uint64_t other_hash = Hash(kOtherOp);
    const uint64_t target_type_hash = MsprofStr2Id(kTargetType, std::strlen(kTargetType));
    if (target_hash == 0 || other_hash == 0 || target_type_hash == 0) {
        return Fail("fake SDK returned a zero hash");
    }

    int call_index = 0;
    MsprofApi target_api{};
    if (ReportApi(target_hash, 0xA101, kThreadId, 1200, 1250, &target_api) != 101 ||
        !VerifyForward(call_index++, kApiCall, 0xA101, &target_api, sizeof(target_api))) {
        return Fail("target API parameters/status were not forwarded once unchanged");
    }
    MsprofApi other_api{};
    if (ReportApi(other_hash, 0xA102, kOtherThreadId, 2200, 2250, &other_api) != 101 ||
        !VerifyForward(call_index++, kApiCall, 0xA102, &other_api, sizeof(other_api))) {
        return Fail("non-target API parameters/status were not forwarded once unchanged");
    }

    MsprofCompactInfo target_compact{};
    if (ReportCompact(target_hash, target_type_hash, 0xB101, kThreadId, 1001,
                      &target_compact) != 103 ||
        !VerifyForward(call_index++, kCompactCall, 0xB101, &target_compact,
                       sizeof(target_compact))) {
        return Fail("target CompactInfo parameters/status were not forwarded unchanged");
    }
    MsprofCompactInfo other_compact{};
    if (ReportCompact(other_hash, target_type_hash, 0xB102, kOtherThreadId, 1002,
                      &other_compact) != 103 ||
        !VerifyForward(call_index++, kCompactCall, 0xB102, &other_compact,
                       sizeof(other_compact))) {
        return Fail("non-target CompactInfo parameters/status were not forwarded unchanged");
    }

    MsprofAdditionalInfo valid_tensor{};
    const uint32_t tensor_info_size = sizeof(MsprofTensorInfo);
    if (ReportAdditional(target_hash, 0xC101, kThreadId, 2001, 1, tensor_info_size,
                         &valid_tensor) != 104 ||
        !VerifyForward(call_index++, kAdditionalCall, 0xC101, &valid_tensor,
                       sizeof(valid_tensor))) {
        return Fail("target TensorInfo parameters/status were not forwarded unchanged");
    }

    MsprofAdditionalInfo zero_tensor{};
    if (ReportAdditional(target_hash, 0xC102, kThreadId, 2002, 0, tensor_info_size,
                         &zero_tensor) != 104 ||
        !VerifyForward(call_index++, kAdditionalCall, 0xC102, &zero_tensor,
                       sizeof(zero_tensor))) {
        return Fail("zero-tensor report parameters/status were not forwarded unchanged");
    }

    // Triton may leave dataLen zero while the fixed MsprofAdditionalInfo buffer
    // still contains a complete tensor payload. Raw audit must expose the bytes
    // before the legacy collector's dataLen guard, without changing that guard.
    MsprofAdditionalInfo short_declared_length{};
    if (ReportAdditional(target_hash, 0xC103, kThreadId, 2003, 4, 0,
                         &short_declared_length) != 104 ||
        !VerifyForward(call_index++, kAdditionalCall, 0xC103, &short_declared_length,
                       sizeof(short_declared_length))) {
        return Fail("short-dataLen report parameters/status were not forwarded unchanged");
    }

    MsprofAdditionalInfo partial_record{};
    FillAdditional(&partial_record, target_hash, kThreadId, 2004, 1, tensor_info_size);
    constexpr size_t partial_length = offsetof(MsprofAdditionalInfo, data) +
                                      offsetof(MsprofTensorInfo, tensorData);
    if (partial_length >= sizeof(partial_record)) {
        return Fail("partial-buffer test does not truncate the tensor slot payload");
    }
    const long system_page_size = sysconf(_SC_PAGESIZE);
    if (system_page_size <= 0 || static_cast<size_t>(system_page_size) <= partial_length) {
        return Fail("could not obtain a usable system page for the truncated-buffer guard");
    }
    const size_t page_size = static_cast<size_t>(system_page_size);
    void *guarded_mapping = mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (guarded_mapping == MAP_FAILED) {
        return Fail("mmap failed for the truncated-buffer guard");
    }
    auto *guard_page = static_cast<uint8_t *>(guarded_mapping) + page_size;
    if (mprotect(guard_page, page_size, PROT_NONE) != 0) {
        (void)munmap(guarded_mapping, page_size * 2);
        return Fail("mprotect failed for the truncated-buffer guard");
    }
    auto *guarded_partial = guard_page - partial_length;
    std::memcpy(guarded_partial, &partial_record, partial_length);
    if (MsprofReportAdditionalInfo(0xC104, guarded_partial,
                                   static_cast<uint32_t>(partial_length)) != 104 ||
        !VerifyForward(call_index++, kAdditionalCall, 0xC104, guarded_partial,
                       static_cast<uint32_t>(partial_length))) {
        (void)munmap(guarded_mapping, page_size * 2);
        return Fail("truncated AdditionalInfo was not forwarded unchanged");
    }
    (void)munmap(guarded_mapping, page_size * 2);

    MsprofAdditionalInfo batch[3]{};
    FillAdditional(&batch[0], target_hash, kThreadId, 3001, 2, tensor_info_size);
    FillAdditional(&batch[1], other_hash, kOtherThreadId, 3002, 1, tensor_info_size);
    FillAdditional(&batch[2], target_hash, kThreadId, 3003, 1, tensor_info_size);
    const uint32_t batch_length = sizeof(batch);
    if (MsprofReportBatchAdditionalInfo(0xD101, batch, batch_length) != 105 ||
        !VerifyForward(call_index++, kBatchAdditionalCall, 0xD101, batch, batch_length)) {
        return Fail("batch parameters/status were not forwarded once unchanged");
    }

    MsprofAdditionalInfo other_tensor{};
    if (ReportAdditional(other_hash, 0xC105, kOtherThreadId, 4001, 1, tensor_info_size,
                         &other_tensor) != 104 ||
        !VerifyForward(call_index++, kAdditionalCall, 0xC105, &other_tensor,
                       sizeof(other_tensor))) {
        return Fail("non-target AdditionalInfo parameters/status were not forwarded unchanged");
    }

    if (FakeMsprofCallCount() != call_index || call_index != 10) {
        return Fail("fake SDK did not observe every report exactly once");
    }
    if (FakeMsprofHashCallCount() != 3) {
        return Fail("hash registration APIs were not forwarded exactly once");
    }
    std::printf("RAW_REPORT_CONTRACT_PROBE_OK calls=%d hashes=%d target_hash=%llu\n",
                call_index, FakeMsprofHashCallCount(),
                static_cast<unsigned long long>(target_hash));
    std::fflush(stdout);
    return 0;
}
