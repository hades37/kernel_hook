#include "profiling/aprof_pub.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" int32_t FakeMsprofCallCount();

namespace {
constexpr char kOpName[] = "ExitFlushProbeOp";
constexpr char kOpType[] = "ExitFlushProbeType";
constexpr uint32_t kThreadId = 47;
constexpr uint64_t kTimestamp = UINT64_C(987654321);

int fail(const char *message)
{
    std::fprintf(stderr, "msprof_exit_flush_probe: %s\n", message);
    return 1;
}
}  // namespace

int main()
{
    const uint64_t op_name = MsprofGetHashId(kOpName, std::strlen(kOpName));
    const uint64_t op_type = MsprofStr2Id(kOpType, std::strlen(kOpType));
    if (op_name == 0 || op_type == 0) return fail("hash API returned zero");

    MsprofApi api{};
    api.level = MSPROF_REPORT_NODE_LEVEL;
    api.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    api.threadId = kThreadId;
    api.beginTime = 1000;
    api.endTime = 1037;
    api.itemId = op_name;
    if (MsprofReportApi(0, &api) != 101) return fail("API report was not forwarded");

    MsprofCompactInfo compact{};
    compact.level = MSPROF_REPORT_NODE_LEVEL;
    compact.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    compact.threadId = kThreadId;
    compact.dataLen = sizeof(MsprofNodeBasicInfo);
    compact.timeStamp = kTimestamp;
    compact.data.nodeBasicInfo.opName = op_name;
    compact.data.nodeBasicInfo.opType = op_type;
    compact.data.nodeBasicInfo.taskType = 2;
    compact.data.nodeBasicInfo.blockDim = 7;
    compact.data.nodeBasicInfo.opFlag = 0;
    if (MsprofReportCompactInfo(0, &compact, sizeof(compact)) != 102)
        return fail("compact report was not forwarded");

    MsprofAdditionalInfo additional{};
    additional.level = MSPROF_REPORT_NODE_LEVEL;
    additional.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    additional.threadId = kThreadId;
    additional.timeStamp = kTimestamp;
    auto *tensor_info = reinterpret_cast<MsprofTensorInfo *>(additional.data);
    tensor_info->opName = op_name;
    tensor_info->tensorNum = 2;
    tensor_info->tensorData[0].tensorType = MSPROF_GE_TENSOR_TYPE_INPUT;
    tensor_info->tensorData[0].format = 2;
    tensor_info->tensorData[0].dataType = 0;
    tensor_info->tensorData[0].shape[0] = 2;
    tensor_info->tensorData[0].shape[1] = 3;
    tensor_info->tensorData[1].tensorType = MSPROF_GE_TENSOR_TYPE_OUTPUT;
    tensor_info->tensorData[1].format = 2;
    tensor_info->tensorData[1].dataType = 3;
    tensor_info->tensorData[1].shape[0] = 2;
    tensor_info->tensorData[1].shape[1] = 3;
    additional.dataLen = sizeof(MsprofTensorInfo);
    if (additional.dataLen > sizeof(additional.data))
        return fail("tensor fixture does not fit the public report structure");
    if (MsprofReportAdditionalInfo(0, &additional, sizeof(additional)) != 103)
        return fail("additional report was not forwarded");

    if (FakeMsprofCallCount() != 5) return fail("hook did not forward every API call");
    std::puts("MSPROF_EXIT_FLUSH_PROBE_OK records=1 hash_names=2 reports=3");
    std::fflush(stdout);
    // Do not call MsprofFinalize: process-exit DSO destruction must flush.
    return 0;
}
