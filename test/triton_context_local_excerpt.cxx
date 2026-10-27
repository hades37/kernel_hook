/*
 * ContextInfo source excerpt from the installed local Triton driver:
 * /root/miniconda3/envs/npu/lib/python3.12/site-packages/triton/backends/
 * ascend/driver.py, cpp_msprof_call_after_launch at lines 744-760.
 * The driver emits this snippet at two generated report sites; both copies
 * are retained below. Only the Python f-string expansions are placeholders:
 * __IS_MIX_TASK_TYPE__ is "true" for MIX task types, and the task type token
 * in the neighboring CompactInfo fixture is supplied by the probe.
 */
if (__IS_MIX_TASK_TYPE__ > 0) {
  MsprofAdditionalInfo info;
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  info.threadId = threadId;
  info.timeStamp = endTime;
  MsprofContextIdInfo ctxId;
  ctxId.opName = opNameHashID;
  ctxId.ctxIdNum = 1;
  for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
    ctxId.ctxIds[i] = i;
  }
  size_t copyLen = sizeof(MsprofContextIdInfo);
  if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
    copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
  }
  memcpy(info.data, &ctxId, copyLen);
  MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
}
/* NEXT_CONTEXT_REPORT_SITE */
if (__IS_MIX_TASK_TYPE__ > 0) {
  MsprofAdditionalInfo info;
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  info.threadId = threadId;
  info.timeStamp = endTime;
  MsprofContextIdInfo ctxId;
  ctxId.opName = opNameHashID;
  ctxId.ctxIdNum = 1;
  for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
    ctxId.ctxIds[i] = i;
  }
  size_t copyLen = sizeof(MsprofContextIdInfo);
  if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
    copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
  }
  memcpy(info.data, &ctxId, copyLen);
  MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
}
