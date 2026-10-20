# kernel_hook

通过 `LD_PRELOAD` 劫持 CANN **Host 侧 profiling 上报接口**（`libprofapi.so` 导出的 `MsprofReport*`），
把算子名称、算子类型、调度信息和输入/输出张量参数（format/dtype/shape）记录下来，输出到本地 CSV。

## 原理

CANN 的算子在上报 profiling 数据前，会用 `MsprofGetHashId()` 把算子名、算子类型等
字符串算成 `uint64_t` hash，再通过下面这些接口上报结构体：

| 接口 | 上报内容 | 关键字段 |
|------|----------|----------|
| `MsprofReportApi` | `MsprofApi`（NODE.launch） | `itemId` = 算子名 hash，`beginTime/endTime` |
| `MsprofReportCompactInfo` | `MsprofCompactInfo`（NODE.node_basic_info） | `opName`/`opType` hash、`taskType`、`blockDim`、`opFlag` |
| `MsprofReportAdditionalInfo` | `MsprofAdditionalInfo`（NODE.tensor_info） | `opName` hash + 输入/输出 tensor（format/dtype/shape） |
| `MsprofReportBatchAdditionalInfo` | 上述附加信息的数组 | 同上 |

由于上报的是 hash，本项目**同时** hook `MsprofGetHashId` / `MsprofStr2Id`，
自建 `hash -> 字符串` 字典，从而在 CSV 里还原出算子名。

hook 通过 `dlsym(RTLD_NEXT, ...)` 调用原始实现，**不影响原有 profiling 功能**。

## 目录结构

```
kernel_hook/
├── build.sh                    # 编译脚本（cmake 配置 + 编译）
├── CMakeLists.txt
├── include/profiling/          # vendor 自 runtime 的 aprof_pub.h / prof_common.h
│   ├── aprof_pub.h
│   └── prof_common.h
├── src/msprof_hook.cpp         # 全部 hook 逻辑
├── scripts/run_with_hook.sh    # 一键带 hook 运行（缺库会自动编译）
└── test/                       # 预留测试
```

## 编译

```bash
cd kernel_hook
bash build.sh                 # 默认 Release
bash build.sh clean build     # 先清理再编译
bash build.sh --debug -j 16   # Debug + 16 线程
# 产物：build/libmsprof_hook.so
```

构建工具链需与加载 hook 的目标 Python/系统 loader 兼容；已遇到 Nix g++ 16 构建要求 `GLIBC_2.38`，目标 NPU Python 无法加载的情况。可用系统 GCC 11 在新目录构建：

```bash
cmake -S . -B build/runtime -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build/runtime
```

使用新目录可避免复用已选 Nix 编译器的 `CMakeCache`；此命令不改 `build.sh` 默认或已有构建缓存。

## 使用

### 方式 1：直接 LD_PRELOAD

```bash
source /home/developer/Ascend/cann/bin/setenv.bash
LD_PRELOAD=/path/to/kernel_hook/build/libmsprof_hook.so \
MSPROF_HOOK_OUT=/tmp/hook_out \
MSPROF_HOOK_VERBOSE=1 \
./your_app
```

### 方式 2：msprof 采集

```bash
LD_PRELOAD=/path/to/kernel_hook/build/libmsprof_hook.so \
MSPROF_HOOK_OUT=/tmp/hook_out \
msprof --output=/tmp/prof --ascendcl=on --task-time=on --ge-api=l1 ./your_app
```

### 方式 3：脚本

```bash
./scripts/run_with_hook.sh /tmp/hook_out ./your_app
```

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `MSPROF_HOOK_OUT` | CSV 输出目录 | `./msprof_hook_out` |
| `MSPROF_HOOK_VERBOSE` | 是否把解码记录打到 stderr | 关 |
| `MSPROF_HOOK_RAW_OP` | 精确算子名；启用该算子的原始 Msprof report 审计 | 未设置 |
| `MSPROF_HOOK_RAW_OP_HASH` | 可选十进制或 `0x` 十六进制 op hash；与名称过滤为 OR | 未设置 |

## 输出

### `ops.csv`（算子名 + 参数，核心）

```
thread_id,timestamp,op_name,op_type,task_type,task_type_name,block_dim,op_flag,
input_num,input_formats,input_dtypes,input_shapes,
output_num,output_formats,output_dtypes,output_shapes
```

示例（aclnnAdd）：
```
54015,5903366807275,aclnnAdd_AddAiCore_Add,Add,2,AI_VECTOR_CORE,1,0,
2,ND;ND,FLOAT;FLOAT,8;8,1,ND,FLOAT,8
```

按 `(thread_id, op_name_hash, timestamp)` 聚合：`node_basic_info` 与同一时刻的
多条 `tensor_info` 会合并成一行，得到本次实际报告的算子名与输入/输出参数。

SDK 单包最多包含 5 个 tensor slot、每个 shape 最多 8 个维度；当前 Triton producer 只发一个包，超出容量会截断。
SDK shape 数组未携带 rank，零长度维度与填充零无法无损区分；CSV 还会省略零项。raw JSON 保留八个原值，但不据此推断完整 rank。
Triton producer overlay 通过 [proxy 或 CLI overlay](docs/triton_tensorinfo_producer.md) 启用：它初始化 TensorInfo 长度，并修复受支持 CompactInfo 的 `dataLen = sizeof(MsprofNodeBasicInfo)` 和缺省 `opFlag = 0`。默认值适用于当前生成器的 `AIV`、`AI_CORE`、`MIX_AIC`、`MIX_AIV` 四种 `taskType`；`taskType` 与 `TaskInfo.execMode` 是不同字段，HF32 走单独的执行模式路径，不从 task type 推断。已识别但缺少一个或两个字段的 CompactInfo 只补缺项；未知 task type、冲突赋值或带条件控制流的布局会拒绝修补。CompactInfo 的 report 长度仍为 `sizeof(MsprofCompactInfo)`；仅用 `LD_PRELOAD` 加载 hook 不会修改 producer，SDK ABI、安装的 driver 和 cache 实现也不变。

有 CANN SDK 头文件和 C++ 编译器时，可在 CPU 环境运行 CompactInfo producer 合约检查；它会对 local/physical 两种模板分别验证四种 task type，无需 NPU 或 CANN 运行时库：

```bash
python3 test/test_compact_header_contract.py \
  --cxx /usr/bin/g++ \
  --sdk-include-dir /root/miniconda3/envs/cann910/Ascend/cann-9.1.0/include/experiment/msprof
```

`--sdk-include-dir` 指向包含 `profiling/aprof_pub.h` 的 SDK 目录。要诊断旧 AIV-only guard，可额外传入 `--overlay-script /path/to/baseline-kernel-hook/scripts/triton_tensorinfo_overlay.py`；AIV case 应通过，随后第一个 `AI_CORE` case 会被拒绝，产生预期的非零 RED 结果。详见上方 producer 文档中的边界与审计计数说明。

### dtype 标签

`input_dtypes` 和 `output_dtypes` 按 Msprof tensor report 中的 `ge::DataType` 数值生成。
CANN 9.1 的枚举定义见 SDK 公共头文件 `graph/c_types.h`（`C_DataType` / `C_DT_*`）；ACL 的
`aclDataType` 是其中一部分，两者对应的数据类型使用相同数值。CSV 标签去掉 `C_DT_` 前缀，
例如 `27` 为 `BF16`、`29` 为 `INT4`、`30` 为 `UINT1`，`28` 为 `UNDEFINED`。其余 GE 类型也按
枚举名输出，例如 `14`/`15` 为 `DUAL_SUB_INT8`/`DUAL_SUB_UINT8`，`18`–`26` 为量化、资源、
引用字符串、双输出及变体类型，`31`/`32` 为 `INT2`/`UINT2`，`42` 为 `HIFLOAT4`。

数值 `5` 是保留值，`43` 是 `C_DT_MAX` 边界标记；它们、ACL 的 `-1` 哨兵（report 的无符号字段
中表现为 `0xffffffff`）以及其他未知数值都输出 `UNKNOWN`。更新 hook 后，之后新采集的 CSV 会使用
新的标签映射；已经写出的 CSV 是静态文件，不会被回写或改名。可选的 `raw_reports.jsonl` 保留
report 中原始数字 dtype，不随标签映射变化。

可在有 CANN SDK 头文件和 C++ 编译器的 CPU 环境运行契约检查；无需 NPU 或 CANN 运行时库：

```bash
python3 test/test_dtype_contract.py --sdk-include-dir /path/to/cann/x86_64-linux/include
```

该 include 目录需同时包含 `acl/acl_base.h` 和 `graph/c_types.h`。若已通过环境变量
`ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`CANN_HOME` 或 `ASCEND_HOME` 配置 SDK，也可直接运行
`python3 test/test_dtype_contract.py`。

### format 标签

`input_formats` 和 `output_formats` 的主格式标签按 CANN 9.1 SDK 公共头文件
`graph/c_types.h` 中的 `C_Format`（`C_FORMAT_*`）枚举生成。低 8 位是主格式，和 SDK 的
`GetPrimaryFormat(format)` 一致；例如 HWCN 是 `16`、NDHWC 是 `27`、FRACTAL_NZ 是 `29`。
有效枚举标签去掉 `C_FORMAT_` 前缀输出，包括 `ALL`、`NULL` 等具名项。

`C_FORMAT_RESERVED`（`40`）、`C_FORMAT_END`（`55`）、`C_FORMAT_MAX`（`255`）是保留值或范围
边界，输出 `UNKNOWN`；其他未识别主格式也输出 `UNKNOWN`。当 format 含有高位子格式或 C0 信息时，
CSV 只显示低 8 位主格式；`raw_reports.jsonl` 中的 `format` 保留完整原始数字，例如 packed format
仍可从 raw 记录中检查高位信息。

可在有 CANN SDK 头文件和 C++ 编译器的 CPU 环境运行格式契约检查；无需 NPU 或 CANN 运行时库：

```bash
python3 test/test_format_contract.py --sdk-include-dir /path/to/cann/x86_64-linux/include
```

该 include 目录需包含 `graph/c_types.h`。若已通过环境变量
`ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`CANN_HOME` 或 `ASCEND_HOME` 配置 SDK，也可直接运行
`python3 test/test_format_contract.py`。

### `launch.csv`

```
thread_id,op_name,begin_ns,end_ns,duration_ns
```

### `hash_dic.csv`

```
hash,string
```

所有进程内出现过的 `hash -> 字符串`，便于离线 join 或排查未还原的 hash。

### `raw_reports.jsonl`（可选原始 report 审计）

只有设置 `MSPROF_HOOK_RAW_OP` 或 `MSPROF_HOOK_RAW_OP_HASH` 时才会在
`MSPROF_HOOK_OUT/raw_reports.jsonl` 写入该文件。它逐条记录匹配目标的 hash 注册、API、
compact basic info 和 tensor info；batch tensor info 按原始数组位置逐项记录，不跨 report
join。每行保留 `entrypoint`、单进程捕获顺序 `event_seq`、header 时间/TID/hash、调用长度及
原始五个 tensor slot 的数字 dtype/format/type 和八维 shape。未知 hash 不按相似名称或邻近
记录推断归属；可用显式 `MSPROF_HOOK_RAW_OP_HASH` 精确筛选 hash。

TensorInfo 的 `data_len` 原样保留为声明值；raw audit 读取时只以回调实际 `length` 和固定
payload 容量为界，因此可检查声明长度与可读 payload 不一致的报告。`legacy_accept` 和
`legacy_reject_reason` 显示原有 CSV collector 是否接受该记录，便于区分“原始报告含张量信息”
与“既有 CSV 解码器接受了报告”。该审计不改原有 CSV 的解析/聚合，也不改 SDK 调用：输入指针、
长度和 flags 原样转发一次，并返回原接口状态。审计文件打开或写入失败时只关闭 raw audit，
不影响原 collector 或 SDK 结果。未启用时不会创建该文件。

## 实测样例（Kimi-K3 mini 微调）

```bash
cd /path/to/MindSpeed-MM
source /home/developer/Ascend/cann/bin/setenv.bash
export LD_PRELOAD=/path/to/kernel_hook/build/libmsprof_hook.so
export MSPROF_HOOK_OUT=/tmp/kimik3_ops
NPROC_PER_NODE=1 msprof --output=/tmp/msprof_out \
    --ascendcl=on --task-time=on --runtime-api=on --ge-api=l1 \
    bash examples/kimi_k3/finetune_kimik3_mini.sh
```

结果（1 个 iteration）：25,723 次算子执行 / 173 个去重算子 / 543 条 hash 字典。
样例产物见 [`examples/kimik3/`](examples/kimik3/)：
- `op_summary.csv`：按算子聚合的「算子名 + 类型 + task + blockDim + 输入/输出 shape/dtype + 次数」
- `hash_dic.csv`：hash → 字符串

## 注意事项

1. **必须先开启 profiling**。这些上报接口本身是被 profiling 开关（`opProfilingSwitch.reportFlag`
   等）保护的，未开启时上层根本不会调用；hook 也就不会触发。开启方式：`msprof` 命令行、
   `aclprofInit/Start` API、`acl.json` 或 `PROFILER_SAMPLECONFIG`。
2. **hash 需在同一进程内被计算**。Host 侧上报的 hash 都是进程内 `MsprofGetHashId` 算的，
   hook 能捕获；若出现未还原的 hash，会以 `0x...` 形式输出，可在 `hash_dic.csv` 里查。
3. **变长 additional info**：目前只解码标准 256B 的 `tensor_info`；长度 > 256 的
   variable block（如 `capture_op_info`）未做解析。
4. **性能**：所有记录在内存中聚合，退出时（`MsprofFinalize` 或析构）统一落盘；
   高频算子会占用一定内存。若需长时间采集，可改成边收边写或加周期 flush。
5. **多进程**：每个被注入的进程各写一份 CSV 到同一目录，会互相覆盖。多进程场景请给每个
   进程设置不同的 `MSPROF_HOOK_OUT`（例如带 `$PID`）。

## 已修复/可扩展

- 当前已覆盖：`MsprofReportApi`、`MsprofReportEvent`、`MsprofReportCompactInfo`、
  `MsprofReportAdditionalInfo`、`MsprofReportBatchAdditionalInfo`、`MsprofGetHashId`、
  `MsprofStr2Id`、`MsprofFinalize`。
- 可扩展：`MsprofReportData`（模块回调路径）、`MsprofRegTypeInfo`（type 名表）、
  `MsprofReportAdditionalInfo` 的 `node_attr_info`（算子属性）。
