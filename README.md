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
多条 `tensor_info` 会合并成一行，得到完整的算子名 + 输入/输出参数。

### `launch.csv`

```
thread_id,op_name,begin_ns,end_ns,duration_ns
```

### `hash_dic.csv`

```
hash,string
```

所有进程内出现过的 `hash -> 字符串`，便于离线 join 或排查未还原的 hash。

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
