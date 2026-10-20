# Triton producer metadata overlay

The Ascend Triton launcher builds `MsprofAdditionalInfo tensorInfo` without value-initializing it or setting `dataLen`. The profiling hook receives the fixed-size additional-info payload, but rejects the TensorInfo report when its declared length is zero. This process-local overlay repairs both generated TensorInfo blocks before the native launcher factory sees the C++ source.

The same factory boundary now also repairs supported `MsprofCompactInfo` blocks. If a block has no `dataLen`, the overlay sets it to `sizeof(MsprofNodeBasicInfo)`—the payload size, not the enclosing `sizeof(MsprofCompactInfo)` report size. If `opFlag` is absent, it sets zero for the four task types emitted by the current generator: `MSPROF_GE_TASK_TYPE_AIV`, `MSPROF_GE_TASK_TYPE_AI_CORE`, `MSPROF_GE_TASK_TYPE_MIX_AIC`, and `MSPROF_GE_TASK_TYPE_MIX_AIV`. This `taskType` is distinct from `TaskInfo.execMode`: HF32 is carried through that separate execution-mode path and is not inferred from the task type. An existing recognized `opFlag` assignment is preserved, and a recognized block with either or both supported fields missing gets only the missing field or fields filled. Unknown task types, conflicting assignments, and conditional block shapes fail closed. The overlay does not zero-initialize the whole CompactInfo object, change the report call or its length, alter the SDK/ABI, or modify the installed driver or cache implementation.

The overlay wraps `make_npu_launcher_stub` in the already registered Ascend driver's globals. It does not import a second `triton.backends.ascend.driver` module or instantiate the driver to find that namespace. Both supported driver layouts use this factory: one builds source through `generate_npu_wrapper_src`, the other through `make_launcher`. The native factory still receives the source and performs its existing cache lookup, debug dump, and compile. The cache key naturally includes the changed source; the overlay leaves old cache entries and installed driver files alone.

Every new launcher factory call is checked. A source with exactly two uninitialized TensorInfo blocks gets value-initialized declarations and `dataLen = sizeof(MsprofTensorInfo)` in both blocks. A source with exactly two correctly initialized TensorInfo blocks is passed through unchanged; partial or unknown TensorInfo layouts fail closed. CompactInfo `dataLen` and `opFlag` repairs are counted separately from TensorInfo replacements. Per-call `replacement_count` and manifest `replacement_total` continue to count only TensorInfo changes; `compact_replacement_count` and `compact_replacement_total` count CompactInfo blocks changed. `repair_activated` is true only when either repair total is positive, so a factory call alone does not establish activation. For CompactInfo, the recognized block shape can be repaired when either or both supported fields are missing, and only missing fields are added; unknown, conflicting, or conditional block layouts fail closed. Other fields and report calls pass through unchanged, and there is no operator-name selector.

The Python CPU contract tests use fake registered backends and do not need Triton, CANN, or an NPU. From `kernel_hook/`, run:

```bash
python3 test/test_triton_tensorinfo_overlay.py
```

The CompactInfo SDK probe substitutes each of the four generator task types into both captured local and physical source excerpts, one source field at a time. Through the registered factory seam it compiles each result against the actual CANN SDK header, then calls the original report path through a CPU stub. It checks the selected task type, default `opFlag`, SDK field layout, payload length, and report arguments without requiring an NPU or CANN runtime library. Point it at the SDK directory containing `profiling/aprof_pub.h`:

```bash
python3 test/test_compact_header_contract.py \
  --cxx /usr/bin/g++ \
  --sdk-include-dir /root/miniconda3/envs/cann910/Ascend/cann-9.1.0/include/experiment/msprof
```

To check the earlier AIV-only guard, pass its overlay snapshot with `--overlay-script`. The AIV cases should pass, then the first `AI_CORE` case should be rejected; that nonzero exit is the expected RED signal for the expanded matrix:

```bash
python3 test/test_compact_header_contract.py \
  --cxx /usr/bin/g++ \
  --sdk-include-dir /root/miniconda3/envs/cann910/Ascend/cann-9.1.0/include/experiment/msprof \
  --overlay-script /path/to/baseline-kernel-hook/scripts/triton_tensorinfo_overlay.py
```

This CPU SDK probe checks SDK field layout, payload length, and preservation of the report call; it does not validate training behavior or numerical equivalence. Training results are recorded separately.

## Run through the Python proxy

Point the training runner at the proxy and set the real interpreter and a fresh audit directory:

```bash
export KIMIK3_PYTHON=/root/npu-env/kernel_hook/scripts/triton_tensorinfo_python_proxy.sh
export KIMIK3_PRODUCER_REAL_PYTHON=/absolute/path/to/npu-env/bin/python
export KIMIK3_TENSORINFO_AUDIT_DIR="$RUN/producer-audit"
```

`KIMIK3_PRODUCER_OUT` is accepted as an alternate audit-directory variable. If neither audit variable is set, the proxy uses `$KIMIK3_FAKE_ARTIFACT_DIR/producer-audit` when that directory variable exists. The selected path must be absolute and fresh or empty.

The proxy intercepts only `mindspeed_mm/fsdp/train/trainer.py`, either as that exact relative path or as the same file under `MINDSPEED_ROOT`. Python preflights, `-c`, `-m`, stdin, and other scripts are executed by `KIMIK3_PRODUCER_REAL_PYTHON` unchanged. The overlay runs the original trainer with its original arguments, working directory, environment, `sys.argv`, and trainer directory at `sys.path[0]`.

The overlay can also wrap a Python script directly:

```bash
/absolute/path/to/python \
  /root/npu-env/kernel_hook/scripts/triton_tensorinfo_overlay.py \
  --audit-dir "$RUN/producer-audit" -- path/to/script.py arg1 arg2
```

The registered Ascend backend must be available in that process. When the script returns or raises, the original factory binding is restored in `finally`. Do not remove the patch concurrently with in-flight launcher calls; the proxy keeps it installed until the wrapped script finishes.

## Audit and bounds

`run_manifest.json` records the registered driver namespace and source hash, launcher call totals, exit status, and whether any launcher factory call activated the overlay. A successful script with zero factory calls is marked `no_launcher_calls`; it does not establish that the repair ran. `launcher_calls.jsonl` records source hashes, original and effective cache keys, replacement count, returned shared-object path and hash, and per-call status. The audit stores hashes and metadata, not full generated C++.

The SDK payload remains five tensor slots by eight dimensions, and the current Triton producer clips a report to that fixed capacity. This overlay retains those limits and does not add report packets or change the ABI. The KDA four-slot and DHU five-slot observations describe the slots already reported for those captures; they do not imply that every kernel argument is represented when a kernel has more than five tensors. Although CANN's Opbase producer can emit multiple five-slot reports under the same op and summary identifiers, the SDK does not define a generic merge guarantee and this hook appends reports as received.

The SDK's five-slot/eight-dimension limits and the producer's current single-report behavior remain unchanged. Context-info headers, dtype-index mapping, collector gates, and kernel report counts remain outside this repair. To disable it, restore the prior `KIMIK3_PYTHON` value or stop pointing it at this proxy; no installed package or persistent cache needs rollback. The native factory sees the transformed wrapper source, so its normal source-derived cache identity applies while existing cache entries remain untouched.
