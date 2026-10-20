# Triton TensorInfo producer overlay

The Ascend Triton launcher builds `MsprofAdditionalInfo tensorInfo` without value-initializing it or setting `dataLen`. The profiling hook receives the fixed-size additional-info payload, but rejects the TensorInfo report when its declared length is zero. This process-local overlay repairs both generated TensorInfo blocks before the native launcher factory sees the C++ source.

The overlay wraps `make_npu_launcher_stub` in the already registered Ascend driver's globals. It does not import a second `triton.backends.ascend.driver` module or instantiate the driver to find that namespace. Both supported driver layouts use this factory: one builds source through `generate_npu_wrapper_src`, the other through `make_launcher`. The native factory still receives the source and performs its existing cache lookup, debug dump, and compile. The cache key naturally includes the changed source; the overlay leaves old cache entries and installed driver files alone.

Every new launcher factory call is checked. A source with exactly two uninitialized TensorInfo blocks gets value-initialized declarations and `dataLen = sizeof(MsprofTensorInfo)` in both blocks. A source with exactly two correctly initialized blocks is passed through unchanged and records `replacement_count: 0`. Partial or unknown layouts fail closed. Other fields and report calls pass through unchanged, and there is no operator-name selector.

The CPU-only contract tests use fake registered backends and do not need Triton, CANN, or an NPU. From `kernel_hook/`, run:

```bash
python3 test/test_triton_tensorinfo_overlay.py
```

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

The change only initializes the two TensorInfo records. CompactInfo and context-info headers, dtype-index mapping, collector gates, and kernel report counts remain outside this repair. To disable it, restore the prior `KIMIK3_PYTHON` value or stop pointing it at this proxy; no installed package or persistent cache needs rollback.
