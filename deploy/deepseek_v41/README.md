# DeepSeek V4.1 Flash deployment overlay

This directory is the complete image-pinned source for an **isolated original
grid-A experiment with only the extra token-slot/count correction**. It is not
the shared production deployment branch. This candidate is prepared on
`odin/pro6k-grida-only-extra-fix-1107-5`; deployments must pin its final immutable
commit and bundle, never a floating branch. The original grid-A parser/environ
guards are retained, while the independent logprob consumer change from
`04d1bda93308b4aeb6d0c625033f650da6b3be16` is deliberately excluded.
Only the SM120 attention wrapper differs from the reconstructed original grid-A
runtime. No observer, alternative Triton selector or minimal-ablation switches
are included. Four earlier synthetic GPU cases support this wrapper correction;
the new original-grid-A versus fixed online-B300 ABC round has not run.
Do **not** `pip install` this checkout or serve from its generic `python/` tree.

## Baseline and contents

- Runtime image: `docker.io/lmsysorg/sglang@sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349`.
- Platform manifest: `sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5` (`linux/amd64`).
- Image config: `sha256:a4cfe088e1c40c1f10b08e716ace45f62c82e050ffaef983fd783ca52f3d870c`.
- Complete package source layer: `sha256:52f078c87b74c74ff1f1a469469a62031ce00c710700b2f5c278ad791961ab08`, 12,544,141 compressed bytes.
- Layer prefix: `sgl-workspace/sglang/python/sglang/`.
- The image's Git/build/overlay commit labels are **unknown**. The layer, not a
  guessed Git commit, is the exact baseline. The later compile layer adds only
  `.pyc` files, not replacement source.

`runtime-overlay/sglang/` contains the entire recovered package: 4,752 regular
files plus its internal `.clang-format` symlink, including headers, kernels,
assets and third-party notices. Common fixes are already in that tree.
`profiles/` contains complete replacement files for the optional sm120 changes.
No native binary, model weight, production request, response, observer, or new
sampling/tie-break experiment is added by this overlay.

`manifest.json` records every baseline file hash, symlink, executable-file list,
final common file hash and profile file hash. The baseline tree digest is SHA256
of sorted UTF-8 `relative_path + NUL + hex_sha256 + LF` records for regular files;
symlinks are recorded separately. Keep all license and attribution files.

The historical `2bb25dc18bc27321fb116bbefcaddaf13c1c1754` tree is **not** this
baseline; only part of it matched the image. In particular, this package retains
the actual image's draft worker, rather than replacing it with the older fork
copy. The old CPFS overlay did not contain guards at these batch boundaries;
the identity of the previously reported private incident fix remains unknown.
This is not a claim that the issue was never fixed before.

## Changes relative to the immutable image

| Location | Change | Origin / scope |
| --- | --- | --- |
| `srt/managers/scheduler_components/batch_result_processor.py` | No change from the immutable image / original grid A | Deliberately excludes the separate Tensor/list logprob guard; original consumer SHA734bb2f4 remains fixed, and no logprob request is part of this experiment |
| `srt/function_call/function_call_parser.py` | Bound held streaming text, flush it and latch parsing off after the guard fires | Existing B300 v5 / PRO6K v6 parser guard, byte-identical between those captured deployments |
| `srt/environ.py` | Declare `SGLANG_TOOL_CALL_BUFFER_LIMIT` | Existing guard, no upstream fields removed |
| `profiles/sm120/kernels/ops/attention/flash_mla_sm120.py` | Extra-pool footer reblock and separate buffer; preserve token-slot indices and candidate counts | Inherited page-layout adaptation plus a minimal candidate correction: remove incorrect index expansion and length multiplication; no numerical/quality acceptance claimed |
| `profiles/sm120-shim/srt/layers/attention/deepseek_v4_backend.py` | Strict decode-indexer metadata shim | Existing optional shim; only selected with explicit `--sm120-shim on` |

There are exactly two changed regular files in the common tree: the original
grid-A parser/environ guards. All other common bytes, including the consumer,
are the immutable-image bytes. Profile overrides never replace these files.

Relative to the image, grid A already had the necessary extra-pool footer
reblocking and independent buffer. This candidate changes only the incorrect
slot expansion and candidate-length multiplication in that existing wrapper.
FlashInfer 0.6.18 consumes linear token slots and candidate counts; footer
reblocking preserves those units. The splitter and its independent extra buffer
remain. The known independent logprob defect is intentionally not fixed here;
this experiment is not a recommended general-purpose production overlay.

Original grid-A expected tree, statically reconstructed from the immutable image
and archived entrypoint/patches (not a historical full-process tree capture):
`fa272e98e0b59f92f0d547058c69a0c0fb0b72acdc41a9eba073607a17036d32`.
The only-extra SM120/shim-off expected tree is
`ddd83ee3b207c7b8ccc2647e861afefeac51ed36205f7d5bb36c22b4610f5ef5`.
Serving DSPARK/cache/graphs/overlap/SWA settings are restored from original grid
A in the deployment manifest, not from the later minimal condition. Fresh
process startup does not restore the original GPU/KV/RNG state.

## Runtime compatibility and profiles

Keep **all installed wheels and native libraries from the pinned image**. Its
build history specifies Python 3.12 / CUDA 13.0.3, and the bundled kernel source
identifies `0.4.6.post1`; these are provenance, not permission to reinstall an
arbitrary matching wheel. No native build or ABI upgrade is part of this change.
An image reference supplied to the installer is a deployment assertion, not an
in-container proof of the running OCI digest. The deployment controller must pin
and verify the actual image ID as well.

| Profile | Common tree | Extra-pool patch | Decode-indexer shim |
| --- | --- | --- | --- |
| `b300` | Complete | No | No |
| `sm120 --sm120-shim off` | Complete | Yes | No |
| `sm120 --sm120-shim on` | Complete | Yes | Yes |

PRO5000 and PRO6000 share the sm120 source profile, not automatically the same
serving settings. PRO5000 live compatibility has **not** been verified. The
captured PRO6K experimental deployment explicitly skipped the shim. Preserve
that choice unless a separately approved experiment changes it; there is no
implicit shim default.

TP/EP, memory fraction, context, DSPARK, cache policy, parser selection, model
paths, effort fallback and resource settings remain in `jetd1/llm-deploy`.
The overlay does not alter `/v1/messages`, reasoning/default semantics, request
parameters, model weights or routing. Those source files are retained and
included in whole-package verification.

## One installation order

Run only during an approved container startup, **before any serving process or
SGLang import**. Never install into a running worker. Existing cache setup,
weight checks and serving arguments remain the deployment entrypoint's job.

1. Pin the runtime image and overlay Git commit; distribute the complete bundle.
2. Replace the old parser/environ copy and sm120 append/rewrite steps with this
   installer. Do **not** run old CM patches after it, or prepend an old source
   tree through `PYTHONPATH`.
3. Install the complete common tree, followed by the selected sm120 overrides.
   The installer first validates the entire package against baseline/final
   hashes, rejecting an unknown file, changed draft, stale source or wrong
   image assertion before writing. Identical repeated installation is allowed;
   changing an already-installed architecture profile is rejected when its
   bytes do not match the new profile. Use a fresh container for profile changes.
4. Run final `verify` with exactly the same options, then immediately `exec` the
   original server command. Do not import SGLang or apply another patch between
   verification and `exec`.
5. Record overlay commit, bundle SHA256, manifest SHA256, profile/shim choice,
   actual image ID and installation report with the deployment record.

Example (paths are discovered or supplied, never baked into source):

```bash
set -euo pipefail
: "${OVERLAY_DIR:?complete extracted bundle directory}"
: "${RUNTIME_IMAGE_DIGEST:?digest pinned by the deployment}"
PKG=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("sglang").origin).parent)')
python3 "$OVERLAY_DIR/overlay.py" install --package-root "$PKG" \
  --image-digest "$RUNTIME_IMAGE_DIGEST" --profile b300
python3 "$OVERLAY_DIR/overlay.py" verify --package-root "$PKG" \
  --image-digest "$RUNTIME_IMAGE_DIGEST" --profile b300
# Immediately exec the existing, approved server command here.
```

For sm120, replace `--profile b300` in **both** commands with
`--profile sm120 --sm120-shim off` (or explicitly `on`). The installer emits a
JSON report with version, manifest SHA, source layer, image assertion and changed
file hashes. It does not start a server or modify environment defaults.

## Package and CPU validation

From this directory, using Python 3.12 and an unused output filename:

```bash
python3 overlay.py pack --output deepseek-overlay.tar.gz
sha256sum deepseek-overlay.tar.gz manifest.json
```

The bundle contains the complete runtime tree, profiles, manifest, installer,
licenses and focused tests. File order, ownership, timestamps and gzip metadata
are deterministic; an existing output file is never overwritten.

Run the focused regression against the complete common tree, with the pinned
runtime's test dependencies or a separately recorded CPU environment:

```bash
PYTHONPATH=runtime-overlay python3 -m pytest -q tests/test_overlay.py tests/test_sm120_extra_slots.py
```

The package tests require the original consumer bytes to be retained and reject
late consumer drift. Installer tests also cover fail-before-write, explicit
profile selection, cache invalidation on changed files and complete reproducible
archives. The historical `tests/test_logprob_rows.py` remains in the checkout as
evidence of the intentionally uncorrected original consumer contract, but is not
shipped in this experiment bundle or represented as passing here. Do not request
logprobs to probe that unrelated defect during the original-payload comparison.

The candidate's additional CPU regression executes the actual prefill adapter
function from this profile with real torch index/length operations. Only the
GPU page-copy and native attention boundary are replaced; it is not a handwritten
translation of the index formula. PBS128/256, absent/present candidate lengths,
negative padding and a final legal pool slot fail in four subcases before the
minimal change and pass after it. It also checks the independent extra buffer key.
The earlier candidate's deployment tests passed8 with19 subtests; that result
includes a logprob guard not present in this experiment and is not its test count.
CPU boundary tests do
not execute the splitter kernels or native attention. A first numerical probe
attempt expired before usable container/output evidence, so its infrastructure
failure supplies no GPU correctness result. A separately approved child process
then completed the frozen four-case single-GPU numerical comparison described
below. Full-model and long-task gates remain pending; no candidate server or
production deployment is implied.

Single-GPU synthetic validation on 2026-11-07 used the pinned image's PyTorch
2.13.0+cu130 / CUDA13.0, FlashInfer0.6.18 and AOT binary SHA
`a17d69ee8024686be65cbfcc22e0f01a35475ec1eb8cc04fe6bc8170ccd07764`.
Q65/H8/D512, mainPBS128/topk128, extraPBS128 or256/topk512, negative padding,
per-query lengths and absent/present attention sink were held fixed, with
seed20261107 and atol=rtol0.05 declared before execution. All original and
rewritten indices were allocation-safe. Each case called the original adapter
and minimal candidate once, eight native calls total, with an independent
FP32 reference dequantizing and gathering the original packed source pages.

| Case | Candidate vs intended max abs | Original vs intended max abs | Original vs rewritten-input reference max abs |
| --- | ---: | ---: | ---: |
| PBS128 / no sink | 0.004896 | 0.765049 | 0.004430 |
| PBS128 / sink | 0.002666 | 0.630166 | 0.001552 |
| PBS256 / no sink | 0.007381 | 0.995464 | 0.004722 |
| PBS256 / sink | 0.002713 | 0.850774 | 0.002841 |

All candidate/intended and original/rewritten-reference elements satisfied the
frozen tolerance; all four original/intended outputs failed it. The tested
candidate prefill function AST matches this profile's function. Complete probe
wall time was4.6113s, including its initialization and first kernel compilation;
Torch allocator peak650657792B excludes driver/native allocations. Timing is
not a benchmark: original/candidate call order and compilation history differ.
The original service's process identity, health, zero request/queue gauges and
all GPU memory readings were checked restored afterward. Its GPU was used by
the probe and is not an untouched-scene control.

This establishes the numerical token-slot/count mismatch and correction for
these synthetic inputs, not that every CUDA path, distributed execution, real
activation distribution or historical long-task quality issue is repaired.
No CPU out-of-bounds fixture was sent to the GPU. Raw synthetic inputs, per-mode
outputs and references were saved before advancing; result SHA256 is
`247f172bafd80815b8fa50d719dd150e45a90fc1e12a53c9bfba5ae23ad625a4`.

The following is historical validation of the separate complete common/logprob
overlay, **not the test result of this only-extra experiment**. It used Python
3.12.3 / PyTorch 2.15.0+cpu on 2026-11-07,
with CUDA unavailable (not the pinned image's GPU environment):

- Unmodified image consumer: 10 failing selected-token subcases, with the
  expected `AttributeError: 'list' object has no attribute 'tolist'`.
- Complete overlay tests: 7 passed, 15 subtests passed, including the reviewed
  extra-model-symlink rejection (verified red before the installer fix).
- Neighboring hidden-state, speculative-grammar, fast-input/chunk-stitching
  logprob and output-streamer fixtures from `2bb25dc18bc27321fb116bbefcaddaf13c1c1754`:
  19 passed, 7 CUDA-only skips, 14 subtests passed against this complete runtime.
- `compileall` of SRT, kernels and profiles passed. New installer/tests pass Ruff
  checks/formatting. Full-import `git diff --check` reports inherited whitespace
  in seven byte-identical image files; those bytes are intentionally retained.
- Independent source/runtime and packaging reviews completed; one installer P2
  (extra symlink not rejected) was reproduced, fixed and re-reviewed closed.

CPU tests do not validate CUDA D2H, kernel execution, GPU memory, distributed
scheduling, DSPARK serving, long-input quality or native Messages end-to-end.
Before production rollout, separately validate the approved single-worker
experiment, then drain/switch/check one B300 arm at a time; stop on failure.

## Rollback

Keep the previous image digest, original entrypoint/CM, environment and exact
serving configuration in `llm-deploy`. Roll back through a separately approved
container replacement using those original artifacts, not by mixing old source
files into an already-running process or changing only one Python file. The
installer deliberately provides no in-process undo. New startup loses GPU/KV/RNG
state; retain experiment evidence before a controlled restart.
