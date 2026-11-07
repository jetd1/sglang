# DeepSeek V4.1 Flash deployment overlay

This directory is the deployment source, independent of the repository's generic
`python/` tree. This revision is an **experimental sm120 candidate**, based on the
validated `deploy/deepseek-v4.1-flash-overlay` baseline commit
`dd3618209ea01705d07f36ecf56b24a2f536743c`. Package the candidate from its own
`odin/deepseek-sm120-extra-slots-fix-3` branch and pin an immutable commit;
the shared deployment branch and production references have not been advanced.
Four frozen single-GPU synthetic prefill cases satisfy their numerical criteria.
A subsequent isolated full-model original-ABC round on immutable commit
`04d1bda93308b4aeb6d0c625033f650da6b3be16` was transport-complete but scored
2 reasonable next actions / 1 unreasonable output. Long-task quality acceptance
remains **unpassed**; the numerical contract fix is not a complete semantic fix.
This documentation follow-up does not change the tested runtime/profile bytes.
Do **not** `pip install`
this checkout, put its generic `python/` on the serving `PYTHONPATH`,
or use an older DeepSeek branch as a replacement for this runtime.

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
| `srt/managers/scheduler_components/batch_result_processor.py` | Convert selected-token logprob rows only when they are Tensors, in both prefill and decode | Reuses the existing normalization convention and CPU regression from `c3e375fe6a8e7c5932f563a8ae81cdff030d548c`; all other consumer bytes retained |
| `srt/function_call/function_call_parser.py` | Bound held streaming text, flush it and latch parsing off after the guard fires | Existing B300 v5 / PRO6K v6 parser guard, byte-identical between those captured deployments |
| `srt/environ.py` | Declare `SGLANG_TOOL_CALL_BUFFER_LIMIT` | Existing guard, no upstream fields removed |
| `profiles/sm120/kernels/ops/attention/flash_mla_sm120.py` | Extra-pool footer reblock and separate buffer; preserve token-slot indices and candidate counts | Inherited page-layout adaptation plus a minimal candidate correction: remove incorrect index expansion and length multiplication; synthetic numerical criteria pass, full long-task quality acceptance does not |
| `profiles/sm120-shim/srt/layers/attention/deepseek_v4_backend.py` | Strict decode-indexer metadata shim | Existing optional shim; only selected with explicit `--sm120-shim on` |

There are exactly three changed regular files in the common tree. The remaining
baseline package files are retained byte-for-byte. Profile overrides never
replace the consumer, parser, or environment file.

The logprob regression proves a consumer contract defect, not the identity of
the live failing batch's list rows and **not** the cause or cure of PRO6K output
quality differences. The inherited decode-indexer shim is unchanged. The sm120
candidate corrects a separate source/native token-slot/count contract, not the
logprob defect: FlashInfer 0.6.18 consumes linear token slots and candidate counts,
while footer reblocking preserves those units. Expanding a slot by the page
split ratio or multiplying its length changes the gathered keys. The footer
splitter and its independent extra buffer remain necessary and are retained.
This changes only the sm120 profile; the complete common/B300 source is unchanged.

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
PYTHONPATH=runtime-overlay python3 -m pytest -q tests/test_logprob_rows.py tests/test_overlay.py tests/test_sm120_extra_slots.py
```

The selected-token regression uses the real producer, output object and both
normalization helpers. It covers mixed optional rows, all-Tensor rows, already
materialized lists, order, selected-token indices and list-row identity. It
fails on the unmodified image consumer. Installer tests cover fail-before-write,
profile selection and full-bundle reproducibility/completeness.

The candidate's additional CPU regression executes the actual prefill adapter
function from this profile with real torch index/length operations. Only the
GPU page-copy and native attention boundary are replaced; it is not a handwritten
translation of the index formula. PBS128/256, absent/present candidate lengths,
negative padding and a final legal pool slot fail in four subcases before the
minimal change and pass after it. It also checks the independent extra buffer key.
Candidate deployment tests: 8 passed, 19 subtests passed. CPU boundary tests do
not execute the splitter kernels or native attention. A first numerical probe
attempt expired before usable container/output evidence, so its infrastructure
failure supplies no GPU correctness result. A separately approved child process
then completed the frozen four-case single-GPU numerical comparison described
below. A later isolated candidate server passed full-source/default-FlashInfer
and original-minimal admission and completed one original ABC round; the
long-task semantic gate still failed as described below. No production
promotion is implied.

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

## Separate full-model observation, not quality acceptance

On 2026-11-07, one isolated SM120 server using immutable candidate commit
`04d1bda93308b4aeb6d0c625033f650da6b3be16` completed one original A→B→C round.
Actual whole-tree/profile/manifest verification, runtime image identity,
shim-off/default-FlashInfer selection, original minimal flags and stable process
identity passed before requests. DSPARK, HiCache/radix reuse, overlap and CUDA
graphs were disabled; the existing FP4 indexer/MoE and model sampling defaults
were retained. The original request fields were not overridden; only the
necessary served-model mapping changed. No returned tool was executed.

| Original sample | Input tokens | Output tokens | First-content / E2E seconds | Transport | Bounded visible-output review |
| --- | ---: | ---: | --- | --- | --- |
| A | 344839 | 125 | 96.035 / 112.026 | Complete, HTTP200/tool_use/message_stop | Reasonable, limited grounded self-authored report-title Edit |
| B | 239726 | 2269 | 45.051 / 337.078 | Complete, HTTP200/tool_use/message_stop | Unreasonable: a relevant issue proposal contains two unsupported local factual claims |
| C | 241763 | 432 | 45.294 / 101.237 | Complete, HTTP200/tool_use/message_stop | Reasonable, limited read-only migration/lock inspection proposal |

The independent review used each original task's own history, factual evidence,
scope and tool schema, not another platform or GPU correctness as gold. Hidden
thinking was not used for scoring. Reasonable means a usable next action, not
completed work. First-content may be a thinking/tool fragment, not visible prose;
usage includes generated tokens beyond visible text. There was no retry,
additional request, contemporaneous candidate-Triton comparison or statistical
success-rate estimate. The complete numerical contract fix and the remaining
semantic failure are separate findings. Details and round/budget boundaries are
recorded in `jetd1/llm-deploy`; private payloads/outputs are not part of this repo.
This documentation revision does not change the tested manifest/runtime/profile
or advance the shared deployment branch. Keep the experiment pinned to its
actual tested commit/bundle, not this branch's later documentation HEAD.

## Baseline CPU validation history

The baseline's earlier offline verification on 2026-11-07 used Python 3.12.3 / PyTorch 2.15.0+cpu,
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
