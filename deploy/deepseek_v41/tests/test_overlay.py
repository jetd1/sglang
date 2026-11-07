"""The deployment package must reject source drift before any replacement."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deepseek_overlay", ROOT / "overlay.py")
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


class TestOverlayPackage(CustomTestCase):
    def test_source_drift_does_not_partially_install(self):
        # A different draft worker must not be silently combined with this
        # consumer, even when an earlier file was eligible for replacement.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "consumer.py").write_text("old")
            (root / "draft.py").write_text("unexpected")
            manifest = {
                "runtime": {"accepted_image_digests": ["synthetic-image"]},
                "baseline_files": {
                    "consumer.py": overlay.sha256(b"old"),
                    "draft.py": overlay.sha256(b"known-draft"),
                },
                "baseline_symlinks": {},
                "overlay_files": {"replacement.py": overlay.sha256(b"new")},
            }
            with self.assertRaisesRegex(ValueError, "draft.py"):
                overlay.apply_overlay(
                    root,
                    manifest,
                    {"consumer.py": "replacement.py"},
                    image_digest="synthetic-image",
                )
            self.assertEqual((root / "consumer.py").read_text(), "old")

    def test_reject_extra_model_symlink(self):
        # pkgutil can discover a linked backup model as a second implementation.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("model")
            (root / "model_old.py").symlink_to("model.py")
            manifest = {
                "baseline_files": {"model.py": overlay.sha256(b"model")},
                "baseline_symlinks": {},
            }
            with self.assertRaisesRegex(ValueError, "symlink"):
                overlay.check_package(root, manifest, {}, installed=False)

    def test_manifest_and_architecture_contract(self):
        manifest = overlay.load_manifest()
        canonical = "".join(
            f"{name}\0{digest}\n"
            for name, digest in sorted(manifest["baseline_files"].items())
        )
        self.assertEqual(
            overlay.sha256(canonical.encode()), manifest["baseline_tree_sha256"]
        )
        common = overlay.desired_files(manifest, "b300", None)
        sm120 = overlay.desired_files(manifest, "sm120", "off")
        shim = overlay.desired_files(manifest, "sm120", "on")
        # Every architecture receives the common guard AND the batch fix;
        # architecture files can never overwrite the consumer later in boot.
        consumer = "srt/managers/scheduler_components/batch_result_processor.py"
        self.assertEqual(common[consumer], sm120[consumer])
        self.assertEqual(common[consumer], shim[consumer])
        backend = "srt/layers/attention/deepseek_v4_backend.py"
        self.assertEqual(sm120[backend], common[backend])
        self.assertNotEqual(sm120[backend], shim[backend])
        self.assertNotEqual(
            sm120["kernels/ops/attention/flash_mla_sm120.py"],
            common["kernels/ops/attention/flash_mla_sm120.py"],
        )
        for profile, value in (("sm120", None), ("b300", "on")):
            with self.assertRaises(ValueError):
                overlay.desired_files(manifest, profile, value)

    def test_install_profiles_and_detect_late_consumer_overwrite(self):
        manifest = overlay.load_manifest()
        consumer = "srt/managers/scheduler_components/batch_result_processor.py"
        fixed = (ROOT / "runtime-overlay/sglang" / consumer).read_text()
        original = fixed.replace(
            "v.tolist() if isinstance(v, torch.Tensor) else v\n"
            "                    for v in logits_output.next_token_token_ids_logprobs_val",
            "v.tolist() for v in logits_output.next_token_token_ids_logprobs_val",
        )
        self.assertEqual(
            overlay.sha256(original.encode()), manifest["baseline_files"][consumer]
        )
        digest = manifest["runtime"]["accepted_image_digests"][0]
        for profile, shim in (("b300", None), ("sm120", "off"), ("sm120", "on")):
            with self.subTest(profile=profile, shim=shim):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "sglang"
                    shutil.copytree(
                        ROOT / "runtime-overlay/sglang",
                        root,
                        symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                    target = root / consumer
                    target.write_text(original)
                    cache = (
                        target.parent
                        / "__pycache__/batch_result_processor.synthetic.pyc"
                    )
                    cache.parent.mkdir(exist_ok=True)
                    cache.write_bytes(b"synthetic stale bytecode")
                    desired = overlay.desired_files(manifest, profile, shim)
                    first = overlay.apply_overlay(
                        root, manifest, desired, image_digest=digest
                    )
                    self.assertEqual(target.read_text(), fixed)
                    self.assertFalse(cache.exists())
                    self.assertEqual(
                        overlay.apply_overlay(
                            root, manifest, desired, image_digest=digest
                        ),
                        first,
                    )
                    overlay.apply_overlay(
                        root, manifest, desired, image_digest=digest, verify=True
                    )
                    # A legacy late copy must fail final verification, not hide
                    # the new consumer behind a previously successful install.
                    target.write_text(original)
                    with self.assertRaisesRegex(
                        ValueError, "batch_result_processor.py"
                    ):
                        overlay.apply_overlay(
                            root, manifest, desired, image_digest=digest, verify=True
                        )

    def test_archive_is_reproducible_and_complete(self):
        manifest = overlay.load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.tar.gz"
            second = Path(directory) / "second.tar.gz"
            self.assertEqual(
                overlay.pack(manifest, first), overlay.pack(manifest, second)
            )
            with overlay.tarfile.open(first) as archive:
                contents = {
                    member.name: archive.extractfile(member).read()
                    for member in archive.getmembers()
                    if member.isfile()
                }
                for name, target in manifest["baseline_symlinks"].items():
                    member = archive.getmember(
                        f"deepseek_v41/runtime-overlay/sglang/{name}"
                    )
                    self.assertTrue(member.issym())
                    self.assertEqual(member.linkname, target)
            packed = json.loads(contents["deepseek_v41/manifest.json"])
            for name, digest in packed["overlay_files"].items():
                self.assertEqual(
                    overlay.sha256(contents[f"deepseek_v41/{name}"]), digest
                )
            for name, digest in packed["runtime_files"].items():
                self.assertEqual(
                    overlay.sha256(
                        contents[f"deepseek_v41/runtime-overlay/sglang/{name}"]
                    ),
                    digest,
                )
            with self.assertRaises(FileExistsError):
                overlay.pack(manifest, first)


if __name__ == "__main__":
    unittest.main()
