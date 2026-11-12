#!/usr/bin/env python3
"""Package and install the image-pinned DeepSeek overlay before importing SGLang."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    artifacts = {
        f"runtime-overlay/sglang/{name}": digest
        for name, digest in manifest["runtime_files"].items()
    }
    artifacts.update(manifest["overlay_files"])
    for name, digest in artifacts.items():
        data = (ROOT / name).read_bytes()
        if sha256(data) != digest:
            raise ValueError(f"overlay artifact checksum mismatch: {name}")
    for name, target in manifest["baseline_symlinks"].items():
        path = ROOT / "runtime-overlay/sglang" / name
        if not path.is_symlink() or path.readlink().as_posix() != target:
            raise ValueError(f"overlay symlink mismatch: {name}")
    return manifest


def desired_files(manifest, profile, shim):
    if (profile == "sm120") != (shim is not None):
        raise ValueError("sm120 requires explicit --sm120-shim on|off; b300 forbids it")
    groups = []
    if profile == "sm120":
        groups.append("sm120")
        if shim == "on":
            groups.append("sm120-shim")
    desired = {
        name: f"runtime-overlay/sglang/{name}" for name in manifest["runtime_files"]
    }
    for group in groups:
        prefix = f"profiles/{group}/"
        for name in manifest["overlay_files"]:
            if name.startswith(prefix):
                desired[name[len(prefix) :]] = name
    return desired


def artifact_hash(manifest, name):
    prefix = "runtime-overlay/sglang/"
    if name.startswith(prefix):
        return manifest["runtime_files"][name[len(prefix) :]]
    return manifest["overlay_files"][name]


def package_files(root):
    return {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }


def check_package(root, manifest, desired, *, installed):
    actual = package_files(root)
    baseline = manifest["baseline_files"]
    if actual.keys() != baseline.keys():
        missing = sorted(baseline.keys() - actual.keys())
        extra = sorted(actual.keys() - baseline.keys())
        raise ValueError(
            f"package file set differs: missing={missing[:5]}, extra={extra[:5]}"
        )
    links = {
        path.relative_to(root).as_posix(): path.readlink().as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    }
    if links != manifest["baseline_symlinks"]:
        raise ValueError("package symlink set or target mismatch")
    for name, baseline_hash in baseline.items():
        final_hash = (
            artifact_hash(manifest, desired[name]) if name in desired else baseline_hash
        )
        allowed = {final_hash} if installed else {baseline_hash, final_hash}
        if sha256(actual[name].read_bytes()) not in allowed:
            raise ValueError(f"package checksum mismatch: {name}")


def apply_overlay(root, manifest, desired, *, image_digest, verify=False):
    if image_digest not in manifest["runtime"]["accepted_image_digests"]:
        raise ValueError(
            "unsupported runtime image digest; native dependencies are image-pinned"
        )
    # Check the entire package before writing anything. This catches a stale
    # parser CM, a different draft worker, and an incompatible image/source tree.
    check_package(root, manifest, desired, installed=verify)
    if not verify:
        for name, source in desired.items():
            target = root / name
            data = (ROOT / source).read_bytes()
            if target.read_bytes() == data:
                continue
            target.write_bytes(data)
            # compileall in the base image creates timestamp-based caches.
            # Remove this module's cached code before a subsequent import.
            for cache in target.parent.glob(f"__pycache__/{target.stem}.*.pyc"):
                cache.unlink()
        check_package(root, manifest, desired, installed=True)
    return {
        "overlay_version": manifest["overlay_version"],
        "manifest_sha256": sha256((ROOT / "manifest.json").read_bytes()),
        "runtime_image_digest": image_digest,
        "baseline_source_layer": manifest["runtime"]["source_layer"],
        "installed_tree_sha256": sha256(
            "".join(
                f"{name}\0{artifact_hash(manifest, source)}\n"
                for name, source in sorted(desired.items())
            ).encode()
        ),
        "installed_files": {
            name: artifact_hash(manifest, source)
            for name, source in sorted(desired.items())
            if artifact_hash(manifest, source) != manifest["baseline_files"][name]
        },
    }


def pack(manifest, output):
    names = ["LICENSE", "README.md", "manifest.json", "overlay.py"]
    names += sorted(manifest["overlay_files"])
    names += [f"runtime-overlay/sglang/{name}" for name in manifest["runtime_files"]]
    names += [
        f"runtime-overlay/sglang/{name}" for name in manifest["baseline_symlinks"]
    ]
    names += [
        "tests/test_logprob_rows.py",
        "tests/test_overlay.py",
        "tests/test_sm120_extra_slots.py",
        "tests/test_dspark_request_budget.py",
    ]
    # Exclusive create: never overwrite an earlier deployment artifact.
    with output.open("xb") as stream:
        with gzip.GzipFile(
            fileobj=stream, mode="wb", filename="", mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(names):
                    path = ROOT / name
                    entry = tarfile.TarInfo(f"deepseek_v41/{name}")
                    entry.mtime = 0
                    if path.is_symlink():
                        entry.type = tarfile.SYMTYPE
                        entry.linkname = path.readlink().as_posix()
                        entry.mode = 0o777
                        archive.addfile(entry)
                    else:
                        data = path.read_bytes()
                        entry.size = len(data)
                        entry.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                        archive.addfile(entry, io.BytesIO(data))
    return {"archive_sha256": sha256(output.read_bytes()), "files": len(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("install", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--package-root", required=True, type=Path)
        sub.add_argument("--image-digest", required=True)
        sub.add_argument("--profile", required=True, choices=("b300", "sm120"))
        sub.add_argument("--sm120-shim", choices=("on", "off"))
    commands.add_parser("pack").add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        if args.command == "pack":
            report = pack(manifest, args.output)
        else:
            desired = desired_files(manifest, args.profile, args.sm120_shim)
            report = apply_overlay(
                args.package_root,
                manifest,
                desired,
                image_digest=args.image_digest,
                verify=args.command == "verify",
            )
            report.update(profile=args.profile, sm120_shim=args.sm120_shim)
        print(json.dumps(report, sort_keys=True))
    except (OSError, ValueError) as error:
        parser.exit(1, f"FATAL: {error}\n")


if __name__ == "__main__":
    main()
