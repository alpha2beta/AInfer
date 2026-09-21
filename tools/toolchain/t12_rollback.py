#!/usr/bin/env python3
"""T1.2 offline rootless toolchain snapshot and rollback verification.

This is intentionally a user-space test, not a system package manager action.
It treats the checked-out CachyOS package cache and extracted sysroot as the
rollback unit, runs the Level Zero timestamp smoke in a rootless mount/user
namespace, replaces the loader with a deterministic invalid ELF candidate,
checks that the smoke fails, restores the pinned loader, and checks recovery.
No host files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import re


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], env: dict[str, str], timeout: int = 30) -> tuple[int, str]:
    p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.strip()


def stable_output(output: str, private_root: Path | None = None) -> str:
    if private_root is not None:
        output = output.replace(str(private_root), "<private-sysroot>")
    return re.sub(r"start=\d+ end=\d+", "start=<device> end=<device>", output)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--smoke", type=Path,
                    default=None,
                    help="built l0_timestamp_smoke executable")
    ap.add_argument("--out", type=Path,
                    default=None,
                    help="JSON report path")
    args = ap.parse_args()

    repo = args.repo.resolve()
    cache = repo / "tools/toolchain/cache"
    sysroot = repo / "tools/toolchain/sysroot"
    smoke = (args.smoke or repo / "build-258v/tools/l0probe/l0_timestamp_smoke").resolve()
    out = args.out or repo / "tools/toolchain/report_t12_rollback.json"
    loader = sysroot / "usr/lib/libze_loader.so.1"
    if not smoke.is_file() or not loader.is_file():
        raise SystemExit(f"missing smoke or loader: {smoke} / {loader}")

    packages = sorted(p for p in cache.glob("*.pkg.tar.zst") if p.is_file())
    snapshot = {
        "packages": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in packages],
        "sysroot_buildinfo_sha256": sha256(sysroot / ".BUILDINFO"),
        "sysroot_pkginfo_sha256": sha256(sysroot / ".PKGINFO"),
        "loader_sha256": sha256(loader),
        "loader_bytes": loader.stat().st_size,
    }

    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = str(sysroot / "usr/lib")
    # A clean mount namespace prevents accidental dependence on a host-mounted
    # package/runtime path. User namespace mapping is unprivileged on CachyOS.
    ns = ["unshare", "--user", "--mount", "--map-root-user", "--", str(smoke)]

    def smoke_run() -> dict[str, object]:
        started = time.monotonic()
        rc, output = run(ns, base_env)
        return {"returncode": rc, "seconds": round(time.monotonic() - started, 3),
                "output": stable_output(output)[-2000:]}

    baseline = smoke_run()
    if baseline["returncode"] != 0:
        raise SystemExit(f"baseline smoke failed:\n{baseline['output']}")

    with tempfile.TemporaryDirectory(prefix="ainfer-t12-") as td:
        rollback_root = Path(td)
        backup = rollback_root / "libze_loader.so.1.pinned"
        shutil.copy2(loader, backup)
        bad = rollback_root / "libze_loader.so.1.bad"
        bad.write_bytes(b"AINFER-T12-INVALID-LOADER\n")
        os.chmod(bad, 0o755)

        # The sysroot is tracked as an immutable input in normal operation.
        # Copy it into a private root for the mutation/recovery exercise.
        private_root = rollback_root / "sysroot"
        shutil.copytree(sysroot, private_root, symlinks=True)
        private_loader = private_root / "usr/lib/libze_loader.so.1"
        private_env = base_env.copy()
        private_env["LD_LIBRARY_PATH"] = str(private_root / "usr/lib")
        private_ns = ["unshare", "--user", "--mount", "--map-root-user",
                      "--", str(smoke)]

        def private_run() -> dict[str, object]:
            started = time.monotonic()
            rc, output = run(private_ns, private_env)
            return {"returncode": rc, "seconds": round(time.monotonic() - started, 3),
                    "output": stable_output(output, private_root)[-2000:]}

        private_baseline = private_run()
        shutil.copy2(bad, private_loader)
        rollback_failure = private_run()
        shutil.copy2(backup, private_loader)
        restored = private_run()

    checks = {
        "baseline_pass": baseline["returncode"] == 0,
        "private_baseline_pass": private_baseline["returncode"] == 0,
        "bad_runtime_rejected": rollback_failure["returncode"] != 0,
        "restored_pass": restored["returncode"] == 0,
        "snapshot_unchanged": sha256(loader) == snapshot["loader_sha256"],
    }
    report = {
        "task": "T1.2",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "scope": "offline pinned CachyOS Intel runtime/sysroot; rootless user+mount namespace",
        "rollback_unit": "tools/toolchain/cache/*.pkg.tar.zst + tools/toolchain/sysroot",
        "snapshot": snapshot,
        "baseline": baseline,
        "private_baseline": private_baseline,
        "rollback_failure": rollback_failure,
        "restored": restored,
        "checks": checks,
        "notes": [
            "No host package database or system files were modified.",
            "The negative phase substitutes an invalid loader in a private copied sysroot.",
            "This validates the pinned runtime rollback unit, not a full distro upgrade rollback.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
