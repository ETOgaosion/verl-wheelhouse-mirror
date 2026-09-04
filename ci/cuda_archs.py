#!/usr/bin/env python3
"""CUDA SM-arch parsing and wheel fatbin inspection for the wheelhouse.

Two responsibilities:

1. Turn a TORCH_CUDA_ARCH_LIST-style string ("8.0;9.0;10.0", "80 90a",
   "12.0f+PTX", ...) into canonical arch tokens ("8.0", "9.0a", "12.0f"),
   so skip detection compares the *set* of SM arches a build targets rather
   than an opaque separator-sensitive string.

2. Extract the SM arches actually fat-binned into a built/published wheel
   (its .so members' .nv_fatbin sections, plus any loose .cubin members),
   used:
   - as a build-time gate in .github/workflows/_build.yml (fail before
     uploading a wheel whose fatbin does not cover the requested arches -
     e.g. deep-ep's sm_80 cubin, which only exists because
     ci/patches/enable_deep_ep_sm80.py rewrites the checkout);
   - by generate_matrix.py's release skip detection, which downloads the
     wheels already attached to a release and requires their fatbin arches
     to cover the arches versions.yaml promises.

Stdlib-only by design (the matrix-computation runner only has pyyaml):
`cuobjdump` is used when it is on PATH / under CUDA_HOME, otherwise the
`.nv_fatbin` ELF section is located with a minimal ELF section parser and
scanned for `sm_XX` / `compute_XX` tokens directly (PTX blobs embed a
`.target sm_90a` directive; fatbin cubin entries carry sm_XX name strings).
When neither path can read any arch, callers fail *closed* (rebuild) rather
than risk skipping a wheel that does not cover what it should.

Usage:
    # build-time gate (exit 1 if the wheels' fatbin misses a required arch):
    python ci/cuda_archs.py dist/*.whl --require "8.0;9.0;10.0"
    # report only (components that hardcode their own gencode flags):
    python ci/cuda_archs.py dist/*.whl
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# sm_80 / sm_90a / compute_100 / .target sm_120f ... Two-to-three digit
# arch number (75 .. 121 today), optional feature suffix letter (a/f/...).
# The lookarounds keep e.g. "xsm_900" or "sm_900" from matching.
_ARCH_TOKEN_RE = re.compile(rb"(?<![A-Za-z0-9_])(?:sm|compute)_(\d{2,3})([a-z]?)(?![A-Za-z0-9])")

_ARCH_LIST_SPLIT_RE = re.compile(r"[;,\s]+")
_PTX_SUFFIX_RE = re.compile(r"\+?ptx$", re.IGNORECASE)
_ARCH_PART_RE = re.compile(r"^(\d+)(?:\.(\d+))?([a-z]?)$")
_TOKEN_SUFFIX_RE = re.compile(r"^(\d+\.\d+)([a-z]?)$")

# Sections nvcc emits fatbins under. The standard name is ".nv_fatbin";
# accept the prefix to tolerate per-TU suffixed variants.
_FATBIN_SECTION_PREFIX = b".nv_fatbin"

# e_machine value NVIDIA's cubin ELF objects use (EM_CUDA). A loose .cubin
# member is an ELF with this machine type; a host .so is EM_X86_64 (62) or
# EM_AARCH64 (183).
_EM_CUDA = 190


def canonical_token(number: int, suffix: str = "") -> str:
    """Encode a numeric SM arch as the dotted canonical form, e.g. 80 -> "8.0",
    100 -> "10.0", 103 -> "10.3", 121 -> "12.1"."""
    return f"{number // 10}.{number % 10}{suffix}"


def parse_arch_list(raw: Optional[str]) -> List[str]:
    """Parse a TORCH_CUDA_ARCH_LIST-style string into sorted unique tokens.

    Accepts the shapes every wheelhouse consumer uses: semicolon/space/comma
    separators, dotted ("8.0") and undotted ("80") spellings, feature
    suffixes ("9.0a", "12.0f"), and a trailing "+PTX" (PTX is forward
    compatibility for the same base arch, not a distinct SM). Unparseable
    segments are dropped rather than raising, so a malformed value degrades
    to "no arches known" - the fail-closed direction for skip detection.
    """
    tokens: Set[str] = set()
    if not raw:
        return []
    for part in _ARCH_LIST_SPLIT_RE.split(str(raw)):
        part = part.strip()
        if not part:
            continue
        part = _PTX_SUFFIX_RE.sub("", part)
        match = _ARCH_PART_RE.match(part)
        if not match:
            continue
        suffix = match.group(3) or ""
        if match.group(2) is not None:
            token = f"{int(match.group(1))}.{int(match.group(2))}{suffix}"
        else:
            number = int(match.group(1))
            token = canonical_token(number, suffix)
        tokens.add(token)
    return sorted(tokens)


def base_arch(token: str) -> str:
    """Drop a token's feature suffix, e.g. "9.0a" -> "9.0", "12.0f" -> "12.0".

    Coverage is compared on base arches: an sm_90a cubin loads on sm_90
    hardware, so a wheel fat-binning "9.0a" satisfies a requirement for
    "9.0".
    """
    match = _TOKEN_SUFFIX_RE.match(token)
    if not match:
        return token
    return match.group(1)


def base_arch_set(tokens: Set[str]) -> Set[str]:
    return {base_arch(token) for token in tokens}


def find_cuobjdump() -> Optional[str]:
    """Locate cuobjdump via $CUOBJDUMP, PATH, then $CUDA_HOME/$CUDA_PATH."""
    env_override = os.environ.get("CUOBJDUMP")
    candidates = [env_override] if env_override else []
    candidates.append(shutil.which("cuobjdump"))
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    candidates.append(str(Path(cuda_home) / "bin" / "cuobjdump"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _arch_tokens_from_bytes(data: bytes) -> Set[str]:
    return {
        canonical_token(int(number), (suffix or b"").decode("ascii"))
        for number, suffix in _ARCH_TOKEN_RE.findall(data)
    }


def _elf_section_bytes(data: bytes, section_name: bytes) -> List[bytes]:
    """Minimal ELF section extractor (32/64-bit, either endianness).

    Wheels ship Linux ELF shared objects (x86_64 / aarch64 are both
    64-bit little-endian, but the 32-bit/big-endian paths come for free);
    everything beyond the section header table - the one thing needed to
    find .nv_fatbin - is ignored.
    """
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return []
    ei_class = data[4]
    ei_data = data[5]
    if ei_data == 1:
        endian = "<"
    elif ei_data == 2:
        endian = ">"
    else:
        return []

    try:
        if ei_class == 2:  # ELF64
            (e_shoff,) = struct.unpack_from(endian + "Q", data, 0x28)
            (e_shentsize,) = struct.unpack_from(endian + "H", data, 0x3A)
            (e_shnum,) = struct.unpack_from(endian + "H", data, 0x3C)
            (e_shstrndx,) = struct.unpack_from(endian + "H", data, 0x3E)
            sh_fmt = endian + "IIQQQQIIQQ"  # sizeof == 64
        elif ei_class == 1:  # ELF32
            (e_shoff,) = struct.unpack_from(endian + "I", data, 0x20)
            (e_shentsize,) = struct.unpack_from(endian + "H", data, 0x2E)
            (e_shnum,) = struct.unpack_from(endian + "H", data, 0x30)
            (e_shstrndx,) = struct.unpack_from(endian + "H", data, 0x32)
            sh_fmt = endian + "IIIIIIIIII"  # sizeof == 40
        else:
            return []

        if e_shoff == 0 or e_shnum == 0 or e_shentsize < struct.calcsize(sh_fmt):
            return []

        sections: List[Tuple[int, int, int]] = []  # (name offset, data offset, size)
        for index in range(e_shnum):
            offset = e_shoff + index * e_shentsize
            fields = struct.unpack_from(sh_fmt, data, offset)
            sh_name, _sh_type = fields[0], fields[1]
            sh_offset, sh_size = fields[4], fields[5]
            sections.append((sh_name, sh_offset, sh_size))

        if e_shstrndx >= len(sections):
            return []
        _, strtab_offset, strtab_size = sections[e_shstrndx]
        strtab = data[strtab_offset : strtab_offset + strtab_size]

        matches: List[bytes] = []
        for name_offset, sh_offset, sh_size in sections:
            end = strtab.find(b"\x00", name_offset)
            name = strtab[name_offset:end] if end >= 0 else strtab[name_offset:]
            if name.startswith(_FATBIN_SECTION_PREFIX):
                matches.append(data[sh_offset : sh_offset + sh_size])
        return matches
    except (struct.error, IndexError):
        return []


def _elf_machine(data: bytes) -> Optional[int]:
    """e_machine of an ELF blob (62=x86_64, 183=aarch64, 190=CUDA cubin)."""
    if len(data) < 0x14 or data[:4] != b"\x7fELF":
        return None
    if data[5] == 1:
        endian = "<"
    elif data[5] == 2:
        endian = ">"
    else:
        return None
    try:
        return struct.unpack_from(endian + "H", data, 0x12)[0]
    except struct.error:
        return None


def cuobjdump_archs(path: Path, cuobjdump: str) -> Set[str]:
    """Arch tokens `cuobjdump --list-elf/--list-ptx` reports for one binary."""
    try:
        result = subprocess.run(
            [cuobjdump, "--list-elf", "--list-ptx", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return _arch_tokens_from_bytes((result.stdout + result.stderr).encode("utf-8", "replace"))


def binary_archs(data: bytes, cuobjdump: Optional[str] = None) -> Set[str]:
    """All SM arch tokens embedded in one .so / .cubin blob."""
    found: Set[str] = set()

    # cuobjdump is authoritative (it parses fatbin entry headers and cubin
    # metadata); only run it on real files, not in-memory bytes.
    if cuobjdump is not None:
        with tempfile.NamedTemporaryFile(prefix="cuda-arch-", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            found |= cuobjdump_archs(tmp_path, cuobjdump)
        finally:
            tmp_path.unlink(missing_ok=True)

    # Pure-Python fallback / complement: fatbin sections (PTX text + cubin
    # name strings carry sm_XX tokens), plus a whole-blob scan for loose
    # .cubin members - themselves ELF objects, but with e_machine EM_CUDA and
    # no .nv_fatbin wrapper. A *host* ELF without a fatbin section is not
    # scanned whole: its diagnostics/strings can mention sm_XX without
    # containing any cubin, and such a false positive would mask a missing
    # arch (the unsafe failure direction for the build gate).
    fatbins = _elf_section_bytes(data, _FATBIN_SECTION_PREFIX)
    if fatbins:
        for blob in fatbins:
            found |= _arch_tokens_from_bytes(blob)
    elif not found and _elf_machine(data) in (None, _EM_CUDA):
        # Non-ELF blob or CUDA cubin: scan everything.
        found |= _arch_tokens_from_bytes(data)
    return found


def wheel_cuda_archs(wheel_path: Path, cuobjdump: Optional[str] = "auto") -> Set[str]:
    """Arch tokens fat-binned anywhere in a wheel's .so / .cubin members."""
    if cuobjdump == "auto":
        cuobjdump = find_cuobjdump()
    found: Set[str] = set()
    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            base = member.rsplit("/", 1)[-1]
            if not (base.endswith(".so") or ".so." in base or base.endswith(".cubin")):
                continue
            found |= binary_archs(archive.read(member), cuobjdump=cuobjdump)
    return found


def wheels_cuda_archs(wheel_paths: List[Path], cuobjdump: Optional[str] = "auto") -> Set[str]:
    found: Set[str] = set()
    for wheel in wheel_paths:
        found |= wheel_cuda_archs(Path(wheel), cuobjdump=cuobjdump)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wheels", nargs="+", help="Wheel file(s) to inspect.")
    parser.add_argument(
        "--require",
        default="",
        help=(
            "TORCH_CUDA_ARCH_LIST-style arch set the wheels must cover "
            "(e.g. '8.0;9.0;10.0'). Exit 1 if any base arch is missing from "
            "the fatbin. Omit for report-only mode (components that hardcode "
            "their own gencode flags)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the found arch tokens as a JSON list instead of text.",
    )
    args = parser.parse_args()

    wheel_paths = [Path(name) for name in args.wheels]
    missing_files = [str(path) for path in wheel_paths if not path.is_file()]
    if missing_files:
        print(f"No such wheel file(s): {', '.join(missing_files)}", file=sys.stderr)
        sys.exit(2)

    cuobjdump = find_cuobjdump()
    if cuobjdump:
        print(f"Using cuobjdump: {cuobjdump}", file=sys.stderr)
    else:
        print("cuobjdump not found; scanning .nv_fatbin sections directly.", file=sys.stderr)

    found = wheels_cuda_archs(wheel_paths, cuobjdump=cuobjdump)

    if args.json:
        import json

        print(json.dumps(sorted(found)))
    else:
        for wheel in wheel_paths:
            wheel_archs = wheel_cuda_archs(wheel, cuobjdump=cuobjdump)
            print(f"{wheel}: {', '.join(sorted(wheel_archs)) or 'no CUDA arch markers found'}")

    required = parse_arch_list(args.require)
    if required:
        missing = base_arch_set(set(required)) - base_arch_set(found)
        if missing:
            print(
                f"error: wheel fatbin is missing required CUDA arch(es): "
                f"{', '.join(sorted(missing))} (required {', '.join(required)}, "
                f"found {', '.join(sorted(base_arch_set(found))) or 'none'}).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"All required arches present in fatbin: {', '.join(required)}.", file=sys.stderr)


if __name__ == "__main__":
    main()
