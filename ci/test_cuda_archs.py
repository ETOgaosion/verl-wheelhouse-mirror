#!/usr/bin/env python3
"""Tests for cuda_archs: arch-list parsing and fatbin arch extraction.

The fatbin reader is exercised against *synthetic* ELF files assembled here
(minimal Elf64 section tables) rather than real nvcc output: the logic under
test is "find .nv_fatbin, read sm_XX tokens", and a hand-built ELF with a
known section layout makes missing-section / wrong-machine cases deterministic.
cuobjdump is never assumed to exist on the test machine; the pure-Python
fallback is what runs.
"""

import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cuda_archs


def make_elf64(sections, machine=62):
    """Build a minimal little-endian ELF64 blob.

    sections: list of (section_name: bytes, data: bytes). A .shstrtab is
    added automatically. machine defaults to 62 (EM_X86_64); pass 190 for a
    CUDA cubin-flavoured object.
    """
    shstrtab = b"\x00"
    name_offsets = {}
    for name in [b".shstrtab"] + [name for name, _ in sections]:
        name_offsets[name] = len(shstrtab)
        shstrtab += name + b"\x00"
    ordered = [(b".shstrtab", shstrtab)] + sections

    ehdr_size, shentsize = 64, 64
    shnum = 1 + len(ordered)  # null section + .shstrtab + requested sections
    data_start = ehdr_size + shnum * shentsize

    shdrs = [b"\x00" * shentsize]
    blob = bytearray()
    cursor = data_start
    for name, data in ordered:
        sh_type = 3 if name == b".shstrtab" else 1  # SHT_STRTAB / SHT_PROGBITS
        shdrs.append(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offsets[name],
                sh_type,
                0,
                0,
                cursor,
                len(data),
                0,
                0,
                1,
                0,
            )
        )
        blob += data
        cursor += len(data)

    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF" + bytes([2, 1, 1]) + b"\x00" * 9,
        2,  # e_type ET_EXEC
        machine,
        1,  # e_version
        0,  # e_entry
        0,  # e_phoff
        ehdr_size,  # e_shoff
        0,  # e_flags
        ehdr_size,
        0,  # e_phentsize
        0,  # e_phnum
        shentsize,
        shnum,
        1,  # e_shstrndx
    )
    return ehdr + b"".join(shdrs) + bytes(blob)


def fatbin_elf(arches, machine=62):
    """A host ELF whose .nv_fatbin section embeds the given sm_XX tokens."""
    payload = b"\x00" + b"\x00".join(arches) + b"\x00"
    return make_elf64([(b".nv_fatbin", payload)], machine=machine)


def make_wheel(path: Path, so_members):
    """Write a wheel (zip) whose .so members carry the given ELF blobs."""
    with zipfile.ZipFile(path, "w") as archive:
        for index, blob in enumerate(so_members):
            archive.writestr(f"package/_ext{index}.cpython-312-x86_64-linux-gnu.so", blob)
        archive.writestr("package-1.0.dist-info/METADATA", b"sm_80 in a text file must be ignored")
    return path


class ParseArchListTests(unittest.TestCase):
    def test_dotted_semicolon_form(self):
        self.assertEqual(cuda_archs.parse_arch_list("8.0;9.0;10.0"), ["10.0", "8.0", "9.0"])

    def test_undotted_and_space_separated(self):
        self.assertEqual(cuda_archs.parse_arch_list("80 90 100"), ["10.0", "8.0", "9.0"])

    def test_feature_suffixes_are_kept(self):
        self.assertEqual(
            cuda_archs.parse_arch_list("8.0 9.0a 10.0a 12.0f"),
            ["10.0a", "12.0f", "8.0", "9.0a"],
        )

    def test_undotted_with_suffix(self):
        self.assertEqual(cuda_archs.parse_arch_list("90a"), ["9.0a"])

    def test_ptx_suffix_is_dropped(self):
        self.assertEqual(cuda_archs.parse_arch_list("12.0f+PTX"), ["12.0f"])
        self.assertEqual(cuda_archs.parse_arch_list("9.0+ptx"), ["9.0"])

    def test_comma_separated(self):
        self.assertEqual(cuda_archs.parse_arch_list("8.0,9.0,10.0"), ["10.0", "8.0", "9.0"])

    def test_empty_and_garbage(self):
        self.assertEqual(cuda_archs.parse_arch_list(""), [])
        self.assertEqual(cuda_archs.parse_arch_list(None), [])
        self.assertEqual(cuda_archs.parse_arch_list("not-an-arch; ; 8.0"), ["8.0"])

    def test_equivalent_spellings_normalize_to_the_same_set(self):
        dotted = set(cuda_archs.parse_arch_list("8.0;9.0;10.0"))
        undotted = set(cuda_archs.parse_arch_list("80;90;100"))
        spaced = set(cuda_archs.parse_arch_list("8.0 9.0 10.0"))
        self.assertEqual(dotted, undotted)
        self.assertEqual(dotted, spaced)


class BaseArchTests(unittest.TestCase):
    def test_suffix_stripped(self):
        self.assertEqual(cuda_archs.base_arch("9.0a"), "9.0")
        self.assertEqual(cuda_archs.base_arch("12.0f"), "12.0")
        self.assertEqual(cuda_archs.base_arch("8.0"), "8.0")

    def test_suffixed_cubin_covers_plain_requirement(self):
        found = cuda_archs.base_arch_set({"9.0a", "10.0a", "8.0"})
        self.assertEqual(found, {"8.0", "9.0", "10.0"})


class ElfSectionTests(unittest.TestCase):
    def test_fatbin_section_is_extracted(self):
        payload = b"\x00sm_80\x00sm_90a\x00"
        elf = make_elf64([(b".nv_fatbin", payload), (b".text", b"\x00" * 16)])
        sections = cuda_archs._elf_section_bytes(elf, cuda_archs._FATBIN_SECTION_PREFIX)
        self.assertEqual(sections, [payload])

    def test_non_elf_returns_nothing(self):
        self.assertEqual(cuda_archs._elf_section_bytes(b"not an elf file at all", b".nv_fatbin"), [])

    def test_truncated_elf_returns_nothing(self):
        self.assertEqual(cuda_archs._elf_section_bytes(b"\x7fELF" + b"\x00" * 10, b".nv_fatbin"), [])

    def test_machine_is_read(self):
        self.assertEqual(cuda_archs._elf_machine(fatbin_elf([b"sm_80"], machine=62)), 62)
        self.assertEqual(cuda_archs._elf_machine(fatbin_elf([b"sm_80"], machine=190)), 190)
        self.assertIsNone(cuda_archs._elf_machine(b"plain bytes"))


class BinaryArchTests(unittest.TestCase):
    def test_fatbin_tokens_are_found(self):
        elf = fatbin_elf([b"sm_80", b"sm_90a", b"sm_100"])
        self.assertEqual(
            cuda_archs.binary_archs(elf, cuobjdump=None), {"8.0", "9.0a", "10.0"}
        )

    def test_host_elf_without_fatbin_is_not_scanned_whole(self):
        # A host .so with no .nv_fatbin but a stray "sm_99" string (diagnostics,
        # symbol names) must not report coverage: that false positive is the
        # unsafe direction for the build gate.
        elf = make_elf64([(b".rodata", b"this kernel needs sm_99 to run\x00")], machine=62)
        self.assertEqual(cuda_archs.binary_archs(elf, cuobjdump=None), set())

    def test_cubin_elf_is_scanned_whole(self):
        # Loose .cubin members are EM_CUDA ELF objects with no fatbin wrapper.
        elf = make_elf64([(b".text.sm_90a", b"\x00")], machine=190)
        elf = elf + b"\x00sm_90a\x00sm_100\x00"
        self.assertEqual(cuda_archs.binary_archs(elf, cuobjdump=None), {"9.0a", "10.0"})

    def test_word_boundaries(self):
        # "xsm_900" must not be read as an arch token.
        tokens = cuda_archs._arch_tokens_from_bytes(b"prefix xsm_900 suffix")
        self.assertEqual(tokens, set())


class WheelArchTests(unittest.TestCase):
    def test_wheel_fatbin_arches(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(
                Path(tmp) / "kernels-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                [fatbin_elf([b"sm_80", b"sm_90", b"sm_100"])],
            )
            self.assertEqual(
                cuda_archs.wheel_cuda_archs(wheel, cuobjdump=None),
                {"8.0", "9.0", "10.0"},
            )

    def test_pure_python_wheel_has_no_arches(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "pure-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("pure/__init__.py", b"# sm_80 mention is harmless without a .so")
            self.assertEqual(cuda_archs.wheel_cuda_archs(wheel, cuobjdump=None), set())


class CliGateTests(unittest.TestCase):
    def run_cli(self, wheels, require=None):
        args = [sys.executable, str(Path(cuda_archs.__file__))]
        args.extend(str(w) for w in wheels)
        if require is not None:
            args.extend(["--require", require])
        return subprocess.run(args, capture_output=True, text=True)

    def test_gate_passes_when_all_arches_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(
                Path(tmp) / "kernels-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                [fatbin_elf([b"sm_80", b"sm_90a", b"sm_100"])],
            )
            result = self.run_cli([wheel], require="8.0;9.0;10.0")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_gate_fails_when_an_arch_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(
                Path(tmp) / "kernels-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                [fatbin_elf([b"sm_90", b"sm_100"])],  # sm_80 cubin never materialized
            )
            result = self.run_cli([wheel], require="8.0;9.0;10.0")
            self.assertEqual(result.returncode, 1)
            self.assertIn("8.0", result.stderr)

    def test_missing_wheel_file_is_an_error(self):
        result = self.run_cli([Path("/nonexistent/fake-1.0-py3-none-any.whl")], require="8.0")
        self.assertEqual(result.returncode, 2)

    def test_report_only_mode_never_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(
                Path(tmp) / "kernels-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                [fatbin_elf([b"sm_90"])],
            )
            result = self.run_cli([wheel])
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
