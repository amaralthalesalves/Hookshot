"""The pointer table of task, mesh and compute programs is read again each submission (issue 574).

It comes from the contents of small storage buffers, not from the bindings: a rebind of the
same buffer keeps bind_gen, so the table is also keyed on the submission's stamp. Within one
recording, the c0 bank that holds it and a task/mesh program's descriptors are kept on the
same stamp too (a pool remade or mirrored mid-recording).
RX_VK_BIND_EVERY=1 makes every bind a new one again (the A/B for the issue).
python -I -B tests/test_pointer_table.py [--cxx g++]
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    bank = (ROOT / "gpu" / "vk" / "vk_bank.cpp").read_text(encoding="utf-8")
    table = block(bank, "const PointerEntry* pointer_table(")
    read = table.index("memory_read_as_submitted(")
    if "gpu_bind_stamp()" not in table[:read]:
        raise SystemExit("the pointer table is kept across submissions")
    if "pointer_table_kept(" not in table[:read]:
        raise SystemExit("the pointer table is kept without checking its key")
    if "for_stamp = stamp" not in table[:read]:
        raise SystemExit("the pointer table does not remember the stamp it was read at")
    memory = (ROOT / "gpu" / "vk" / "vk_memory.cpp").read_text(encoding="utf-8")
    if "mirror_epoch" not in block(memory, "uint64_t gpu_bind_stamp("):
        raise SystemExit("the stamp does not change with each submission")
    replay = (ROOT / "gpu" / "vk" / "vk_replay.cpp").read_text(encoding="utf-8")
    every = block(replay, "bool bind_every(")
    if 'getenv("RX_VK_BIND_EVERY")' not in every:
        raise SystemExit("RX_VK_BIND_EVERY is not read")
    lines = replay[replay.index("case NOP_BIND_PROGRAM:"):replay.index("case NOP_BARRIER:")].splitlines()
    for i, line in enumerate(lines):
        if "s.bind_gen++" not in line or "NOP_SET_TEX_POOL" in line:
            continue
        if "bind_every()" not in line and "bind_every()" not in lines[i - 1]:
            raise SystemExit(f"a bind ignores RX_VK_BIND_EVERY: {line.strip()}")
    draw = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    uniform = block(draw, "VkDescriptorBufferInfo uniform_range(")
    reuse = uniform.index("return {cur->stream, c.at, n};")
    if "bank_kept(" not in uniform[:reuse] or "bank_reads_table(" not in uniform[:reuse]:
        raise SystemExit("the c0 bank with the pointer table is kept on bind_gen alone")
    if "c.stamp = stamp" not in uniform:
        raise SystemExit("the c0 bank does not remember the stamp it was filled at")
    desc = draw[draw.index("const bool same_desc ="):draw.index("if (!same_desc) {")]
    if "desc_stamp == last_desc_stamp" not in desc or "p.mesh ? gpu_bind_stamp()" not in draw:
        raise SystemExit("a task/mesh program's descriptors are kept on bind_gen alone")
    if "last_desc_stamp = desc_stamp" not in draw:
        raise SystemExit("the descriptors do not remember the stamp they were built at")
    reads = block(draw, "bool bank_reads_table(")
    bkept = block(draw, "bool bank_kept(")
    kept = block(bank, "bool pointer_table_kept(")
    fixture = f'''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do {{ if (!(x)) {{ std::fprintf(stderr, "FAIL:%d %s\\n", __LINE__, #x); std::exit(1); }} }} while (0)
{kept}
{every}
{reads}
{bkept}
int main(int argc, char** argv) {{
    int a = 0, b = 0;
    CHECK(pointer_table_kept(5, 9, &a, 6, 5, 9, &a, 6));
    CHECK(!pointer_table_kept(5, 10, &a, 6, 5, 9, &a, 6));
    CHECK(!pointer_table_kept(6, 9, &a, 6, 5, 9, &a, 6));
    CHECK(!pointer_table_kept(5, 9, &b, 6, 5, 9, &a, 6));
    CHECK(!pointer_table_kept(5, 9, &a, 7, 5, 9, &a, 6));
    CHECK(!pointer_table_kept(~0ull, ~0ull, nullptr, -1, 5, 9, &a, 6));
    CHECK(bank_reads_table(0, 5, 0x890) && bank_reads_table(0, 6, 0x890) && bank_reads_table(0, 7, 0xA00));
    CHECK(!bank_reads_table(0, 6, 0x880) && !bank_reads_table(2, 6, 0xA00) && !bank_reads_table(0, 0, 0xA00));
    CHECK(!bank_reads_table(0, 1, 0xA00) && !bank_reads_table(0, 4, 0xA00));
    CHECK(bank_kept(5, 5, true, 9, 9) && !bank_kept(5, 5, true, 10, 9) && !bank_kept(6, 5, true, 9, 9));
    CHECK(bank_kept(5, 5, false, 10, 9) && !bank_kept(6, 5, false, 9, 9));
    CHECK(argc == 2);
    CHECK(bind_every() == (argv[1][0] == '1'));
    std::printf("PASS: the pointer table, its bank and descriptors are read again for new pools or a new submission (RX_VK_BIND_EVERY %s)\\n", argv[1]);
}}
'''
    with tempfile.TemporaryDirectory(prefix="pointer-table-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        exe = Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        env = {k: v for k, v in os.environ.items() if k != "RX_VK_BIND_EVERY"}
        subprocess.run([str(exe), "0"], check=True, env=env)
        subprocess.run([str(exe), "1"], check=True, env={**env, "RX_VK_BIND_EVERY": "1"})


if __name__ == "__main__":
    main()
