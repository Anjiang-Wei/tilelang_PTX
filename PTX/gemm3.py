#!/usr/bin/env python3
"""
Collect multiple equivalent CUDA kernel sources for GEMM using TileLang by sweeping (or auto-generating)
multiple schedule/config parameters, then optionally compile each source to PTX (sm_80).

Requirements:
  - tilelang >= 0.1.5
  - CUDA toolkit (nvcc) available in PATH, if --compile is used
  - PyTorch (optional, for quick correctness spot-checks with --check)

Example:
  python gemm.py \
      --M 4096 --N 4096 --K 4096 \
      --topk 8 \
      --use-carver \
      --outdir build/cuda_variants \
      --compile --sm 80

This script purposefully compiles *many* schedule variants for the same GEMM so the generated CUDA sources
are equivalent in semantics but differ in tiling, pipelining, etc. Use smaller sizes while testing.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

# --- Optional deps for correctness checks
try:
    import torch
except Exception:
    torch = None

# --- TileLang imports
import tilelang as tl
import tilelang.language as T
import tilelang.utils.target as CUDA

try:
    # Carver (hint generator) is optional; guarded behind a flag
    from tilelang.carver.template.matmul import MatmulTemplate
except Exception:
    MatmulTemplate = None


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    thread_num: int
    enable_rasteration: bool

    def short(self) -> str:
        ras = "ras" if self.enable_rasteration else "nor"
        return f"BM{self.block_M}_BN{self.block_N}_BK{self.block_K}_S{self.num_stages}_T{self.thread_num}_{ras}"

    def to_kwargs(self) -> Dict[str, int | bool]:
        return dataclasses.asdict(self)


# ---------------- Kernel factory (adapted from TileLang GEMM example with reserved params) ----------------
# We keep the kernel body minimal; the *schedule* comes from the parameters above.

def build_kernel(M: int, N: int, K: int):
    """Return a Python callable `kernel(**config)` that yields a prim_func with reserved params.
    This mirrors the pattern in TileLang tutorials where we reserve tuning parameters.
    """

    def kernel(
        block_M: int,
        block_N: int,
        block_K: int,
        num_stages: int,
        thread_num: int,
        enable_rasteration: bool,
    ):
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


# ---------------- Candidate config generation ----------------

def generate_configs_grid() -> List[GemmConfig]:
    # Small grid with only 2-3 examples for testing
    grid = itertools.product(
        [32, 64],             # block_M
        [64],              # block_N
        [32],              # block_K
        [0],               # num_stages
        [256],             # thread_num
        [False],           # enable_rasteration
    )
    cfgs = []
    for (bm, bn, bk, s, t, r) in grid:
        # A light sanity filter to keep thread count in a reasonable range
        if t > 1024:
            continue
        cfgs.append(GemmConfig(bm, bn, bk, s, t, r))
    return cfgs


def generate_configs_carver(M: int, N: int, K: int, topk: int) -> List[GemmConfig]:
    if MatmulTemplate is None:
        raise RuntimeError("Carver is not available in this TileLang build; omit --use-carver.")

    arch = CUDA("cuda")
    tmpl = MatmulTemplate(
        M=M, N=N, K=K, in_dtype="float16", out_dtype="float16", accum_dtype="float"
    ).with_arch(arch)

    hints = tmpl.recommend_hints(topk=topk)
    cfgs: List[GemmConfig] = []
    for h in hints:
        # Heuristic thread selection from hint's tile shape
        block_m, block_n = h.block_shape
        block_rows, block_cols = h.block_warps
        thread_num = block_rows * block_cols * 32
        cfgs.append(
            GemmConfig(
                block_M=block_m,
                block_N=block_n,
                block_K=h.rstep[0],
                num_stages=h.pipeline_stage,
                thread_num=thread_num,
                enable_rasteration=(h.rasterization_plan is not None),
            )
        )
    # Deduplicate just in case
    uniq = {c.short(): c for c in cfgs}
    return list(uniq.values())


# ---------------- Compilation, extraction, and (optional) PTX build ----------------

def ensure_outdir(path: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit_cuda_sources(M: int, N: int, K: int, cfgs: List[GemmConfig], outdir: pathlib.Path,
                      arch_sm: int | None, do_compile: bool, check: bool) -> None:
    kernel = build_kernel(M, N, K)

    # Optional quick numerical check harness
    if check:
        if torch is None:
            print("[WARN] --check requested but torch is not available; skipping checks.")
        else:
            torch.manual_seed(0)
            a_t = torch.randn(M, K, dtype=torch.float16, device="cuda")
            b_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
            ref = (a_t.to(torch.float32) @ b_t.t().to(torch.float32)).to(torch.float16).cpu()
    
    for idx, cfg in enumerate(cfgs, 1):
        print(f"[Compile] {idx}/{len(cfgs)}  {cfg.short()}")
        prim = kernel(**cfg.to_kwargs())

        # Build a JIT kernel and trigger compilation
        jit_kernel = tl.JITKernel(prim, out_idx=[-1], target="cuda")
        # Run a no-op launch that forces compile without allocating big real tensors
        # (we can rely on the JIT compile stage without running if inputs are not supplied)
        if check and torch is not None:
            # Numerical correctness check with actual matrix dimensions
            try:
                a = torch.randn(M, K, dtype=torch.float16, device="cuda")
                b = torch.randn(N, K, dtype=torch.float16, device="cuda")
                out = jit_kernel(a, b)
                ref = (a.to(torch.float32) @ b.t().to(torch.float32)).to(torch.float16)
                max_diff = (out - ref).abs().max().item()
                if max_diff > 1e-2:
                    print(f"  [WARN] numerical diff {max_diff:.3e}; keeping source anyway.")
            except Exception as e:
                print(f"  [WARN] numerical check failed: {e}; keeping source anyway.")

        # Extract CUDA source and host wrapper
        cu_src = jit_kernel.get_kernel_source()
        host_src = jit_kernel.get_host_source()

        # Write files
        base = f"gemm_{cfg.short()}"
        cu_path = outdir / f"{base}.cu"
        host_path = outdir / f"{base}_host.cpp"
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cu_src)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(host_src)

        # Optionally compile to PTX
        if do_compile:
            if shutil.which("nvcc") is None:
                print("  [ERROR] nvcc not found; skipping PTX build for this variant.")
            else:
                # Get TileLang include paths
                import tilelang
                tl_include_path = os.path.join(os.path.dirname(tilelang.__file__), "src")
                cutlass_include_path = os.path.join(os.path.dirname(tilelang.__file__), "3rdparty", "cutlass", "include")
                
                sm = arch_sm if arch_sm is not None else 80
                ptx_path = outdir / f"{base}.ptx"
                cmd = [
                    "nvcc", "-std=c++17", "-O3", "-Xptxas", "-v",
                    f"-I{tl_include_path}",
                    f"-I{cutlass_include_path}",
                    f"-arch=sm_{sm}", "-ptx", str(cu_path), "-o", str(ptx_path)
                ]
                print("  [nvcc] ", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [ERROR] nvcc failed for {base}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multiple equivalent CUDA sources for GEMM via TileLang configs.")
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--outdir", type=str, default="build/gemm3")

    # Config sourcing
    p.add_argument("--use-carver", action="store_true", help="Use Carver to recommend top-k configs (requires TileLang with Carver)")
    p.add_argument("--topk", type=int, default=8, help="Top-k configs to fetch from Carver when --use-carver is set")
    p.add_argument("--grid", default=True, type=bool, help="Also include a small manual grid sweep of configs")

    # Build options  
    p.add_argument("--compile", action="store_true", default=True, help="Also build PTX with nvcc for each variant")
    p.add_argument("--sm", type=int, default=80, help="SM architecture for nvcc (e.g., 80 for A100, 90 for H100)")
    p.add_argument("--check", action="store_true", default=True, help="Run numerical correctness check using PyTorch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    cfgs: List[GemmConfig] = []
    if args.use_carver:
        cfgs.extend(generate_configs_carver(args.M, args.N, args.K, args.topk))
    if args.grid or not cfgs:
        cfgs.extend(generate_configs_grid())

    # Deduplicate by short-name
    cfgs = list({c.short(): c for c in cfgs}.values())
    print(f"Total candidate configs: {len(cfgs)}")

    emit_cuda_sources(
        M=args.M,
        N=args.N,
        K=args.K,
        cfgs=cfgs,
        outdir=outdir,
        arch_sm=args.sm,
        do_compile=args.compile,
        check=args.check,
    )


if __name__ == "__main__":
    main()
