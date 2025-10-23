#!/usr/bin/env python3
"""
Generate two equivalent GEMM variants (TC vs BASIC) with NO cp.async.

Both kernels:
  - Same CTA tile: block_M=64, block_N=64, block_K=32
  - Same threads per block: 128
  - Same num_stages: 1
Differences:
  - TC variant uses T.gemm(..., transpose_B=True) -> WMMA/WGMMA (tensor cores)
  - BASIC variant uses explicit SIMT FMA loops (no tensor cores)

We avoid cp.async by:
  - Not using T.Pipelined(...)
  - Not using T.copy(...) for global->shared
  - Doing manual element loads and CTA barriers
"""

from __future__ import annotations
import argparse, dataclasses, os, pathlib, shutil, subprocess
from typing import Dict, List

try:
    import torch
except Exception:
    torch = None

import tilelang as tl
import tilelang.language as T

# ---------------- Config ----------------
@dataclasses.dataclass(frozen=True)
class GemmConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    thread_num: int
    use_tc: bool

    def short(self) -> str:
        core = "tc" if self.use_tc else "basic"
        return f"{core}_BM{self.block_M}_BN{self.block_N}_BK{self.block_K}_S{self.num_stages}_T{self.thread_num}"

    def to_kwargs(self) -> Dict[str, int | bool]:
        return dataclasses.asdict(self)


def generate_tc_basic_pair() -> List[GemmConfig]:
    return [
        GemmConfig(32, 32, 32, 1, 128, True),   # TC
        GemmConfig(32, 32, 32, 1, 128, False),  # BASIC
    ]


# ---------------- Kernel factory (no cp.async) ----------------
def build_kernel(M: int, N: int, K: int):
    def kernel(block_M: int, block_N: int, block_K: int, num_stages: int, thread_num: int, use_tc: bool):
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),   # row-major
            B: T.Tensor((N, K), dtype),   # row-major; we will treat as (n,k) and use transpose_B=True for GEMM
            C: T.Tensor((M, N), dtype),   # row-major
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_sh = T.alloc_shared((block_M, block_K), dtype)
                B_sh = T.alloc_shared((block_N, block_K), dtype)
                C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_loc)

                # number of K tiles
                kt_max = T.ceildiv(K, block_K)
                for kt in range(kt_max):
                    # ---- manual global -> shared for A_sh ----
                    base_m = by * block_M
                    base_k = kt * block_K
                    for i, kk in T.Parallel(block_M, block_K):
                        g_m = base_m + i
                        g_k = base_k + kk
                        inb = (g_m < M) and (g_k < K)
                        A_sh[i, kk] = T.if_then_else(inb, A[g_m, g_k], T.cast(0, dtype))

                    # ---- manual global -> shared for B_sh (note: B is (N,K)) ----
                    base_n = bx * block_N
                    for j, kk in T.Parallel(block_N, block_K):
                        g_n = base_n + j
                        g_k = base_k + kk
                        inb = (g_n < N) and (g_k < K)
                        B_sh[j, kk] = T.if_then_else(inb, B[g_n, g_k], T.cast(0, dtype))

                    T.builtin.sync_threads()  # CTA barrier before using shared tiles

                    # ---- compute ----
                    if use_tc:
                        # Tensor core path (A_sh [M x Kt]) * (B_sh^T [Kt x N]) -> C_loc [M x N]
                        T.gemm(A_sh, B_sh, C_loc, transpose_B=True)
                    else:
                        # BASIC SIMT: C_loc[i,j] += sum_kk A_sh[i,kk] * B_sh[j,kk]
                        for kk in range(block_K):
                            for i, j in T.Parallel(block_M, block_N):
                                C_loc[i, j] = C_loc[i, j] + \
                                    T.cast(A_sh[i, kk], accum_dtype) * T.cast(B_sh[j, kk], accum_dtype)

                    T.builtin.sync_threads()  # safe to reuse shared tiles next iteration

                # ---- store C_loc -> C (manual, elementwise) ----
                for i, j in T.Parallel(block_M, block_N):
                    g_m = by * block_M + i
                    g_n = bx * block_N + j
                    if (g_m < M) and (g_n < N):
                        C[g_m, g_n] = T.cast(C_loc[i, j], dtype)

        return main

    return kernel


# ---------------- Build & emit ----------------
def ensure_outdir(path: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit_cuda_sources(M: int, N: int, K: int, outdir: pathlib.Path, arch_sm: int, do_compile: bool, check: bool) -> None:
    kernel_factory = build_kernel(M, N, K)
    cfgs = generate_tc_basic_pair()

    if check and torch is not None:
        torch.manual_seed(0)

    for idx, cfg in enumerate(cfgs, 1):
        print(f"[Compile] {idx}/2  {cfg.short()}")
        prim = kernel_factory(**cfg.to_kwargs())
        jit_kernel = tl.JITKernel(prim, out_idx=[-1], target="cuda")

        # optional quick check
        if check and torch is not None:
            try:
                a = torch.randn(M, K, dtype=torch.float16, device="cuda")
                b = torch.randn(N, K, dtype=torch.float16, device="cuda")
                out = jit_kernel(a, b)
                ref = (a.to(torch.float32) @ b.t().to(torch.float32)).to(torch.float16)
                max_diff = (out - ref).abs().max().item()
                if max_diff > 1e-2:
                    print(f"  [WARN] numerical diff {max_diff:.3e}")
            except Exception as e:
                print(f"  [WARN] numerical check failed: {e}")

        cu_src = jit_kernel.get_kernel_source()
        host_src = jit_kernel.get_host_source()
        base = f"gemm_{cfg.short()}"
        cu_path = outdir / f"{base}.cu"
        host_path = outdir / f"{base}_host.cpp"
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cu_src)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(host_src)

        if do_compile:
            if shutil.which("nvcc") is None:
                print("  [ERROR] nvcc not found; skipping PTX build.")
            else:
                import tilelang
                tl_inc = os.path.join(os.path.dirname(tilelang.__file__), "src")
                cutlass_inc = os.path.join(os.path.dirname(tilelang.__file__), "3rdparty", "cutlass", "include")
                ptx_path = outdir / f"{base}.ptx"
                cmd = [
                    "nvcc", "-std=c++17", "-O3", "-Xptxas", "-v",
                    f"-I{tl_inc}",
                    f"-I{cutlass_inc}",
                    f"-arch=sm_{arch_sm}", "-ptx", str(cu_path), "-o", str(ptx_path)
                ]
                print("  [nvcc] ", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [ERROR] nvcc failed for {base}: {e}")


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two GEMM variants (TC vs BASIC), no cp.async.")
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--outdir", type=str, default="build/gemm1")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--sm", type=int, default=80, help="SM arch (80=A100, 90=H100)")
    p.add_argument("--check", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    emit_cuda_sources(
        M=args.M, N=args.N, K=args.K,
        outdir=outdir, arch_sm=args.sm,
        do_compile=args.compile, check=args.check
    )


if __name__ == "__main__":
    main()
