#!/usr/bin/env python3
"""
Generate multiple CUDA kernel sources for Conv2D using TileLang by sweeping
configurations, then optionally compile each to PTX (sm_80 or sm_90).

Example:
  python conv2d_codegen.py \
      --N 128 --C 128 --H 64 --W 64 --F 128 --K 3 --S 1 --P 1 \
      --outdir build/conv2d_variants \
      --compile --sm 80
"""

import argparse
import dataclasses
import itertools
import os
import pathlib
import shutil
import subprocess

import torch
import tilelang as tl
import tilelang.language as T


# -------------------------------------------------------------------------------------
# Helper dataclass for configurations
# -------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Conv2dConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    thread_num: int
    enable_rasteration: bool

    def short(self) -> str:
        ras = "ras" if self.enable_rasteration else "nor"
        return f"BM{self.block_M}_BN{self.block_N}_BK{self.block_K}_S{self.num_stages}_T{self.thread_num}_{ras}"

    def to_kwargs(self):
        return dataclasses.asdict(self)


# -------------------------------------------------------------------------------------
# Core Conv2D kernel factory
# -------------------------------------------------------------------------------------
def check_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (9, 0)


def build_conv2d_kernel(N, C, H, W, F, K, S, D, P):
    """Return a callable that constructs a prim_func given a config."""
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
        is_hopper = check_hopper()
        KH, KW = K, K
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        @T.prim_func
        def main(
            data: T.Tensor((N, H, W, C), dtype),
            kernel: T.Tensor((KH, KW, C, F), dtype),
            out: T.Tensor((N, OH, OW, F), dtype),
        ):
            with T.Kernel(
                T.ceildiv(F, block_N),
                T.ceildiv(N * OH * OW, block_M),
                threads=thread_num,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared((block_M, block_N), dtype)

                kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
                out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

                T.annotate_layout({
                    out_shared: tl.layout.make_swizzled_layout(out_shared),
                    data_shared: tl.layout.make_swizzled_layout(data_shared),
                    kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
                })

                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                    if is_hopper:
                        T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                    else:
                        for i, j in T.Parallel(block_M, block_K):
                            k = k_iter * block_K + j
                            m = by * block_M + i
                            access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                            access_w = m % OW * S + k // C % KW * D - P
                            in_bound = (
                                (access_h >= 0)
                                and (access_w >= 0)
                                and (access_h < H)
                                and (access_w < W)
                            )
                            data_shared[i, j] = T.if_then_else(
                                in_bound,
                                data[m // (OH * OW), access_h, access_w, k % C],
                                0,
                            )
                    T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                    T.gemm(data_shared, kernel_shared, out_local)

                T.copy(out_local, out_shared)
                T.copy(out_shared, out_flat[by * block_M, bx * block_N])

        return main

    return kernel


# -------------------------------------------------------------------------------------
# Config sweeps
# -------------------------------------------------------------------------------------
def generate_configs_grid():
    grid = itertools.product(
        [64, 128],     # block_M
        [128, 256],    # block_N
        [32, 64],      # block_K
        [1],        # num_stages
        [128],    # thread_num
        [False], # enable_rasteration
    )
    cfgs = []
    for bm, bn, bk, s, t, r in grid:
        if t > 1024:
            continue
        cfgs.append(Conv2dConfig(bm, bn, bk, s, t, r))
    return cfgs[:2]  # Only two variants for demonstration


# -------------------------------------------------------------------------------------
# Output management + nvcc compilation
# -------------------------------------------------------------------------------------
def ensure_outdir(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit_cuda_sources(
    N, C, H, W, F, K, S, D, P, cfgs, outdir, arch_sm=80, do_compile=True
):
    kernel_factory = build_conv2d_kernel(N, C, H, W, F, K, S, D, P)

    for idx, cfg in enumerate(cfgs, 1):
        print(f"[Generate] {idx}/{len(cfgs)} {cfg.short()}")
        prim = kernel_factory(**cfg.to_kwargs())

        jit_kernel = tl.JITKernel(prim, out_idx=[2], target="cuda")

        # Extract CUDA and host source
        cu_src = jit_kernel.get_kernel_source()
        host_src = jit_kernel.get_host_source()

        base = f"conv2d_{cfg.short()}"
        cu_path = outdir / f"{base}.cu"
        host_path = outdir / f"{base}_host.cpp"

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cu_src)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(host_src)

        print(f"  CUDA source written to {cu_path}")

        # Optionally compile to PTX
        if do_compile:
            if shutil.which("nvcc") is None:
                print("  [WARN] nvcc not found; skipping PTX build.")
                continue
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
            print("  [nvcc]", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(f"  PTX compiled to {ptx_path}")
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] nvcc failed for {base}: {e}")


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate CUDA/PTX variants for Conv2D.")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--C", type=int, default=128)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--F", type=int, default=128)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--D", type=int, default=1)
    p.add_argument("--P", type=int, default=1)
    p.add_argument("--outdir", type=str, default="build/conv2d")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--sm", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    cfgs = generate_configs_grid()
    emit_cuda_sources(
        args.N, args.C, args.H, args.W, args.F, args.K, args.S, args.D, args.P,
        cfgs=cfgs,
        outdir=outdir,
        arch_sm=args.sm,
        do_compile=args.compile,
    )


if __name__ == "__main__":
    main()
