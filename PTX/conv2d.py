#!/usr/bin/env python3
"""
Generate two Conv2D CUDA/PTX variants with identical CTA tiles (64x64):
  1) TC    : tensor-core GEMM (FP16 inputs, FP32 accum)
  2) BASIC : SIMT FMA micro-kernel (no tensor cores)

Design goals:
- Same work per CTA (64x64 tile), suitable for equivalence checking.
- Avoid cp.async in emitted PTX by:
  * Removing pipelined copies
  * Replacing T.copy(...) with manual element-wise loads into shared memory
"""

import argparse
import dataclasses
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
    use_tc: bool  # NEW: True for Tensor Cores, False for basic SIMT

    def short(self) -> str:
        ras = "ras" if self.enable_rasteration else "nor"
        core = "tc" if self.use_tc else "basic"
        return f"{core}_BM{self.block_M}_BN{self.block_N}_BK{self.block_K}_S{self.num_stages}_T{self.thread_num}_{ras}"

    def to_kwargs(self):
        return dataclasses.asdict(self)


# -------------------------------------------------------------------------------------
# Core Conv2D kernel factory
# -------------------------------------------------------------------------------------
def build_conv2d_kernel(N, C, H, W, F, K, S, D, P):
    """Return a callable that constructs a prim_func given a config."""
    def kernel(
        block_M: int,
        block_N: int,
        block_K: int,
        num_stages: int,
        thread_num: int,
        enable_rasteration: bool,
        use_tc: bool,
    ):
        # Types: TC uses fp16->fp32; BASIC also accumulates in fp32.
        dtype = "float16"
        accum_dtype = "float"

        KH, KW = K, K
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        @T.prim_func
        def main(
            data: T.Tensor((N, H, W, C), dtype),
            kernel: T.Tensor((KH, KW, C, F), dtype),
            out: T.Tensor((N, OH, OW, F), dtype),
        ):
            # Grid: (F / block_N) x ((N*OH*OW) / block_M)
            with T.Kernel(
                T.ceildiv(F, block_N),
                T.ceildiv(N * OH * OW, block_M),
                threads=thread_num,
            ) as (bx, by):
                # Shared tiles
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                w_shared    = T.alloc_shared((block_K, block_N), dtype)

                # Accumulator & an intermediate shared buffer for coalesced store-out
                out_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared   ((block_M, block_N), dtype)

                # Flattened logical tensors
                w_flat   = T.Tensor((KH * KW * C, F), dtype, kernel.data)
                out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

                # Prefer simple layouts; keep deterministic addressing
                T.clear(out_local)

                # Number of K tiles for im2col * channels
                ktiles = T.ceildiv(KH * KW * C, block_K)

                # ---- K-TILE LOOP (no pipelining; no cp.async) ----
                for kt in range(ktiles):
                    # ---- Load ACTIVATION tile to shared (manual element-wise; NO T.copy) ----
                    # Each CTA computes rows [by*block_M ..) of the im2col'ed matrix (M dimension).
                    for i, j in T.Parallel(block_M, block_K):
                        k_idx = kt * block_K + j       # [0 .. KH*KW*C)
                        m_idx = by * block_M + i       # row into (N*OH*OW)

                        # Map (m_idx, k_idx) back to (n, oh, ow, c, kh, kw)
                        # m_idx -> (n, oh, ow)
                        n  = m_idx // (OH * OW)
                        oh = (m_idx % (OH * OW)) // OW
                        ow = (m_idx % (OH * OW)) %  OW

                        # k_idx -> (kh, kw, c)
                        tmp   = k_idx // C
                        cc    = k_idx % C
                        kh    = tmp // KW
                        kw    = tmp %  KW

                        ih = oh * S + kh * D - P
                        iw = ow * S + kw * D - P
                        inb = (0 <= ih) and (ih < H) and (0 <= iw) and (iw < W) and (k_idx < KH*KW*C)

                        data_shared[i, j] = T.if_then_else(
                            inb,
                            data[n, ih, iw, cc],
                            T.cast(0, dtype)
                        )

                    # ---- Load WEIGHT tile to shared (manual element-wise; NO T.copy) ----
                    # w_tile = w_flat[kt*block_K : (kt+1)*block_K, bx*block_N : bx*block_N+block_N]
                    base_k = kt * block_K
                    base_n = bx * block_N
                    for kk, nn in T.Parallel(block_K, block_N):
                        src_k = base_k + kk
                        src_n = base_n + nn
                        inb_w = (src_k < KH*KW*C) and (src_n < F)
                        w_shared[kk, nn] = T.if_then_else(
                            inb_w,
                            w_flat[src_k, src_n],
                            T.cast(0, dtype)
                        )

                    T.builtin.sync_threads()

                    # ---- Compute: TC vs BASIC ----
                    if use_tc:
                        # Tensor-core path (FP16->FP32 GEMM)
                        # This will typically lower to WMMA / wgmma depending on arch & dtype.
                        T.gemm(data_shared, w_shared, out_local)  # accumulate into out_local
                    else:
                        # BASIC SIMT FMA micro-kernel (no tensor cores).
                        # out_local[m,n] += sum_k data_shared[m,k] * w_shared[k,n]
                        for kk in range(block_K):
                            for m, n2 in T.Parallel(block_M, block_N):
                                out_local[m, n2] = out_local[m, n2] + T.cast(
                                    data_shared[m, kk], accum_dtype
                                ) * T.cast(w_shared[kk, n2], accum_dtype)

                    T.builtin.sync_threads()

                # ---- Store out_local -> out (no async copy) ----
                # Cast down to dtype then write to global
                for i, j in T.Parallel(block_M, block_N):
                    out_shared[i, j] = T.cast(out_local[i, j], dtype)
                T.builtin.sync_threads()

                # Write back to global out matrix at (by*block_M, bx*block_N)
                base_m = by * block_M
                base_n = bx * block_N
                for i, j in T.Parallel(block_M, block_N):
                    if (base_m + i) < (N * OH * OW) and (base_n + j) < F:
                        out_flat[base_m + i, base_n + j] = out_shared[i, j]

        return main

    return kernel


# -------------------------------------------------------------------------------------
# Two fixed configs: identical tiles; one TC, one BASIC
# -------------------------------------------------------------------------------------
def generate_configs_pair():
    # Keep identical M,N (<=64) as Rahul prefers; same K-tile for both
    bm, bn, bk = 64, 64, 32
    threads = 128
    return [
        Conv2dConfig(bm, bn, bk, 1, threads, False, True),   # TC
        Conv2dConfig(bm, bn, bk, 1, threads, False, False),  # BASIC (no TC)
    ]


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
    p = argparse.ArgumentParser(description="Generate CUDA/PTX variants for Conv2D (blog config defaults).")
    # Blog defaults: N=100, C=3, H=W=224, F=96, K=11, S=4, P=2, D=1
    p.add_argument("--N", type=int, default=100, help="Batch size")
    p.add_argument("--C", type=int, default=3,   help="Input channels")
    p.add_argument("--H", type=int, default=224, help="Input height")
    p.add_argument("--W", type=int, default=224, help="Input width")
    p.add_argument("--F", type=int, default=96,  help="Output channels")
    p.add_argument("--K", type=int, default=11,  help="Kernel size (square)")
    p.add_argument("--S", type=int, default=4,   help="Stride")
    p.add_argument("--D", type=int, default=1,   help="Dilation")
    p.add_argument("--P", type=int, default=2,   help="Padding")
    p.add_argument("--outdir", type=str, default="build/conv2d")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--sm", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    cfgs = generate_configs_pair()
    emit_cuda_sources(
        args.N, args.C, args.H, args.W, args.F, args.K, args.S, args.D, args.P,
        cfgs=cfgs,
        outdir=outdir,
        arch_sm=args.sm,
        do_compile=args.compile,
    )


if __name__ == "__main__":
    main()
