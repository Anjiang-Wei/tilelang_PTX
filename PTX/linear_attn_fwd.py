#!/usr/bin/env python3
"""
Generate two equivalent CUDA kernel sources for Linear Attention Forward by running 
the same kernel with two different configurations and extracting the PTX.

Based on the working example in examples/linear_attention/example_linear_attn_fwd.py
"""

import torch
import tilelang as tl
import tilelang.language as T
import argparse
import os
import pathlib
import shutil
import subprocess


@tl.jit(
    out_idx=[3, 4],
    pass_configs={
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True
    })
def chunk_linear_attn_fwd_kernel_v1(
    B,
    S,
    H,
    DK,
    DV,
    dtype: str = 'float16',
    scale: float = None,
) -> torch.Tensor:

    if scale is None:
        scale = DK**-0.5
    accum_dtype = 'float'

    chunk_size = 64
    BK = BV = 64  # Variant 1: Use 64
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def chunk_linear_attn_fwd(
            Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
            O: T.Tensor([NK, B, S, H, DV], dtype),  # type: ignore
            final_state: T.Tensor([B, H, DK, DV], accum_dtype)):  # type: ignore
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            # Layout annotations without full swizzle for v1
            # Only basic swizzle
            T.use_swizzle(10)

            for i in T.Pipelined(0, NT, num_stages=1):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)

                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(k, v, h, transpose_A=True)
                T.gemm(q, h_shared, o)
                T.copy(
                    o, O[i_k, i_b, i * chunk_size:(i + 1) * chunk_size, i_h,
                         i_v * BV:(i_v + 1) * BV])

            # Output final state
            T.copy(h, final_state[i_b, i_h, i_k * BK:(i_k + 1) * BK, i_v * BV:(i_v + 1) * BV])

    return chunk_linear_attn_fwd


@tl.jit(
    out_idx=[3, 4],
    pass_configs={
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True
    })
def chunk_linear_attn_fwd_kernel_v2(
    B,
    S,
    H,
    DK,
    DV,
    dtype: str = 'float16',
    scale: float = None,
) -> torch.Tensor:

    if scale is None:
        scale = DK**-0.5
    accum_dtype = 'float'

    chunk_size = 64
    BK = BV = 32  # Variant 2: Use 32 for different memory layout
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def chunk_linear_attn_fwd(
            Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
            O: T.Tensor([NK, B, S, H, DV], dtype),  # type: ignore
            final_state: T.Tensor([B, H, DK, DV], accum_dtype)):  # type: ignore
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            # Layout annotations with full swizzle for v2
            T.annotate_layout({
                q: tl.layout.make_swizzled_layout(q),
                k: tl.layout.make_swizzled_layout(k),
                v: tl.layout.make_swizzled_layout(v),
                h_shared: tl.layout.make_swizzled_layout(h_shared),
                s_shared: tl.layout.make_swizzled_layout(s_shared),
            })
            T.use_swizzle(10)

            for i in T.Pipelined(0, NT, num_stages=2):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)

                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(k, v, h, transpose_A=True)
                T.gemm(q, h_shared, o)
                T.copy(
                    o, O[i_k, i_b, i * chunk_size:(i + 1) * chunk_size, i_h,
                         i_v * BV:(i_v + 1) * BV])

            # Output final state
            T.copy(h, final_state[i_b, i_h, i_k * BK:(i_k + 1) * BK, i_v * BV:(i_v + 1) * BV])

    return chunk_linear_attn_fwd


def postprocess(o, h):
    o = o[0] if o.size(0) == 1 else o.sum(0)
    return o, h


def ensure_outdir(path: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit_cuda_sources(B: int, S: int, H: int, D: int, outdir: pathlib.Path, 
                      arch_sm: int | None, do_compile: bool) -> None:
    
    variants = [
        ("v1_basic", chunk_linear_attn_fwd_kernel_v1),
        ("v2_swizzled", chunk_linear_attn_fwd_kernel_v2),
    ]
    
    for idx, (name, variant_func) in enumerate(variants, 1):
        print(f"[Compile] {idx}/{len(variants)}  {name}")
        
        # Generate kernel
        jit_kernel = variant_func(B, S, H, D, D)

        # Test the kernel to trigger compilation
        q = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
        k = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
        v = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
        
        try:
            o, h = postprocess(*jit_kernel(q, k, v))
            print(f"  [INFO] Generated output with shapes O:{o.shape}, final_state:{h.shape}")
        except Exception as e:
            print(f"  [WARN] kernel test failed: {e}; continuing with source extraction...")

        # Extract CUDA source and host wrapper
        cu_src = jit_kernel.get_kernel_source()
        try:
            host_src = jit_kernel.get_host_source()
        except AttributeError:
            # If host source is not available, create a minimal placeholder
            host_src = f"""// Host wrapper for {name}
// This is a placeholder - the actual host code would be generated by TileLang
#include <cuda_runtime.h>
#include <iostream>

// Kernel launch wrapper would be generated here
void launch_{name}_kernel() {{
    std::cout << "Kernel {name} launch placeholder" << std::endl;
}}
"""

        # Write files
        base = f"linear_attn_fwd_{name}"
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
    p = argparse.ArgumentParser(description="Generate two equivalent CUDA sources for Linear Attention Forward")
    p.add_argument("--B", type=int, default=2, help="Batch size")
    p.add_argument("--S", type=int, default=128, help="Sequence length") 
    p.add_argument("--H", type=int, default=4, help="Number of heads")
    p.add_argument("--D", type=int, default=64, help="Head dimension")
    p.add_argument("--outdir", type=str, default="build/linear_attn_fwd")

    # Build options  
    p.add_argument("--compile", action="store_true", default=True, help="Also build PTX with nvcc for each variant")
    p.add_argument("--sm", type=int, default=80, help="SM architecture for nvcc (e.g., 80 for A100, 90 for H100)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    print(f"Generating 2 equivalent kernel variants")

    emit_cuda_sources(
        B=args.B,
        S=args.S,
        H=args.H,
        D=args.D,
        outdir=outdir,
        arch_sm=args.sm,
        do_compile=args.compile,
    )


if __name__ == "__main__":
    main()