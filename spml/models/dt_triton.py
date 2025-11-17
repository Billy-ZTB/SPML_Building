import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ------------------------------------------------------------------
# Production-Ready Triton Domain Transform Implementation
# ------------------------------------------------------------------

@triton.jit
def _domain_transform_ltr_kernel(
        x_ptr, w_ptr, y_ptr, DIM_SCAN, BLOCK_SIZE: tl.constexpr,
):
    """Left-to-right scan kernel."""
    pid = tl.program_id(0)

    for i in range(DIM_SCAN):
        if i < BLOCK_SIZE:
            x_i = tl.load(x_ptr + pid * DIM_SCAN + i)
            w_i = tl.load(w_ptr + pid * DIM_SCAN + i)

            if i == 0:
                y_i = x_i
            else:
                y_prev = tl.load(y_ptr + pid * DIM_SCAN + (i - 1))
                y_i = (1.0 - w_i) * x_i + w_i * y_prev

            tl.store(y_ptr + pid * DIM_SCAN + i, y_i)


@triton.jit
def _domain_transform_rtl_kernel(
        x_ptr, w_ptr, y_ptr, DIM_SCAN, BLOCK_SIZE: tl.constexpr,
):
    """Right-to-left scan kernel."""
    pid = tl.program_id(0)

    for i in range(DIM_SCAN - 1, -1, -1):
        if i < BLOCK_SIZE:
            x_i = tl.load(x_ptr + pid * DIM_SCAN + i)

            if i == DIM_SCAN - 1:
                y_i = x_i
            else:
                w_i = tl.load(w_ptr + pid * DIM_SCAN + i)
                y_next = tl.load(y_ptr + pid * DIM_SCAN + (i + 1))
                y_i = (1.0 - w_i) * x_i + w_i * y_next

            tl.store(y_ptr + pid * DIM_SCAN + i, y_i)


def _apply_horizontal_ltr(features: torch.Tensor, w: torch.Tensor):
    """Apply left-to-right horizontal scan."""
    N, C, H, W = features.shape
    y = torch.empty_like(features)

    x_reshaped = features.reshape(-1, W).contiguous()
    w_broadcasted = w.expand(-1, C, -1, -1).reshape(-1, W).contiguous()
    y_reshaped = y.reshape(-1, W).contiguous()

    grid = (x_reshaped.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(W)

    _domain_transform_ltr_kernel[grid](
        x_reshaped, w_broadcasted, y_reshaped,
        W, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


def _apply_horizontal_rtl(features: torch.Tensor, w: torch.Tensor):
    """Apply right-to-left horizontal scan."""
    N, C, H, W = features.shape
    y = torch.empty_like(features)

    x_reshaped = features.reshape(-1, W).contiguous()
    # Shift weights for RTL: use w[i+1] at position i
    w_shifted = torch.zeros_like(w)
    w_shifted[:, :, :, :-1] = w[:, :, :, 1:]
    w_broadcasted = w_shifted.expand(-1, C, -1, -1).reshape(-1, W).contiguous()
    y_reshaped = y.reshape(-1, W).contiguous()

    grid = (x_reshaped.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(W)

    _domain_transform_rtl_kernel[grid](
        x_reshaped, w_broadcasted, y_reshaped,
        W, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


def _apply_vertical_ttb(features: torch.Tensor, w: torch.Tensor):
    """Apply top-to-bottom vertical scan."""
    N, C, H, W = features.shape

    # Transpose to make vertical dimension scannable
    x_transposed = features.permute(0, 1, 3, 2).reshape(-1, H).contiguous()
    w_transposed = w.expand(-1, C, -1, -1).permute(0, 1, 3, 2).reshape(-1, H).contiguous()
    y_transposed = torch.empty(N * C * W, H, device=features.device, dtype=features.dtype)

    grid = (x_transposed.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(H)

    _domain_transform_ltr_kernel[grid](
        x_transposed, w_transposed, y_transposed,
        H, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_transposed.reshape(N, C, W, H).permute(0, 1, 3, 2)


def _apply_vertical_btt(features: torch.Tensor, w: torch.Tensor):
    """Apply bottom-to-top vertical scan."""
    N, C, H, W = features.shape

    x_transposed = features.permute(0, 1, 3, 2).reshape(-1, H).contiguous()
    # Shift weights for BTT: use w[i+1] at position i
    w_shifted = torch.zeros_like(w)
    w_shifted[:, :, :-1, :] = w[:, :, 1:, :]
    w_transposed = w_shifted.expand(-1, C, -1, -1).permute(0, 1, 3, 2).reshape(-1, H).contiguous()
    y_transposed = torch.empty(N * C * W, H, device=features.device, dtype=features.dtype)

    grid = (x_transposed.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(H)

    _domain_transform_rtl_kernel[grid](
        x_transposed, w_transposed, y_transposed,
        H, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_transposed.reshape(N, C, W, H).permute(0, 1, 3, 2)

# Propagate along diagonal directions

def _apply_diag_trbl(features: torch.Tensor, w: torch.Tensor):
    """Apply Top-Right to Bottom-Left diagonal scan."""
    N, C, H, W = features.shape
    PAD_LEN = H - 1
    DIM_SCAN_DIAG = W + PAD_LEN
    DIM_SEQ_DIAG = H

    # 1. Pad (right)
    x_pad = F.pad(features, (0, PAD_LEN), 'constant', 0)
    w_expand = w.expand(-1, C, -1, -1)
    w_pad = F.pad(w_expand, (0, PAD_LEN), 'constant', 0)

    # 2. Skew (shear)
    x_skewed = torch.empty_like(x_pad)
    w_skewed = torch.empty_like(w_pad)
    for h_i in range(H):
        x_skewed[:, :, h_i, :] = torch.roll(x_pad[:, :, h_i, :], shifts=h_i, dims=-1)
        w_skewed[:, :, h_i, :] = torch.roll(w_pad[:, :, h_i, :], shifts=h_i, dims=-1)

    # 3. Permute, Reshape, and Scan (like vertical)
    x_scannable = x_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    w_scannable = w_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    y_scannable = torch.empty_like(x_scannable)

    grid = (x_scannable.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(DIM_SEQ_DIAG)
    _domain_transform_ltr_kernel[grid](
        x_scannable, w_scannable, y_scannable,
        DIM_SEQ_DIAG, BLOCK_SIZE=BLOCK_SIZE,
    )

    # 4. Un-permute
    y_skewed = y_scannable.reshape(N, C, DIM_SCAN_DIAG, DIM_SEQ_DIAG).permute(0, 1, 3, 2)

    # 5. Un-skew
    y_pad = torch.empty_like(x_pad)
    for h_i in range(H):
        y_pad[:, :, h_i, :] = torch.roll(y_skewed[:, :, h_i, :], shifts=-h_i, dims=-1)

    # 6. Un-pad
    return y_pad[:, :, :, :W].contiguous()


def _apply_diag_bltr(features: torch.Tensor, w: torch.Tensor):
    """Apply Bottom-Left to Top-Right diagonal scan."""
    N, C, H, W = features.shape
    PAD_LEN = H - 1
    DIM_SCAN_DIAG = W + PAD_LEN
    DIM_SEQ_DIAG = H

    # 1. Pad (right)
    x_pad = F.pad(features, (0, PAD_LEN), 'constant', 0)
    w_expand = w.expand(-1, C, -1, -1)
    w_pad = F.pad(w_expand, (0, PAD_LEN), 'constant', 0)

    # 2. Skew (shear)
    x_skewed = torch.empty_like(x_pad)
    w_skewed = torch.empty_like(w_pad)
    for h_i in range(H):
        x_skewed[:, :, h_i, :] = torch.roll(x_pad[:, :, h_i, :], shifts=h_i, dims=-1)
        w_skewed[:, :, h_i, :] = torch.roll(w_pad[:, :, h_i, :], shifts=h_i, dims=-1)

    # 3. Permute and Shift weights
    w_scannable_view = w_skewed.permute(0, 1, 3, 2)
    w_shifted = torch.zeros_like(w_scannable_view)
    w_shifted[..., :-1] = w_scannable_view[..., 1:] # Shift para RTL

    # 4. Reshape and Scan
    x_scannable = x_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    w_scannable = w_shifted.reshape(-1, DIM_SEQ_DIAG).contiguous()
    y_scannable = torch.empty_like(x_scannable)

    grid = (x_scannable.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(DIM_SEQ_DIAG)
    _domain_transform_rtl_kernel[grid](
        x_scannable, w_scannable, y_scannable,
        DIM_SEQ_DIAG, BLOCK_SIZE=BLOCK_SIZE,
    )

    # 5. Un-permute
    y_skewed = y_scannable.reshape(N, C, DIM_SCAN_DIAG, DIM_SEQ_DIAG).permute(0, 1, 3, 2)

    # 6. Un-skew
    y_pad = torch.empty_like(x_pad)
    for h_i in range(H):
        y_pad[:, :, h_i, :] = torch.roll(y_skewed[:, :, h_i, :], shifts=-h_i, dims=-1)

    # 7. Un-pad
    return y_pad[:, :, :, :W].contiguous()


def _apply_diag_tlbr(features: torch.Tensor, w: torch.Tensor):
    """Apply Top-Left to Bottom-Right diagonal scan."""
    N, C, H, W = features.shape
    PAD_LEN = H - 1
    DIM_SCAN_DIAG = W + PAD_LEN
    DIM_SEQ_DIAG = H

    # 1. Pad (left)
    x_pad = F.pad(features, (PAD_LEN, 0), 'constant', 0)
    w_expand = w.expand(-1, C, -1, -1)
    w_pad = F.pad(w_expand, (PAD_LEN, 0), 'constant', 0)

    # 2. Skew (shear)
    x_skewed = torch.empty_like(x_pad)
    w_skewed = torch.empty_like(w_pad)
    for h_i in range(H):
        x_skewed[:, :, h_i, :] = torch.roll(x_pad[:, :, h_i, :], shifts=-h_i, dims=-1)
        w_skewed[:, :, h_i, :] = torch.roll(w_pad[:, :, h_i, :], shifts=-h_i, dims=-1)

    # 3. Permute, Reshape, and Scan
    x_scannable = x_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    w_scannable = w_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    y_scannable = torch.empty_like(x_scannable)

    grid = (x_scannable.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(DIM_SEQ_DIAG)
    _domain_transform_ltr_kernel[grid](
        x_scannable, w_scannable, y_scannable,
        DIM_SEQ_DIAG, BLOCK_SIZE=BLOCK_SIZE,
    )

    # 4. Un-permute
    y_skewed = y_scannable.reshape(N, C, DIM_SCAN_DIAG, DIM_SEQ_DIAG).permute(0, 1, 3, 2)

    # 5. Un-skew
    y_pad = torch.empty_like(x_pad)
    for h_i in range(H):
        y_pad[:, :, h_i, :] = torch.roll(y_skewed[:, :, h_i, :], shifts=h_i, dims=-1)

    # 6. Un-pad
    return y_pad[:, :, :, PAD_LEN:].contiguous()


def _apply_diag_brtl(features: torch.Tensor, w: torch.Tensor):
    """Apply Bottom-Right to Top-Left diagonal scan."""
    N, C, H, W = features.shape
    PAD_LEN = H - 1
    DIM_SCAN_DIAG = W + PAD_LEN
    DIM_SEQ_DIAG = H

    # 1. Pad (left)
    x_pad = F.pad(features, (PAD_LEN, 0), 'constant', 0)
    w_expand = w.expand(-1, C, -1, -1)
    w_pad = F.pad(w_expand, (PAD_LEN, 0), 'constant', 0)

    # 2. Skew (shear)
    x_skewed = torch.empty_like(x_pad)
    w_skewed = torch.empty_like(w_pad)
    for h_i in range(H):
        x_skewed[:, :, h_i, :] = torch.roll(x_pad[:, :, h_i, :], shifts=-h_i, dims=-1)
        w_skewed[:, :, h_i, :] = torch.roll(w_pad[:, :, h_i, :], shifts=-h_i, dims=-1)

    # 3. Permute and Shift weights
    w_scannable_view = w_skewed.permute(0, 1, 3, 2)
    w_shifted = torch.zeros_like(w_scannable_view)
    w_shifted[..., :-1] = w_scannable_view[..., 1:] # Shift para RTL

    # 4. Reshape and Scan
    x_scannable = x_skewed.permute(0, 1, 3, 2).reshape(-1, DIM_SEQ_DIAG).contiguous()
    w_scannable = w_shifted.reshape(-1, DIM_SEQ_DIAG).contiguous()
    y_scannable = torch.empty_like(x_scannable)

    grid = (x_scannable.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(DIM_SEQ_DIAG)
    _domain_transform_rtl_kernel[grid](
        x_scannable, w_scannable, y_scannable,
        DIM_SEQ_DIAG, BLOCK_SIZE=BLOCK_SIZE,
    )

    # 5. Un-permute
    y_skewed = y_scannable.reshape(N, C, DIM_SCAN_DIAG, DIM_SEQ_DIAG).permute(0, 1, 3, 2)

    # 6. Un-skew
    y_pad = torch.empty_like(x_pad)
    for h_i in range(H):
        y_pad[:, :, h_i, :] = torch.roll(y_skewed[:, :, h_i, :], shifts=h_i, dims=-1)

    # 7. Un-pad
    return y_pad[:, :, :, PAD_LEN:].contiguous()

def domain_transform_triton(features: torch.Tensor, ref_edges: torch.Tensor,
                            sigma_s: float = 100.0, sigma_r: float = 1.0,
                            num_iterations: int = 1):
    """
    Fast Triton implementation of Domain Transform filtering.

    Args:
        features: Input features (N, C, H, W)
        ref_edges: Reference edge map (N, 1, H, W)
        sigma_s: Spatial standard deviation
        sigma_r: Range standard deviation
        num_iterations: Number of filtering iterations

    Returns:
        Filtered features (N, C, H, W)
    """
    assert features.is_cuda and ref_edges.is_cuda, "Tensors must be on CUDA"
    assert features.ndim == 4 and ref_edges.ndim == 4, "Inputs must be 4D tensors"

    # Pre-calculate weights
    ref_edges_clamped = torch.relu(ref_edges)
    d = 1.0 + (sigma_s / sigma_r) * ref_edges_clamped
    w = torch.exp(-torch.sqrt(torch.tensor(2.0, device=features.device)) * d / sigma_s)

    y = features.clone()

    for _ in range(num_iterations):
        # Four-pass filtering
        y = _apply_horizontal_ltr(y, w)  # Horizontal left-to-right
        y = _apply_horizontal_rtl(y, w)  # Horizontal right-to-left
        y = _apply_vertical_ttb(y, w)  # Vertical top-to-bottom
        y = _apply_vertical_btt(y, w)  # Vertical bottom-to-top

    return y


def domain_transform_triton_8pass(features: torch.Tensor, ref_edges: torch.Tensor,
                            sigma_s: float = 100.0, sigma_r: float = 1.0,
                            num_iterations: int = 1):
    """
    Fast Triton implementation of Domain Transform filtering.

    Args:
        features: Input features (N, C, H, W)
        ref_edges: Reference edge map (N, 1, H, W)
        sigma_s: Spatial standard deviation
        sigma_r: Range standard deviation
        num_iterations: Number of filtering iterations

    Returns:
        Filtered features (N, C, H, W)
    """
    assert features.is_cuda and ref_edges.is_cuda, "Tensors must be on CUDA"
    assert features.ndim == 4 and ref_edges.ndim == 4, "Inputs must be 4D tensors"

    # Pre-calculate weights
    ref_edges_clamped = torch.relu(ref_edges)
    d = 1.0 + (sigma_s / sigma_r) * ref_edges_clamped
    w = torch.exp(-torch.sqrt(torch.tensor(2.0, device=features.device)) * d / sigma_s)

    y = features.clone()

    for _ in range(num_iterations):
        # Cardinal
        y = _apply_horizontal_ltr(y, w)  # Horizontal left-to-right
        y = _apply_horizontal_rtl(y, w)  # Horizontal right-to-left
        y = _apply_vertical_ttb(y, w)  # Vertical top-to-bottom
        y = _apply_vertical_btt(y, w)  # Vertical bottom-to-top

        # Diagonal
        y = _apply_diag_tlbr(y, w)  # Top-Left to Bottom-Right
        y = _apply_diag_brtl(y, w)  # Bottom-Right to Top-Left
        y = _apply_diag_trbl(y, w)  # Top-Right to Bottom-Left
        y = _apply_diag_bltr(y, w)  # Bottom-Left to Top-Right

    return y

# ------------------------------------------------------------------
# For comparison with PyTorch reference
# ------------------------------------------------------------------

def domain_transform_pytorch(features: torch.Tensor, ref_edges: torch.Tensor,
                             sigma_s: float = 100.0, sigma_r: float = 1.0,
                             num_iterations: int = 1):
    """PyTorch reference implementation."""
    ref_edges_clamped = torch.relu(ref_edges)
    d = 1.0 + (sigma_s / sigma_r) * ref_edges_clamped
    w = torch.exp(-torch.sqrt(torch.tensor(2.0, device=features.device)) * d / sigma_s)

    y = features.clone()

    for _ in range(num_iterations):
        # Horizontal LTR
        for i in range(1, y.shape[3]):
            y[:, :, :, i] = (1 - w[:, :, :, i]) * y[:, :, :, i] + w[:, :, :, i] * y[:, :, :, i - 1]
        # Horizontal RTL
        for i in range(y.shape[3] - 2, -1, -1):
            y[:, :, :, i] = (1 - w[:, :, :, i + 1]) * y[:, :, :, i] + w[:, :, :, i + 1] * y[:, :, :, i + 1]
        # Vertical TTB
        for i in range(1, y.shape[2]):
            y[:, :, i, :] = (1 - w[:, :, i, :]) * y[:, :, i, :] + w[:, :, i, :] * y[:, :, i - 1, :]
        # Vertical BTT
        for i in range(y.shape[2] - 2, -1, -1):
            y[:, :, i, :] = (1 - w[:, :, i + 1, :]) * y[:, :, i, :] + w[:, :, i + 1, :] * y[:, :, i + 1, :]

    return y


# ------------------------------------------------------------------
# Example usage and benchmarking
# ------------------------------------------------------------------

def benchmark_domain_transform():
    """Comprehensive benchmark of the domain transform implementation."""
    print("🚀 Domain Transform Triton Implementation Benchmark")
    print("=" * 60)

    test_cases = [
        ("Small", 1, 8, 64, 64),
        ("Medium", 2, 16, 128, 128),
        ("Large", 4, 32, 256, 256),
        ("XLarge", 8, 64, 512, 512),
    ]

    for name, N, C, H, W in test_cases:
        print(f"\n📊 {name} Test: {(N, C, H, W)}")

        # Generate test data
        features = torch.randn(N, C, H, W, device='cuda', dtype=torch.float32)
        ref_edges = torch.rand(N, 1, H, W, device='cuda', dtype=torch.float32)

        # Correctness check (skip for very large tensors to save time)
        if H <= 256:
            output_pytorch = domain_transform_pytorch(features, ref_edges)
            output_triton = domain_transform_triton(features, ref_edges)

            is_correct = torch.allclose(output_pytorch, output_triton, atol=1e-4, rtol=1e-4)
            print(f"   ✅ Correctness: {'PASS' if is_correct else 'FAIL'}")

            if not is_correct:
                max_diff = (output_pytorch - output_triton).abs().max().item()
                print(f"      Max difference: {max_diff:.6f}")

        # Performance benchmark
        # Warmup
        for _ in range(5):
            _ = domain_transform_triton(features, ref_edges)
            if H <= 256:  # Only benchmark PyTorch for smaller sizes
                _ = domain_transform_pytorch(features, ref_edges)

        # Benchmark Triton
        ms_triton = triton.testing.do_bench(
            lambda: domain_transform_triton(features, ref_edges)
        )

        print(f"   ⚡ Triton: {ms_triton:.4f} ms")

        # Benchmark PyTorch (only for smaller sizes)
        if H <= 256:
            ms_pytorch = triton.testing.do_bench(
                lambda: domain_transform_pytorch(features, ref_edges)
            )
            speedup = ms_pytorch / ms_triton
            print(f"   🐌 PyTorch: {ms_pytorch:.4f} ms")
            print(f"   🚀 Speedup: {speedup:.2f}x")

        # Memory usage estimate
        memory_mb = (features.numel() + ref_edges.numel()) * 4 / 1024 / 1024
        print(f"   💾 Memory: {memory_mb:.1f} MB")


if __name__ == "__main__":
    benchmark_domain_transform()