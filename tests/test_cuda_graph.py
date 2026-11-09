"""CUDA stream handling follows https://github.com/thu-ml/SageAttention/pull/380."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    ("path", "layout", "dtype", "return_lse"),
    [
        ("auto", "NHD", torch.bfloat16, False),
        ("auto", "HND", torch.float16, True),
        ("fp16_smooth_v", "NHD", torch.bfloat16, True),
    ],
)
def test_non_default_stream_and_graph_replay(path, layout, dtype, return_lse):
    from sageattention import sageattn, sageattn_qk_int8_pv_fp16_cuda

    def attention(q, k, v):
        kwargs = {"tensor_layout": layout, "return_lse": return_lse}
        if path == "auto":
            return sageattn(q, k, v, **kwargs)
        return sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, pv_accum_dtype="fp16", smooth_v=True, **kwargs
        )

    def as_tuple(result):
        return result if isinstance(result, tuple) else (result,)

    torch.manual_seed(19)
    shape = (1, 256, 8, 128) if layout == "NHD" else (1, 8, 256, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            attention(q, k, v)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = as_tuple(attention(q, k, v))

    previous = None
    for _ in range(2):
        inputs = [torch.randn_like(q) for _ in range(3)]
        reference = as_tuple(attention(*inputs))
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for target, value in zip((q, k, v), inputs, strict=True):
                target.copy_(value)
            eager_stream = as_tuple(attention(q, k, v))
        stream.synchronize()
        for actual, expected in zip(eager_stream, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        for actual, expected in zip(captured, reference, strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(captured[0], previous)
        previous = captured[0].clone()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
def test_fused_uses_input_device_and_restores_caller():
    from sageattention import _fused

    with torch.cuda.device(1):
        data = torch.randn((1, 256, 8, 128), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(data, dtype=torch.int8)
        scale = torch.empty((1, 8, 2), device="cuda", dtype=torch.float32)
        expected = torch.empty_like(output)
        expected_scale = torch.empty_like(scale)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _fused.quant_per_block_int8_cuda(data, expected, expected_scale, 128, 0)
        stream.synchronize()

    with torch.cuda.stream(stream):
        torch.cuda.set_device(0)
        _fused.quant_per_block_int8_cuda(data, output, scale, 128, 0)
        assert torch.cuda.current_device() == 0
    stream.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
