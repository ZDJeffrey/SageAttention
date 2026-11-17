import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def sage3(q, k, v, *, is_causal=False, per_block_mean=True):
    from sageattn3 import sageattn3_blackwell

    return sageattn3_blackwell(
        q, k, v, is_causal=is_causal, per_block_mean=per_block_mean
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "q_len,k_len,causal,per_block_mean",
    [
        (256, 256, False, True),
        (257, 193, False, True),
        (256, 256, True, True),
        (129, 257, False, False),
    ],
)
def test_changed_input_graph_replay_on_non_default_stream(
    dtype, q_len, k_len, causal, per_block_mean
):
    torch.manual_seed(12)
    q = torch.randn((1, 2, q_len, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 2, k_len, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sage3(q, k.clone(), v, is_causal=causal, per_block_mean=per_block_mean)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = sage3(q, k, v, is_causal=causal, per_block_mean=per_block_mean)
    previous = None
    for _ in range(2):
        inputs = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))
        expected = sage3(
            inputs[0],
            inputs[1].clone(),
            inputs[2],
            is_causal=causal,
            per_block_mean=per_block_mean,
        )
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for target, value in zip((q, k, v), inputs, strict=True):
                target.copy_(value)
            graph.replay()
        stream.synchronize()
        assert torch.isfinite(captured).all()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        assert torch.equal(captured, expected)
        torch.testing.assert_close(k, inputs[1], rtol=0, atol=0)
        first_replay = captured.clone()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        assert torch.equal(captured, expected)
        torch.testing.assert_close(captured, first_replay, rtol=0, atol=0)
        assert torch.equal(captured, first_replay)
        torch.testing.assert_close(k, inputs[1], rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(captured, previous)
        previous = captured.clone()


def test_sage3_does_not_modify_key():
    q = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    original = k.clone()
    sage3(q, k, v)
    torch.testing.assert_close(k, original, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
def test_input_device_differs_from_current_device():
    with torch.cuda.device(1):
        q = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        expected = sage3(q, k.clone(), v)
        torch.cuda.synchronize()
    with torch.cuda.device(0):
        actual = sage3(q, k.clone(), v)
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "q_len,k_len", [(7540, 7540), (31668, 31668), (7540, 512), (31668, 512)]
)
def test_wan_attention_shape_changed_input_replay(q_len, k_len):
    torch.manual_seed(23)
    q = torch.randn((1, 40, q_len, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 40, k_len, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for _ in range(2):
        sage3(q, k.clone(), v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sage3(q, k, v)
    next_q, next_k, next_v = (
        torch.randn_like(q),
        torch.randn_like(k),
        torch.randn_like(v),
    )
    expected = sage3(next_q, next_k.clone(), next_v)
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(k, next_k, rtol=0, atol=0)


def test_noncontiguous_hnd_changed_input_replay():
    q = torch.randn((1, 193, 2, 128), device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    k = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    v = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    assert not q.is_contiguous() and not k.is_contiguous()
    for _ in range(3):
        sage3(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = sage3(q, k, v)
    new_q, new_k, new_v = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
    expected = sage3(new_q, new_k, new_v)
    q.copy_(new_q)
    k.copy_(new_k)
    v.copy_(new_v)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(result, expected)
    torch.testing.assert_close(k, new_k, rtol=0, atol=0)
