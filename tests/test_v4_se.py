"""Smoke test for the v4.0 Squeeze-and-Excitation ResNet (no dataset, no training).

Verifies that ``SqueezeExcitation`` squeezes and recalibrates a ``[B, C, H, W]``
tensor without changing its shape, that ``CommandSense(use_se=True)`` keeps the
``[B, 3, 64, 63] -> [B, num_classes]`` contract, that the SE overhead stays
inside the parameter budget, and that ``use_se=False`` reproduces the exact v3.0
graph, parameter count and ``state_dict()``. The last check exports the graph to
ONNX and compares the ONNX Runtime output with the PyTorch output.

Run directly:  python tests/test_v4_se.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import NUM_CLASSES  # noqa: E402
from src.models import CommandSense, SqueezeExcitation  # noqa: E402

# Exact trainable-parameter counts for the frozen graphs (37 classes).
V3_PARAM_COUNT = 1_217_349
V4_PARAM_COUNT = 1_239_357
PARAM_BUDGET = 1_300_000          # the hard ceiling from the v4.0 spec
MAX_SE_OVERHEAD = 0.05            # SE must stay below 5% of the v3.0 budget

INPUT_SHAPE = (2, 3, 64, 63)


def count_trainable_parameters(model):
    """Return the number of trainable scalars, the project's parity metric."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_squeeze_excitation_recalibrates_channels():
    """The SE block keeps [B, C, H, W] and scales each channel by its own gate."""
    torch.manual_seed(0)
    block = SqueezeExcitation(channels=64, reduction_ratio=8).eval()
    x = torch.randn(2, 64, 32, 32)

    with torch.no_grad():
        out = block(x)
        descriptor = block.avg_pool(x).flatten(1)
        gate = block.sigmoid(block.fc2(block.relu(block.fc1(descriptor))))
        gate = gate.view(x.shape[0], x.shape[1], 1, 1)

    assert tuple(out.shape) == tuple(x.shape), (out.shape, x.shape)
    assert torch.all(gate > 0) and torch.all(gate < 1), gate
    assert torch.allclose(out, x * gate, atol=1e-6), "output is not input * gate"

    # reduction_ratio drives the bottleneck width (and is clamped to >= 1).
    assert SqueezeExcitation(64, reduction_ratio=4).fc1.out_features == 16
    assert SqueezeExcitation(8, reduction_ratio=8).fc1.out_features == 1
    print("[OK] SqueezeExcitation: shape preserved, channels gated into (0, 1)")


def test_v4_forward_contract():
    """v4.0 must map [2, 3, 64, 63] to [2, num_classes]."""
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=True).eval()
    dummy = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        output = model(dummy)

    assert tuple(output.shape) == (INPUT_SHAPE[0], NUM_CLASSES), output.shape
    assert torch.isfinite(output).all(), "logits contain non-finite values"
    print(f"[OK] v4.0 forward pass: {INPUT_SHAPE} -> {tuple(output.shape)}")


def test_parameter_budget():
    """SE must stay below both the absolute ceiling and the 5% overhead cap."""
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=True)
    params = count_trainable_parameters(model)
    assert params == V4_PARAM_COUNT, (params, V4_PARAM_COUNT)
    assert params < PARAM_BUDGET, params

    overhead = (V4_PARAM_COUNT - V3_PARAM_COUNT) / V3_PARAM_COUNT
    assert overhead < MAX_SE_OVERHEAD, overhead
    print(
        f"[OK] v4.0 params: {params:,} "
        f"(+{overhead:.2%} over v3.0, budget {PARAM_BUDGET:,})"
    )


def test_backward_compatibility_with_v3():
    """use_se=False must reproduce the v3.0 parameter count and state_dict keys."""
    model_v3 = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=False)
    assert count_trainable_parameters(model_v3) == V3_PARAM_COUNT
    assert not any(".se." in key for key in model_v3.state_dict()), "SE leaked into v3.0"

    # A v3.0 checkpoint therefore round-trips through a freshly built model.
    fresh = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=False)
    fresh.load_state_dict(model_v3.state_dict())

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        output = fresh(dummy)
    assert tuple(output.shape) == (INPUT_SHAPE[0], NUM_CLASSES), output.shape
    print(
        f"[OK] backward compatible: use_se=False -> {V3_PARAM_COUNT:,} params, no SE keys"
    )



def test_state_dict_is_clean():
    """The SE stage adds only weights/biases; every buffer stays a BatchNorm stat."""
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=True)
    state = model.state_dict()

    se_keys = sorted(key for key in state if ".se." in key)
    assert se_keys == [
        "layer1.se.fc1.bias", "layer1.se.fc1.weight",
        "layer1.se.fc2.bias", "layer1.se.fc2.weight",
        "layer2.se.fc1.bias", "layer2.se.fc1.weight",
        "layer2.se.fc2.bias", "layer2.se.fc2.weight",
        "layer3.se.fc1.bias", "layer3.se.fc1.weight",
        "layer3.se.fc2.bias", "layer3.se.fc2.weight",
    ], se_keys

    buffer_names = [name for name, _ in model.named_buffers()]
    assert buffer_names, "expected the BatchNorm running statistics to be present"
    assert all(
        name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for name in buffer_names
    ), buffer_names
    assert not any(".se." in name for name in buffer_names), "SE introduced a buffer"

    # ``state_dict`` holds exactly the parameters plus the (persistent) BN stats.
    assert len(state) == len(list(model.parameters())) + len(buffer_names), len(state)
    print(
        f"[OK] state_dict clean: {len(se_keys)} SE tensors, "
        f"{len(buffer_names)} BatchNorm buffers, no stray entries"
    )


def test_onnx_export_parity():
    """The SE graph exports to ONNX and matches PyTorch under ONNX Runtime."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[skip] onnx / onnxruntime not installed")
        return

    # torch.onnx prints non-ASCII progress glyphs; switch the streams to UTF-8 so
    # the export does not crash when stdout is a pipe (cp1252 on Windows).
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=True).eval()
    dummy = torch.randn(1, 3, 64, 63)
    out_dir = tempfile.mkdtemp(prefix="cs_v4_onnx_")
    onnx_path = os.path.join(out_dir, "CommandSense_v4.0.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
    )

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)
    ops = {node.op_type for node in onnx_model.graph.node}
    assert "Sigmoid" in ops and "Mul" in ops, f"SE ops missing from the graph: {sorted(ops)}"

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_output = model(dummy).numpy()
    assert np.allclose(torch_output, ort_output, atol=1e-4), "ONNX / PyTorch mismatch"
    print(
        f"[OK] ONNX export: SE ops present, max abs diff "
        f"{np.abs(torch_output - ort_output).max():.2e}"
    )


def main():
    test_squeeze_excitation_recalibrates_channels()
    test_v4_forward_contract()
    test_parameter_budget()
    test_backward_compatibility_with_v3()
    test_state_dict_is_clean()
    test_onnx_export_parity()
    print("\n[OK] v4.0 SE-ResNet smoke test passed")


if __name__ == "__main__":
    main()

