"""Export the deployment graph to ONNX and stamp the calibration onto it.

``CommandSense v2.2`` is a calibration-only release: it reuses the v2.1 weights and
adds no layer, so the graph it exports is exactly the v2.1 graph. Raw logits stay
raw inside ONNX - ``T`` and ``tau`` are applied by ``src/inference.py`` at runtime,
which is why changing the threshold never requires a re-export. What v2.2 does add
to the file is *documentation*: the fitted temperature, the chosen threshold and
the calibration quality are written into ``metadata_props`` under the
``commandsense.calibration.`` namespace.

Resolution order for the checkpoint being exported:

1. ``checkpoints/model_v2.2/<MODEL_NAME>_v2.2.pth``  (only if a future release
   really trains one)
2. ``checkpoints/model_v2.1/<MODEL_NAME>_v2.1.pth``  (the weights v2.2 reuses)
3. ``checkpoints/<MODEL_NAME>_<version>.pth``        (legacy layout)

Run with:  python export_onnx.py
"""

import os
import sys

import onnx
import torch

from configs.config import (
    BASE_MODEL_VERSION,
    CHECKPOINT_DIR,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    checkpoint_dir,
)
from src import calibration
from src.models import CommandSense

# torch.onnx prints progress messages containing non-ASCII glyphs. When stdout is
# a pipe, Python falls back on the legacy console code page (cp1252 on Windows)
# and those prints would raise UnicodeEncodeError in the middle of the export, so
# the stream is switched to UTF-8 first.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic stream types
            pass

OPSET_VERSION = 17


def resolve_checkpoint(version=MODEL_VERSION):
    """Return ``(path, version)`` of the weights to export, canonical layout first."""
    candidates = [
        (checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth", version),
        (
            checkpoint_dir(BASE_MODEL_VERSION) / f"{MODEL_NAME}_{BASE_MODEL_VERSION}.pth",
            BASE_MODEL_VERSION,
        ),
        (CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth", version),
    ]
    for candidate, resolved_version in candidates:
        if os.path.exists(candidate):
            return str(candidate), resolved_version
    return None, None


def resolve_calibration(version=MODEL_VERSION):
    """Return the calibration metadata of the deployed release, or ``{}``."""
    metadata = calibration.load_calibration(version=version)
    if not metadata and version != BASE_MODEL_VERSION:
        metadata = calibration.load_calibration(version=BASE_MODEL_VERSION)
    return metadata

def export_to_onnx():
    checkpoint_path, weights_version = resolve_checkpoint()
    if checkpoint_path is None:
        print(
            f"[!] Error: no checkpoint found at "
            f"'{checkpoint_dir(MODEL_VERSION)}/{MODEL_NAME}_{MODEL_VERSION}.pth' nor at "
            f"'{checkpoint_dir(BASE_MODEL_VERSION)}/{MODEL_NAME}_{BASE_MODEL_VERSION}.pth'."
        )
        return

    if weights_version != MODEL_VERSION:
        print(
            f"[i] {MODEL_VERSION} is a calibration-only release, so the {weights_version} "
            "graph is re-exported with the fitted T/tau attached as metadata."
        )
    print(f"[*] Exporting weights <- {checkpoint_path}")

    # The graph is written next to the checkpoint it comes from, which is also
    # where src/inference.py looks for it when it falls back on
    # BASE_MODEL_VERSION.
    onnx_path = os.path.join(
        os.path.dirname(checkpoint_path), f"{MODEL_NAME}_{weights_version}.onnx"
    )

    device = torch.device("cpu")
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 63, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    embed_calibration_metadata(onnx_path)
    verify_onnx_model(onnx_path)

    external_data_path = f"{onnx_path}.data"
    if os.path.exists(external_data_path):
        os.remove(external_data_path)

    print(f"[+] ONNX model successfully exported -> {onnx_path}")
    print(
        "[*] The graph stays raw-logit: T and tau are applied by src/inference.py, "
        "so they can change without a re-export."
    )

def embed_calibration_metadata(onnx_path: str, version: str = MODEL_VERSION):
    """Attach the fitted calibration to the graph's ``metadata_props``."""
    metadata = resolve_calibration(version)
    if not metadata:
        print(
            "[!] No calibration file found, so the export carries no T/tau metadata. "
            "Run evaluate.py first; the runtime then falls back on the defaults."
        )
        return

    props = calibration.metadata_to_onnx_props(metadata)
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    # Idempotent: refresh only the calibration namespace and keep foreign entries
    # such as the producer information torch adds during export.
    prefix = calibration.ONNX_METADATA_PREFIX
    kept = [prop for prop in onnx_model.metadata_props if not prop.key.startswith(prefix)]
    del onnx_model.metadata_props[:]
    for key, value in [(prop.key, prop.value) for prop in kept] + list(props.items()):
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, str(value)

    onnx.save(onnx_model, onnx_path)
    print(
        f"[+] Embedded calibration metadata (T={metadata.get('temperature')}, "
        f"tau={metadata.get('confidence_threshold')}) under '{prefix}*'"
    )


def verify_onnx_model(onnx_path: str):
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)

    external_initializers = [
        initializer.name
        for initializer in onnx_model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ]
    assert not external_initializers, (
        f"[!] Export Error: Model contains external tensor data: {external_initializers}"
    )
    
    input_tensor = onnx_model.graph.input[0]
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    
    # Check input channels shape: [batch_size, 3, 64, 63]
    num_channels = shape[1]
    assert num_channels == 3, f"[!] Export Error: Expected 3 input channels, got {num_channels}"
    
    print(f"[+] ONNX Strict Shape Verification Passed: declared_shape[1] == {num_channels}")

    # v2.2 keeps the softmax in Python, so the graph must still end in raw logits.
    assert len(onnx_model.graph.output) == 1, "[!] Export Error: expected a single output"
    output_shape = [
        dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_shape[-1] == NUM_CLASSES, (
        f"[!] Export Error: expected {NUM_CLASSES} logits, got {output_shape[-1]}"
    )
    print(f"[+] ONNX output verified: {NUM_CLASSES} raw logits per clip")

    calibration_keys = [
        prop.key
        for prop in onnx_model.metadata_props
        if prop.key.startswith(calibration.ONNX_METADATA_PREFIX)
    ]
    if calibration_keys:
        print(f"[+] Calibration metadata carried by the graph: {len(calibration_keys)} entries")
    else:
        print("[i] Graph carries no calibration metadata: T/tau stay at their defaults")

if __name__ == "__main__":
    export_to_onnx()