import os
import torch
import onnx
from src.models import CommandSense
from configs.config import MODEL_NAME, MODEL_VERSION, NUM_CLASSES

def export_to_onnx():
    checkpoint_path = f"checkpoints/model_{MODEL_VERSION}/{MODEL_NAME}_{MODEL_VERSION}.pth"
    onnx_path = f"checkpoints/model_{MODEL_VERSION}/{MODEL_NAME}_{MODEL_VERSION}.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint {checkpoint_path} does not exist.")
        return

    device = torch.device("cpu")
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Input tensor shape for v2.0.0: [Batch=1, Channels=3, Mel_Bins=64, Time_Frames=63]
    dummy_input = torch.randn(1, 3, 64, 63, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"[+] ONNX model successfully exported -> {onnx_path}")

    # v2.0.0 Guard-rail: Strict shape validation on exported ONNX model
    verify_onnx_model(onnx_path)

def verify_onnx_model(onnx_path: str):
    """Verify ONNX model structure and input tensor channels (v2.0.0 guard-rail)."""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    input_tensor = onnx_model.graph.input[0]
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    
    # Check input channels shape: [batch_size, 3, 64, 63]
    num_channels = shape[1]
    assert num_channels == 3, f"[!] Export Error: Expected 3 input channels for v2.0.0, got {num_channels}"
    
    print(f"[+] ONNX Strict Shape Verification Passed: declared_shape[1] == {num_channels}")

if __name__ == "__main__":
    export_to_onnx()