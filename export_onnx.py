import os
import torch
from src.models import CommandSense
from configs.config import MODEL_VERSION, NUM_CLASSES

def export_to_onnx():
    checkpoint_path = f"checkpoints/CommandSense_{MODEL_VERSION}.pth"
    onnx_path = F"checkpoints/CommandSense_{MODEL_VERSION}.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint {checkpoint_path} does not exist.")
        return

    device = torch.device("cpu")
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Input tensor shape: [Batch=1, Channels=1, Mel_Bins=64, Time_Frames=63]
    dummy_input = torch.randn(1, 1, 64, 63, device=device)

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

if __name__ == "__main__":
    export_to_onnx()