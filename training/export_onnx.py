#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for deployment.
"""

import torch
import argparse
from pathlib import Path
from train import create_model

def export_to_onnx(model_path, output_path, opset_version=18):
    """Export PyTorch model to ONNX format."""

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    classes = checkpoint.get('classes', ['library', 'not_library'])

    print(f"Loaded model from {model_path}")
    print(f"Classes: {classes}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    # Create model and load weights
    model = create_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export with embedded weights (no external data file)
    with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

    # Load and re-save with all data embedded (no external data file)
    import onnx
    # Load with external data first
    model_proto = onnx.load(str(output_path), load_external_data=True)
    # Save with all data embedded in the protobuf
    onnx.save(
        model_proto,
        str(output_path),
        save_as_external_data=False
    )

    # Remove any .data file that was created
    data_file = Path(str(output_path) + '.data')
    if data_file.exists():
        data_file.unlink()
        print(f"Removed external data file (embedded in main ONNX file)")

    print(f"\nONNX model exported to {output_path}")

    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(output_path))

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})

        print(f"\nONNX Runtime test successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {result[0].shape}")

        # Save class labels
        labels_path = output_path.parent / f"{output_path.stem}_labels.txt"
        with open(labels_path, 'w') as f:
            for label in classes:
                f.write(f"{label}\n")
        print(f"Class labels saved to {labels_path}")

    except ImportError:
        print("\nNote: Install onnxruntime to test the exported model")
        print("  pip install onnxruntime")

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model', type=str,
                       default='../models/library_detector.pth',
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str,
                       default='../models/library_detector.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--opset', type=int, default=18,
                       help='ONNX opset version')
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.opset)

if __name__ == '__main__':
    main()
