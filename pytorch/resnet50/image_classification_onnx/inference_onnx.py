import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import onnxruntime
import time
import argparse
import requests
import os

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="Precision options for ResNet50 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed", "int8"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed, or int8"
)
parser.add_argument(
    "--ep",
    type=str,
    choices=["rocm", "migx", "cuda", "openvino", "cpu"],
    default="cpu",
    help="Set the execution provider for inference"
)
args = parser.parse_args()

# 2. Map Execution Provider
ep_map = {
    "rocm": "ROCMExecutionProvider",
    "migx": "MIGraphXExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider"
}
execution_provider = ep_map.get(args.ep)

# 3. Set Model Filename
model_name = {
    "fp32": "resnet50_model.onnx",
    "fp16": "resnet50_model_fp16.onnx",
    "mixed": "resnet50_model_mixed.onnx",
    "int8": "resnet50-v1-12-int8.onnx"
}.get(args.precision)

# 4. Handle INT8 Download if missing
if args.precision == "int8" and not os.path.exists(model_name):
    print(f"Downloading {model_name}...")
    try:
        url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx?download="
        response = requests.get(url)
        response.raise_for_status()
        with open(model_name, "wb") as f:
            f.write(response.content)
        print("Download successful.")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)

# 5. Load Model Session
if not os.path.exists(model_name):
    print(f"Error: Model file {model_name} not found. Please ensure the file exists.")
    exit(1)

session = onnxruntime.InferenceSession(model_name, providers=[execution_provider])

# FIX: Dynamically identify the input name (avoids the "x" vs "input.1" error)
input_node = session.get_inputs()[0]
input_label = input_node.name
print(f"Model loaded. Detected input name: '{input_label}'")

# 6. Image Pre-processing
filename = 'cat.jpg'  # Ensure this file exists in your directory
if not os.path.exists(filename):
    print(f"Error: {filename} not found. Please provide an image.")
    exit(1)

input_image = Image.open(filename).convert('RGB')
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# 7. Match Precision for Input Data
# Convert to Numpy and ensure the data type matches the model's requirements
input_data = input_batch.numpy()

if args.precision == "fp16":
    input_data = input_data.astype(np.float16)
elif args.precision == "int8":
    # Note: Most ONNX INT8 models still take FP32 input and quantize internally
    input_data = input_data.astype(np.float32)

# 8. Warm up run
print("Running warm-up...")
_ = session.run([], {input_label: input_data})

# 9. Measured Inference
print("Running timed inference...")
iterations = 10
start_time = time.time()

for _ in range(iterations):
    ort_outputs = session.run([], {input_label: input_data})[0]

end_time = time.time()
avg_latency = ((end_time - start_time) / iterations) * 1000

# 10. Process Results
# Read categories (ensure imagenet_classes.txt is present)
try:
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
except FileNotFoundError:
    categories = [f"Class {i}" for i in range(1000)]

output = ort_outputs.flatten()
# Softmax
output = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
top5_catid = np.argsort(-output)[:5]

print("\n--- Results ---")
for i in top5_catid:
    print(f"{categories[i]}: {output[i]:.4f}")

print(f"\nAverage ONNX Inference Time: {avg_latency:.2f} ms")