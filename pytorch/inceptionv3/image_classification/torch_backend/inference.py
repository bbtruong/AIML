import torch
from torchvision import transforms
import time
import argparse
from PIL import Image

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for ResNet50 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed"
)
args = parser.parse_args()

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Load the pre-trained InceptionV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
model.eval()

# Pre-process image for inference - sample execution (requires torchvision)
filename = 'dog.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

# If fp16, adjust model and input tensor accordingly
if args.precision == "fp16":
    model = model.half()
    input_tensor = input_tensor.half()

# Adjust tensor size for model (all precisions)
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to the GPU
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
else:
   print("GPU unavailalbe. Defaulting to CPU.")

# Warm-up run to load model and data onto the GPU
with torch.no_grad():
    if args.precision == "mixed":
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(input_batch)
    else:
        output = model(input_batch)

# Perform inference and record time
latency = []
torch.cuda.synchronize()
start = time.time()

if args.precision == "mixed":
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(input_batch)
else:
    with torch.no_grad():
        output = model(input_batch)

torch.cuda.synchronize()
end = time.time()
latency.append(end - start)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))