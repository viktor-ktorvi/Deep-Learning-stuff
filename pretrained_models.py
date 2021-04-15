import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
import numpy as np
from efficientnet_pytorch import EfficientNet
import time

url = "https://images.theconversation.com/files/43861/original/6gqj734n-1394723838.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=926&fit=clip"
img = Image.open(requests.get(url, stream=True).raw)

# https://towardsdatascience.com/efficient-inference-in-deep-learning-where-is-the-problem-4ad59434fe36
# Blog - runtime, accuracy, flops
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model = models.mobilenet_v2(pretrained=True)
model = models.resnet34(pretrained=True)

# C:\Users\HP\.cache\torch\hub\checkpoints

# TODO zasto je sporije na GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = model.to(device)
model.eval()

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
batch_t = batch_t.to(device)

# Inference
infer_time = []
num = 10
for i in range(num):
    tic = time.time()
    out = model(batch_t)
    toc = time.time()
    infer_time.append(toc - tic)

print("Mean inference time = %3.5f" % (sum(infer_time) / num))

LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(LABELS_URL)
labels = {int(key): value for key, value in enumerate(response.json())}

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
percentage = percentage.detach().cpu().numpy()
indexes = np.argsort(percentage)
indexes = indexes[::-1]

dash = '-' * 50
print(dash)
print('{:<30s}{:>12s} %'.format("Class", "Probability"))
print(dash)
for i in range(5):
    print('{:<30s}{:>12.3f} %'.format(labels[indexes[i]], percentage[indexes[i]].item()))
