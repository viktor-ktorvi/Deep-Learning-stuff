import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests


url = "https://images.theconversation.com/files/43861/original/6gqj734n-1394723838.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=926&fit=clip"
img = Image.open(requests.get(url, stream=True).raw)

# mobilenet_v2 = models.mobilenet_v2(pretrained=True)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

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

model.eval()
out = model(batch_t)
print(out.shape)

LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(LABELS_URL)
labels = {int(key): value for key, value in enumerate(response.json())}

_, index = torch.max(out, 1)
index = index.numpy()
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())

