import torch
from dataset import ImageCaptionDataset
from decoder import Decoder
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data = ImageCaptionDataset(data_transforms, 'data/coco/imgs', 'data/coco/dataset.json')
a = torch.randn(64, 196, 512)
b = torch.ones(64, 13).long()

decoder = Decoder()
preds, alphas = decoder(a, b)

print(data.captions)
