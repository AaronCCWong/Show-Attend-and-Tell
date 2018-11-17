import torch.nn as nn
from torchvision.models import vgg19


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = vgg19(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:-1])

    def forward(self, x):
        x = self.vgg(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
