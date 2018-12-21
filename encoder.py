import torch.nn as nn
from torchvision.models import densenet201, resnet152, vgg19


class Encoder(nn.Module):
    def __init__(self, network='vgg19'):
        super(Encoder, self).__init__()
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
        elif network == 'densenet201':
            self.net = densenet201(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
        else:
            self.net = vgg19(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
