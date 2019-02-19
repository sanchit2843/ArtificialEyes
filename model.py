import torchvision.models as models
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        f_extractor = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*f_extractor)
    def forward(self, x):
        return self.resnet(x)
