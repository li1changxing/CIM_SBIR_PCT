import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from CLIP_AT.clip_1 import clip

def freeze_model(m):
    for name, param in m.named_parameters():
        if 'prompt' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder, _ = clip.load('ViT-B/16')
        self.encoder = self.encoder.float()
        self.encoder.apply(freeze_model)

    def forward(self, data):
        feat = self.encoder.encode_image(data, prompt_flag=True)
        return feat

if __name__ == '__main__':
    model = Model().cuda()
    data = torch.randn(4,3,224,224).cuda()

    out = model.encode_img(data)
    print(out.shape)

