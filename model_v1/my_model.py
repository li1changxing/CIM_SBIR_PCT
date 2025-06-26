import torch
import torch.nn as nn
from timm.layers import Mlp

from model_v1.GAT import GAT
from model_v1.vit import VisionTransformer, vit_base_patch16_224

class mymodel(nn.Module):
    def __init__(self,num_classes=25, pre_weigth="../vit.pth",dim=512):
        super().__init__()
        self.backbone:VisionTransformer = vit_base_patch16_224()
        msg = self.backbone.load_state_dict(torch.load(pre_weigth), strict=False)
        for name, p in self.backbone.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.dim = dim
        self.p_head = GAT(out_dim=dim//4)
        self.num_classes = num_classes

        # self.skt2img = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(512, 768),
        # )

        # self.skt2img = nn.Sequential(nn.Linear(768, 768))
        # self.img2skt = nn.Sequential(nn.Linear(768, 768))
        #
    def reset(self, n):
        self.classifier = nn.Linear(768, n).cuda()
        self.p_head = GAT(out_dim=self.dim//4).cuda()

        # self.img2skt = nn.Sequential(nn.Linear(512, 512),)
        # self.skt2img = nn.Sequential(nn.Linear(512, 512),)

    def add_classes(self, num_new_classes):
        # Add new output neurons for new classes
        print(self.num_classes, num_new_classes)
        old_output_layer = self.classifier
        new_output_layer = nn.Linear(768, self.num_classes + num_new_classes)

        # Copy existing weights
        with torch.no_grad():
            new_output_layer.weight[:self.num_classes] = old_output_layer.weight
            new_output_layer.bias[:self.num_classes] = old_output_layer.bias

        # Replace the old classifier with the new one
        self.classifier = new_output_layer
        self.num_classes += num_new_classes

    def forward(self,x):
        fea = self.backbone(x)
        return fea
