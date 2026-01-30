import torch.nn as nn
import torch.nn.functional as F
import torch
class Prototype(nn.Module):
    def __init__(self, skt, img, momentum=0.9, dim=512):
        super(Prototype, self).__init__()
        self.momentum = momentum
        self.register_buffer("center_skt", skt)
        self.register_buffer("center_img", img)
        self.fea_dim = dim
        self.cn = nn.CrossEntropyLoss()

    def class_loss(self, query, key, tau=0.07):
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
        logits = torch.einsum('nc,ck->nk', [query, key.detach().T])
        logits /= tau
        return logits

    def sce_loss(self, x, y, alpha=3):
        # x = F.normalize(x, p=2, dim=-1)
        # y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def forward(self, class_in_batch):
        loss = self.align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])
        return loss

    def forward_img(self, x, l):
        class_in_batch = self.update_center(x, l,"img")
        return self.sce_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch]), class_in_batch
        # return self.sce_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])
        # return self.align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])

    def forward_skt(self, x, l):
        class_in_batch = self.update_center(x, l, "skt")
        return self.sce_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch]), class_in_batch
        # return self.align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])
    def update_center(self, x, l, modality):
        self.center_img = self.center_img.detach()
        self.center_skt = self.center_skt.detach()
        all_l=l
        classes_in_batch, sam2cls_idx, cl_sam_counts = torch.unique(all_l, return_counts=True, sorted=True, return_inverse=True)
        center_tmp = torch.zeros(len(classes_in_batch), self.fea_dim).cuda()
        for i, idx in enumerate(sam2cls_idx):
            center_tmp[idx] += x[i]
        center_tmp = center_tmp / cl_sam_counts.unsqueeze(1)

        if modality == 'img':
            self.center_img[classes_in_batch] = self.center_img[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_img[classes_in_batch] /= self.center_img[classes_in_batch].norm(p=2, dim=1, keepdim=True)
        else:
            self.center_skt[classes_in_batch] = self.center_skt[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_skt[classes_in_batch] /= self.center_skt[classes_in_batch].norm(p=2, dim=1, keepdim=True)

        return classes_in_batch