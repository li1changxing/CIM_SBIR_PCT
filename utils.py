import torch
import torch.nn as nn
from torch_geometric.data import Data
class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MOCO(nn.Module):
    """
    Based on the MoCo Loss.
    GitHub: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    Paper: https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=384, K=1024, m=0.999, T=0.07):
        super(MOCO, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        if self.K % batch_size != 0:
            return

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, query, key):
        with torch.no_grad():  # no gradient to keys
            key = nn.functional.normalize(key, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [query, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(key)

        return logits, labels


if __name__ == '__main__':
    from torch_geometric.nn import GCNConv

    model = GCNConv(384, 384)

    data1 = torch.randn(256,384)
    data2 = torch.randn(128,384)
    sim = torch.mm(data1,data2.T)

    data = Data(x=data1, edge_index=sim)
    out = model(data.x, data.edge_index)
    print(out.shape)

