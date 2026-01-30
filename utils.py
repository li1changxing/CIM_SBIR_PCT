import torch
import torch.nn as nn
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

def sem_loss(query, center_txt,tau=0.07):
        logits = torch.einsum('nc,ck->nk', [query, center_txt.detach().T])
        logits /= tau
        return logits

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

        query = nn.functional.normalize(query,dim=1)

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

def select_knn(skt_fea, img_fea, all_skt, k = 5):
    skt_fea_norm = nn.functional.normalize(skt_fea, p=2, dim=-1).detach()
    img_fea_norm = nn.functional.normalize(img_fea, p=2, dim=-1).detach()

    similarity_matrix = torch.mm(skt_fea_norm, img_fea_norm.T)
    similarity_scores, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    similarity_scores = torch.softmax(similarity_scores, dim=-1).unsqueeze(-1)
    img_select = all_skt[top_k_indices]
    img_select = img_select * similarity_scores
    img_select = torch.sum(img_select, dim=1)

    return img_select.float()


def recover_sketch(image_features, sketch_features, k=5):
    # image_features: Tensor, shape (n_images, feature_dim)
    # sketch_features: Tensor, shape (n_sketches, feature_dim)

    # Compute cosine similarity
    similarity = torch.mm(image_features, sketch_features.T)
    print(similarity.shape)

    # Select top-k most similar sketches for each image
    topk_values, topk_indices = torch.topk(similarity, k, dim=1)

    # Apply softmax to the similarity scores
    weights = nn.functional.softmax(topk_values, dim=1).unsqueeze(-1)

    # Weighted sum to recover sketches
    # recovered_sketches = torch.matmul(weights, sketch_features[topk_indices])
    recovered_sketches = weights * sketch_features[topk_indices]

    return recovered_sketches

if __name__ == '__main__':
    img_fea = torch.randn((4,768))
    skt_fea = torch.randn((16,768))
    print(recover_sketch(img_fea, skt_fea).shape)