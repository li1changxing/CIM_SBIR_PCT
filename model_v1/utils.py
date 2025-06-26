import torch.nn as nn
import torch

def select_knn(skt_fea, img_fea, k = 5):
    skt_fea_norm = nn.functional.normalize(skt_fea, p=2, dim=-1).detach()
    img_fea_norm = nn.functional.normalize(img_fea, p=2, dim=-1).detach()

    similarity_matrix = torch.mm(skt_fea_norm, img_fea_norm.T)
    similarity_scores, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    similarity_scores = torch.softmax(similarity_scores, dim=-1).unsqueeze(-1)
    img_select = img_fea[top_k_indices]

    img_select = img_select * similarity_scores
    img_select = torch.sum(img_select, dim=1)

    return img_select



if __name__ == '__main__':
    # print(timm.list_models())
    # model = mymodel().cuda()
    # data = torch.rand(4, 3, 224, 224).cuda()
    # output = model(data)
    # print(output.shape)
    img_fea = torch.randn(64, 768)
    skt_fea = torch.randn(17, 768)
    img_k = select_knn(skt_fea, img_fea)
    print(img_k.shape)