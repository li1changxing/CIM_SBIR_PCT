import random

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model_v1.my_model import mymodel


class CurrentDatasets(data.Dataset):
    def __init__(self, root,datasets = "Sketchy",stage=1, known_class = 0,img_size=(224, 224),m='sk',s='',dtype="great"):
        self.path = []
        self.label = []
        self.root = root+datasets
        self.dtype = dtype

        if m == 'im':
            with open(s+f"{datasets}/{stage}_{datasets}_img.txt", 'r') as f:data =f.readlines()
        else:
            with open(s+f"{datasets}/{stage}_{datasets}_skt_train.txt", 'r') as f:data =f.readlines()

        for item in data:
            label = int(item[:-1].split(' ')[-1])+known_class
            path = self.root+ item.rsplit(' ', 1)[0]
            self.path.append(path)
            self.label.append(label)

        self.img_size = img_size
        self.datasets = datasets
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):return len(self.path)

    def num_classes(self):
        return len(set(self.label))

    def __getitem__(self, idx):
        path = self.path[idx]
        label = self.label[idx]

        if self.dtype == "random":
            return path, label

        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = self.transform(img)
        # return img,path.replace(self.root,''), label
        return img, path, label


def get_great_sample(model,root,datasets = "Sketchy", known_class = 0,stage=1,only={"img":[],"skt":[]},s=""):
    print("restore old sample",end=" ")
    img_loader = CurrentDatasets(root,datasets,stage,known_class,m='im',s=s)
    skt_loader = CurrentDatasets(root,datasets,stage,known_class,m='sk',s=s)

    print(known_class,"to",img_loader.num_classes() + known_class)
    print(only)
    num_classes = img_loader.num_classes()

    img_loader = DataLoader(img_loader,batch_size=256,num_workers=4)
    skt_loader = DataLoader(skt_loader,batch_size=256,num_workers=4)

    img_bank = []
    img_label = []
    img_path = []
    skt_bank = []
    skt_label = []
    skt_path = []

    with torch.no_grad():
        for idx, (photo,path, label) in enumerate(tqdm(img_loader)):
            photo, label = photo.cuda(), label
            photo_reprs = model(photo).detach().cpu()
            for i in label:
                img_label.append(i)
            for i in path:
                img_path.append(i)
            img_bank.append(photo_reprs)

        image_features = torch.cat(img_bank).detach()
        labels_images = torch.tensor(img_label).view(-1)

        for idx, (sketch,path, label) in enumerate(tqdm(skt_loader)):
            sketch, label = sketch.cuda(), label
            sketch_reprs = model(sketch).detach().cpu()
            for i in label:
                skt_label.append(i)
            for i in path:
                skt_path.append(i)
            skt_bank.append(sketch_reprs)

        sketch_features = torch.cat(skt_bank).detach()
        labels_sketches = torch.tensor(skt_label).view(-1)

    select_img = {}
    select_skt = {}

    for label in range(known_class,num_classes+known_class):
        # dual
        if label not in only["img"] and label not in only["skt"]:
            img_indices = torch.nonzero(labels_images == label, as_tuple=True)
            skt_indices = torch.nonzero(labels_sketches == label, as_tuple=True)
            item_img_path = np.array(img_path)[img_indices[0].numpy()]
            item_skt_path = np.array(skt_path)[skt_indices[0].numpy()]
            item_img = image_features[img_indices[0]]
            item_skt = sketch_features[skt_indices[0]]

            item_img = nn.functional.normalize(item_img, dim=1, p=2)
            item_skt = nn.functional.normalize(item_skt, dim=1, p=2)

            score = torch.mm(item_img, item_skt.T)
            #img
            values, indices = torch.topk(score,len(item_skt)//2, dim=1)
            values = torch.sum(values,dim=1)
            if values.shape[0]<=10:
                select_img[label] = item_img_path.tolist()
            else:
                _, indices = torch.topk(values, 10, dim=0)
                select_img[label] = item_img_path[indices.numpy()].tolist()

            #skt
            values, indices = torch.topk(score.T,len(item_img)//2, dim=1)
            values = torch.sum(values,dim=1)
            if values.shape[0]<=10:
                select_skt[label] = item_skt_path.tolist()
            else:
                _, indices = torch.topk(values, 10, dim=0)
                select_skt[label] = item_skt_path[indices.numpy()].tolist()

        #only img
        elif label in only["img"]:
            img_indices = torch.nonzero(labels_images == label, as_tuple=True)
            item_img_path = np.array(img_path)[img_indices[0].numpy()]
            item_img = image_features[img_indices[0]]
            item_img = nn.functional.normalize(item_img, dim=1, p=2)
            score = torch.mm(item_img, item_img.T)
            values, indices = torch.topk(score, len(item_img) // 2, dim=1)
            values = torch.sum(values, dim=1)
            if values.shape[0]<=10:
                select_img[label] = item_img_path.tolist()
            else:
                _, indices = torch.topk(values, 10, dim=0)
                select_img[label] = item_img_path[indices.numpy()].tolist()
            select_skt[label] = []

        #only skt
        elif label in only["skt"]:
            skt_indices = torch.nonzero(labels_sketches == label, as_tuple=True)
            item_skt_path = np.array(skt_path)[skt_indices[0].numpy()]
            item_skt = sketch_features[skt_indices[0]]

            item_skt = nn.functional.normalize(item_skt, dim=1, p=2)
            score = torch.mm(item_skt, item_skt.T)
            # skt
            values, indices = torch.topk(score, len(item_skt) // 2, dim=1)
            values = torch.sum(values, dim=1)
            if values.shape[0]<=10:
                select_skt[label] = item_skt_path.tolist()
            else:
                _, indices = torch.topk(values, 10, dim=0)
                select_skt[label] = item_skt_path[indices.numpy()].tolist()
            select_img[label] = []

    return select_img,select_skt

def get_random_sample(model,root,datasets = "Sketchy", known_class = 0,stage=1,only={"img":[],"skt":[]},s=""):
    print("restore old sample",end=" ")
    img_loader = CurrentDatasets(root,datasets,stage,known_class,m='im',s=s, dtype="random")
    skt_loader = CurrentDatasets(root,datasets,stage,known_class,m='sk',s=s, dtype="random")

    print(known_class,"to",img_loader.num_classes() + known_class)
    print(only)
    num_classes = img_loader.num_classes()

    img_loader = DataLoader(img_loader,batch_size=256,num_workers=4)
    skt_loader = DataLoader(skt_loader,batch_size=256,num_workers=4)

    img_label = []
    img_path = []
    skt_label = []
    skt_path = []

    with torch.no_grad():
        for idx, (path, label) in enumerate(tqdm(img_loader)):
            for i in label:
                img_label.append(i)
            for i in path:
                img_path.append(i)
        labels_images = torch.tensor(img_label).view(-1)

        for idx, (path, label) in enumerate(tqdm(skt_loader)):
            for i in label:
                skt_label.append(i)
            for i in path:
                skt_path.append(i)

        labels_sketches = torch.tensor(skt_label).view(-1)

    select_img = {}
    select_skt = {}

    for label in range(known_class,num_classes+known_class):
        # dual
        if label not in only["img"] and label not in only["skt"]:
            img_indices = torch.nonzero(labels_images == label, as_tuple=True)
            skt_indices = torch.nonzero(labels_sketches == label, as_tuple=True)
            item_img_path = np.array(img_path)[img_indices[0].numpy()]
            item_skt_path = np.array(skt_path)[skt_indices[0].numpy()]

            if len(item_img_path.tolist())>10:
                select_img[label] = random.sample( item_img_path.tolist(), k=10)
            else:
                select_img[label] = item_img_path.tolist()

            if len(item_skt_path.tolist())>10:
                select_skt[label] = random.sample( item_skt_path.tolist(), k=10)
            else:
                select_skt[label] = item_skt_path.tolist()

        #only img
        elif label in only["img"]:
            img_indices = torch.nonzero(labels_images == label, as_tuple=True)
            item_img_path = np.array(img_path)[img_indices[0].numpy()]
            if len(item_img_path.tolist())>10:
                select_img[label] = random.sample( item_img_path.tolist(), k=10)
            else:
                select_img[label] = item_img_path.tolist()
            select_skt[label] = []

        #only img
        elif label in only["skt"]:
            skt_indices = torch.nonzero(labels_sketches == label, as_tuple=True)
            item_skt_path = np.array(skt_path)[skt_indices[0].numpy()]

            if len(item_skt_path.tolist())>10:
                select_skt[label] = random.sample( item_skt_path.tolist(), k=10)
            else:
                select_skt[label] = item_skt_path.tolist()

            select_img[label] = []

    return select_img,select_skt

if __name__ == '__main__':
    model = mymodel(25).cuda()
    parameters = torch.load(f'../weights/tmp/model_Sketchy_11.pth')
    model.load_state_dict(parameters['model'],strict=False)
    model.eval()
    parameters['img_label'].sort()
    parameters['skt_label'].sort()
    print(parameters['img_label'])
    print(parameters['skt_label'])

    # img, skt = get_great_sample(model,root="../../",missing={"img":[14],"skt":[7]},s="../")
    # print(img)
    # print(skt)
    #
    # save_dict = {
    #     "model": model.state_dict(),
    #     "prototype_txt": parameters["prototype_txt"].cpu(),
    #     "img_label": parameters['img_label'],
    #     "skt_label": parameters['skt_label'],
    #     'img_old' : img,
    #     'skt_old' : skt,
    # }
    # torch.save(save_dict, f'weights/model_Sketchy_11.pth')

    # data = torch.randn((493,548))
    # data = torch.sum(data,dim=1)
    # print(data.shape)
    # values, indices = torch.topk(data, 10, dim=0)
    # print(values.shape,indices)