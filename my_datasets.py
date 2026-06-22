import math
import torch
from torch.utils import data
import torchvision.transforms as transforms
import cv2
import random
from torch.utils.data import DataLoader

class TestDatasets(data.Dataset):
    def __init__(self, root,datasets = "Sketchy", stage = 1,img_size=(224, 224),m='sk'):
        self.path = []
        self.label = []
        self.root = root+datasets
        label_acc = 0

        for i in range(1,stage+1):
            if m == 'im':
                with open(f"{datasets}/{i}_{datasets}_img.txt", 'r') as f:
                    data =f.readlines()
            else:
                with open(f"{datasets}/{i}_{datasets}_skt_test.txt", 'r') as f:
                    data =f.readlines()

            for item in data:
                label = int(item[:-1].split(' ')[-1])+label_acc
                path = self.root+ item.rsplit(' ', 1)[0]
                self.path.append(path)
                self.label.append(label)

            label_acc = max(self.label)+1

        self.root = root
        self.img_size = img_size
        self.datasets = datasets
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        label = self.label[idx]

        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = self.transform(img)
        return img, label

class TestSingleDatasets(data.Dataset):
    def __init__(self, root,datasets = "Sketchy", stage = 1,img_size=(224, 224),m='sk'):
        self.path = []
        self.label = []
        self.root = root+datasets
        label_acc = 0

        for i in range(1):
            if m == 'im':
                with open(f"{datasets}/{stage}_{datasets}_img.txt", 'r') as f:
                    data =f.readlines()
            else:
                with open(f"{datasets}/{stage}_{datasets}_skt_test.txt", 'r') as f:
                    data =f.readlines()

            for item in data:
                label = int(item[:-1].split(' ')[-1])+label_acc
                path = self.root+ item.rsplit(' ', 1)[0]
                self.path.append(path)
                self.label.append(label)

            label_acc = max(self.label)+1

        self.root = root
        self.img_size = img_size
        self.datasets = datasets
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        label = self.label[idx]

        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = self.transform(img)
        return img, label


class Complete_Datasets(data.Dataset):
    def __init__(self, root, stage, datasets="Sketchy", known_number=0,
                 img_size=(224, 224), v_ratio=0.8, img_ratio=0.1):
        self.root = root+datasets
        self.img_size = img_size
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.path = {}
        with open(f"{datasets}/{stage}_{datasets}_img.txt", 'r') as f:
            path = f.readlines()
        for item in path:
            label = int(item.split(' ')[-1]) + known_number
            if label not in self.path.keys():
                self.path[label] = {"img": [], "skt": []}
            self.path[label]["img"].append(self.root + item.rsplit(' ', 1)[0])

        with open(f"{datasets}/{stage}_{datasets}_skt_train.txt", 'r') as f:
            path = f.readlines()
        for item in path:
            label = int(item.split(' ')[-1]) + known_number
            self.path[label]["skt"].append(self.root + item.rsplit(' ', 1)[0])

        self.label = list(self.path.keys())
        random.shuffle(self.label)
        self.visible = math.ceil((len(self.label))*v_ratio)
        self.img = math.ceil((len(self.label))*img_ratio)
        if len(self.label) - self.visible - self.img <=0 and v_ratio!=1:
            self.visible -= 1
        print("Complete number:",self.visible,"IMG number:",self.img,"SKT number:",len(self.label) - self.visible - self.img)

        self.only_img = self.label[self.visible:self.visible+self.img]
        self.only_skt = self.label[self.visible+self.img:]
        complete = self.label[:self.visible]
        complete.sort()
        print("Complete",complete,"\nOnly Img", self.only_img,
              "  Only Skt", self.only_skt)

        self.name = ['' for i in range(len(self.label))]
        for label in self.label:
            item = self.path[label]["img"][0]
            name = item.split('/')[-2].replace("_"," ").lower()
            self.name[label-known_number] = f"a photo of a {name}"
            # self.name[label-known_number] = name
        self.all_img_label = self.label[:self.visible]+self.label[self.visible+self.img:]
        self.all_skt_label = self.label[:self.visible+self.img]

        self.all_skt_label.sort()
        self.all_img_label.sort()

    def add_old(self, img, skt, only_img, only_skt):
        for key in img.keys():
            if key not in self.path.keys():
                self.path[key] = {"img":[], "skt": []}
                self.path[key]["img"] = img[key]
                self.path[key]["skt"] = skt[key]
            else:
                print("wrong old label")

            self.label.append(key)
            if key in only_img:self.all_img_label.append(key)
            if key in only_skt:self.all_skt_label.append(key)

        for item in only_img:self.only_img.append(item)
        for item in only_skt:self.only_skt.append(item)

        self.all_skt_label.sort()
        self.all_img_label.sort()

    def __len__(self):
        return 4096*2
    def get_number_class(self):
        return len(list(set(self.label)))

    def get_neg_class(self,pos, m ):
        neg = pos
        while neg == pos :
            if m == 'img':
                neg = random.choice(self.all_img_label)
            else:
                neg = random.choice(self.all_skt_label)
        return neg

    def imread_pic(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,self.img_size)
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        label = random.choice(self.label)

        if label in self.only_skt:
            m = 2  #missing img, only have skt
            img = random.choice(self.path[label]['skt'])
            skt = random.choice(self.path[label]['skt'])
        elif label in self.only_img:
            m = 1  #misiing skt, only have img
            skt = random.choice(self.path[label]['img'])
            img = random.choice(self.path[label]['img'])
        else:
            m = 0  # complete
            img = random.choice(self.path[label]['img'])
            skt = random.choice(self.path[label]['skt'])

        img = self.imread_pic(img)
        skt = self.imread_pic(skt)

        neg_label = self.get_neg_class(label,'img')
        neg = random.choice(self.path[neg_label]['img'])
        neg = self.imread_pic(neg)

        neg_skt_label = self.get_neg_class(label,'skt')
        neg_skt = random.choice(self.path[neg_skt_label]['skt'])
        neg_skt = self.imread_pic(neg_skt)

        return img, skt, neg, neg_skt, label, m, neg_label, neg_skt_label



# [0.8997848603303231, 0.7671659991556172, 0.685888390941261, 0.6343107624633003, 0.5796088040547303]
if __name__ == '__main__':
    dataset = Complete_Datasets(root="../", stage=2, datasets="Sketchy", v_ratio=0.8,known_number=25)
    print(dataset.name)
    img, skt, label, m = dataset[0]
    print(img, skt,  label,m)

    # loader = DataLoader(dataset,64)
    # print(dataset.name)
    # print(dataset.get_number_class())
    # for img, skt,  label,m in loader:
    #     print(label)
    #     break

