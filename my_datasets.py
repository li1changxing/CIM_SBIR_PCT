import math

from torch.utils import data
import torchvision.transforms as transforms
import cv2
from glob import glob
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


class Complete_Datasets(data.Dataset):
    def __init__(self, root,stage, datasets="Sketchy", img_size=(224,224),v_ratio = 0.8):
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
            label = int(item.split(' ')[-1])
            if label not in self.path.keys():
                self.path[label] = {"img":[],"skt":[]}
            self.path[label]["img"].append(self.root+ item.rsplit(' ', 1)[0])

        with open(f"{datasets}/{stage}_{datasets}_skt_train.txt", 'r') as f:
            path = f.readlines()
        for item in path:
            label = int(item.split(' ')[-1])
            self.path[label]["skt"].append(self.root+ item.rsplit(' ', 1)[0])

        self.label = list(self.path.keys())
        random.shuffle(self.label)
        self.visible = math.ceil((max(self.label)+1)*v_ratio)
        # self.visible = math.ceil((max(self.label)+1)*0.6)
        self.img = math.ceil((max(self.label)+1)*((1-v_ratio)/2))
        if max(self.label)+1 - self.visible - self.img <=0:
            self.visible -= 1
        print("Complete number:",self.visible,"IMG number:",self.img)

        print("Complete",self.label[:self.visible],"\nImg",self.label[self.visible:self.visible+self.img],"\nSkt",self.label[self.visible+self.img:])
        self.batch_size = 128
        self.batch_dual_size = 64
        self.batch_only_image_size = 32
        self.batch_only_sketch_size = 32

        self.name = ['' for i in range(len(self.label))]
        for label in self.label:
            item = self.path[label]["img"][0]
            name = item.split('/')[-2]
            self.name[label] = name

    def __len__(self):
        return 4096

    def get_number_class(self):
        return max(self.label)+1

    def clip(self,batch_size,batch_dual_size,batch_only_image_size,batch_only_sketch_size):
        self.batch_size = batch_size
        self.batch_dual_size = batch_dual_size
        self.batch_only_image_size = batch_only_image_size
        self.batch_only_sketch_size = batch_only_sketch_size

    def __getitem__(self, idx):
        idx = idx % self.batch_size
        if idx < self.batch_dual_size:
            label = random.choice(self.label[:self.visible])
        elif idx < self.batch_only_image_size+self.batch_dual_size:
            label = random.choice(self.label[self.visible:self.visible+self.img])
        else:
            label = random.choice(self.label[self.visible+self.img:])

        path = random.choice(self.path[label]['img'])
        name = self.name[label]

        img = cv2.imread(path)
        img = cv2.resize(img,self.img_size)
        img = self.transform(img)

        path = random.choice(self.path[label]['skt'])
        skt = cv2.imread(path)
        skt = cv2.resize(skt, self.img_size)
        skt = self.transform(skt)

        return img,skt, label,name


if __name__ == '__main__':
    a = Complete_Datasets("../hy-tmp/",1)
    # img,skt, label = a[0]
    # print(img.shape,label)
    # img,skt, label = a[72]
    # print(img.shape,label)
    # img,skt, label = a[-10]
    # print(img.shape,label)

    a.clip(128,116,7,5)
    print(a.name)
    # loader = DataLoader(a, batch_size=128, shuffle=False, drop_last=False)
    #
    # for idx,data in enumerate(loader):
    #     img, skt, label =data
    #     dual_label = label[:116]
    #     img_label = label[116:116+7]
    #     skt_label = label[116+7:]
    #     print(set(dual_label),set(img_label),set(skt_label))
    #     break
    #
    # # for i in batch:
    #     img, label  = i
    #     print(img.shape,label)
