from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import re


class CatsVsDogsDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = os.path.join(root, 'PetImages')
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'cat': 0, 'dog': 1}
        split = range(0, 10000) if train else range(10000, 12500)
        cat_corrupted = [666]
        dog_corrupted = [11702]
        for i in split:
            cat_file = os.path.join(path, 'Cat', str(i)+'.jpg')
            dog_file = os.path.join(path, 'Dog', str(i)+'.jpg')
            if i not in cat_corrupted:
                self.annotations.append({'name': cat_file, 'label':0})
            if i not in dog_corrupted:
                self.annotations.append({'name': dog_file, 'label':1})

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        
        return image, label

class BloodCellsDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = os.path.join(root, 'dataset2-master', 'images', 'TRAIN' if train else 'TEST')
        
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
        for dir in self.class_to_idx.keys():
            for file_name in sorted(os.listdir(os.path.join(path, dir))):
                self.annotations.append({'name': os.path.join(path, dir, file_name), 'label':self.class_to_idx[dir]})

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ViewPredictionDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = os.path.join(root, 'seg_train/seg_train' if train else 'seg_test/seg_test')
        
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
        for dir in self.class_to_idx.keys():
            for file_name in sorted(os.listdir(os.path.join(path, dir))):
                self.annotations.append({'name': os.path.join(path, dir, file_name), 'label':self.class_to_idx[dir]})

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class COVID19Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.TRAIN_SPLIT = 0.8
        path = os.path.join(root, 'COVID-19_Radiography_Dataset')
        
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'normal': 0, 'covid': 1}
        
        normal_path = os.path.join(path, 'Normal', 'images')
        covid_path = os.path.join(path, 'COVID', 'images')
        normal_len = len(os.listdir(normal_path))
        covid_len = len(os.listdir(covid_path))

        normal_split = range(1, int(normal_len*self.TRAIN_SPLIT)) if train else range(int(normal_len*self.TRAIN_SPLIT), normal_len+1)
        covid_split = range(1, int(covid_len*self.TRAIN_SPLIT)) if train else range(int(covid_len*self.TRAIN_SPLIT), covid_len+1)

        for i in normal_split:
            self.annotations.append({'name': normal_path + '/Normal-' + str(i) + '.png', 'label': self.class_to_idx['normal']})
        for i in covid_split:
            self.annotations.append({'name': covid_path + '/COVID-' + str(i) + '.png', 'label': self.class_to_idx['covid']}) 

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class WeatherPredictionDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = os.path.join(root, 'dataset2/dataset2')
        
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}
        self.TRAIN_SPLIT = 0.8

        lengths = {}
        for file_name in os.listdir(path):
            lengths[re.split('\d+', file_name)[0]] += 1

        for weather in self.class_to_idx.keys():
            split = range(1, int(lengths[weather]*self.TRAIN_SPLIT)) if train else range(int(lengths[weather]*self.TRAIN_SPLIT), lengths[weather]+1)
            for i in split:
                file_name = path + '/' + weather + str(i) + '.jpg'
                if not os.path.exists(file_name):
                    file_name = file_name[:-4] + '.jpeg'
                self.annotations.append({'name': file_name, 'label': self.class_to_idx[weather]}) 

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ConcreteCrackDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.TRAIN_SPLIT = 0.8
        path = os.path.join(root, 'Concrete Crack Images for Classification')
        
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {'negative': 0, 'positive': 1}
        
        negative_path = os.path.join(path, 'Negative')
        positive_path = os.path.join(path, 'Positive')
        negative_len = len(os.listdir(negative_path))
        positive_len = len(os.listdir(positive_path))

        negative_split = range(1, int(negative_len*self.TRAIN_SPLIT)) if train else range(int(negative_len*self.TRAIN_SPLIT), negative_len+1)
        positive_split = range(1, int(positive_len*self.TRAIN_SPLIT)) if train else range(int(positive_len*self.TRAIN_SPLIT), positive_len+1)

        for i in negative_split:
            file_name = negative_path + '/' + '0'*(5 - len(str(i))) + str(i) + '.jpg'
            if not os.path.exists(file_name):
                file_name = file_name[:-4] + '_1.jpg'
            self.annotations.append({'name': file_name, 'label': self.class_to_idx['negative']})
        for i in positive_split:
            file_name = positive_path + '/' + '0'*(5 - len(str(i))) + str(i) + '.jpg'
            if not os.path.exists(file_name):
                file_name = file_name[:-4] + '_1.jpg'
            self.annotations.append({'name': file_name, 'label': self.class_to_idx['positive']}) 

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DIORDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = root
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {
                            'golffield': 0, 'chimney': 1, 'airplane': 2, 
                            'groundtrackfield': 3, 'dam': 4, 'Expressway-Service-area': 5,
                            'storagetank': 6, 'harbor': 7, 'stadium': 8,
                            'trainstation': 9, 'ship': 10, 'tenniscourt': 11,
                            'bridge': 12, 'basketballcourt': 13, 'baseballfield': 14,
                            'Expressway-toll-station': 15, 'airport': 16, 'overpass': 17, 'windmill': 18
                        }
        splits = ['train', 'val'] if train else ['test']
        for split in splits:
            split_path = os.path.join(path, split)
            for file_name in sorted(os.listdir(split_path)):
                self.annotations.append({'name': os.path.join(split_path, file_name), 'label': self.class_to_idx[file_name.split('_')[1]]})

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_name = self.annotations[idx]['name']
        label = self.annotations[idx]['label']
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def CIFAR10(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

def CIFAR100(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    target_transform = transforms.Lambda(lambda x: coarse_labels[x])
    dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform, target_transform=target_transform)
    dataset.class_to_idx = {k:v for k, v in [('class_' + str(i), i) for i in range(20)]}
    return dataset

def FMNIST(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)

def CatsVsDogs(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return CatsVsDogsDataset(root=root, train=train, transform=transform)

def BloodCells(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return BloodCellsDataset(root=root, train=train, transform=transform)

def ViewPrediction(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return ViewPredictionDataset(root=root, train=train, transform=transform)

def COVID19(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return COVID19Dataset(root=root, train=train, transform=transform)

def WeatherPrediction(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return WeatherPredictionDataset(root=root, train=train, transform=transform)

def ConcreteCrack(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return ConcreteCrackDataset(root=root, train=train, transform=transform)

def DIOR(root, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return DIORDataset(root=root, train=train, transform=transform)