"""
@author: Wei Han
Arrange information for complex scenes via dynamic clustering

Notes:
    The flow of data is quite complex. It includes
        - feeding all data into encoder for clustering,
        - and taking clusters as data for localized tasks,
        - and batches for encoder update
"""

import numpy as np
import torch
import config as cf
import copy

import torchvision
import torchvision.transforms as transforms

import os
import sys
from sklearn.preprocessing import OneHotEncoder

from PIL import Image
import os.path
import pickle
import torch.utils.data as data

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool-batches-py, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, valid=False, classes=np.arange(100), transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.valid = valid

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = np.array(self.train_labels)

            train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
            train_labels = np.empty((0,)).astype(np.int32)
            same = classes == np.unique(self.train_labels)
            same = same if isinstance(same, bool) else same.all()
            if not same:
                for class_label in classes:
                    train_data = np.vstack((train_data, self.train_data[self.train_labels==class_label]))
                    train_labels = np.hstack((train_labels, self.train_labels[self.train_labels==class_label]))
                self.train_data = train_data
                self.train_labels = train_labels

            if self.valid:
                labels, class_idx = np.unique(self.train_labels, return_inverse=True)

                # Sample 20% data as validation set (each label)
                temp_train_data = self.train_data
                temp_train_labels = self.train_labels
                self.train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                self.train_labels = np.empty((0,)).astype(np.int32)
                self.valid_data  = np.empty((0, 32, 32, 3)).astype(np.uint8)
                self.valid_labels = np.empty((0,)).astype(np.int32)

                for label in labels:
                    num_class = sum((class_idx == label).astype(int))
                    self.train_data = np.vstack((self.train_data, temp_train_data[class_idx == label][int(num_class * 0.2):, :, :, :]))
                    self.train_labels = np.hstack((self.train_labels, temp_train_labels[class_idx == label][int(num_class * 0.2):]))
                    self.valid_data = np.vstack((self.valid_data, temp_train_data[class_idx == label][:int(num_class * 0.2), :, :, :]))
                    self.valid_labels = np.hstack((self.valid_labels, temp_train_labels[class_idx == label][:int(num_class * 0.2)]))


        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            test_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
            test_labels = np.empty((0,)).astype(np.int32)
            same = classes == np.unique(self.test_labels)
            same = same if isinstance(same, bool) else same.all()
            if not same:
                for class_label in classes:
                    test_data = np.vstack((test_data, self.test_data[self.test_labels == class_label]))
                    test_labels = np.hstack((test_labels, self.test_labels[self.test_labels == class_label]))
                self.test_data = test_data
                self.test_labels = test_labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

class Validset():
    def __init__(self, trainset):
        self.trainset = trainset
        self.train_data = trainset.valid_data
        self.train_labels = trainset.valid_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.trainset.transform is not None:
            img = self.trainset.transform(img)

        if self.trainset.target_transform is not None:
            target = self.trainset.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


def get_pretrain_dataLoders(args, valid=False, one_hot=True):
    print('\nData Preparation')
    # Data Uplaod
    data_transform = transforms.Compose([
         transforms.ToTensor(),
    ])

    root_path = '/Users/changgang/Documents/DATA/Data For Research/CIFAR/'
    if(args.dataset == 'cifar-10'):
        trainset = CIFAR10(root=root_path, train=True, valid=valid, classes=np.arange(10), transform=data_transform)
        testset = CIFAR10(root=root_path, train=False, classes=np.arange(10), transform=data_transform)
    else:
        assert args.dataset == 'cifar-100'
        trainset = CIFAR100(root=root_path, train=True, valid=valid, classes=np.arange(100), transform=data_transform)
        testset = CIFAR100(root=root_path, train=False, classes=np.arange(100), transform=data_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)
    trainbase = torch.utils.data.DataLoader(copy.deepcopy(trainset), batch_size=512, shuffle=False, num_workers=0)

    if one_hot:
        label_transformer = OneHotEncoder(sparse=False).fit(np.array(trainloader.dataset.train_labels).reshape(-1, 1)) #, categories='auto'
        trainloader.dataset.train_labels = label_transformer.transform(np.array(trainloader.dataset.train_labels).reshape(-1, 1))
        testloader.dataset.test_labels = label_transformer.transform(np.array(testloader.dataset.test_labels).reshape(-1, 1))

    validloader = None
    if valid:
        validset = Validset(copy.deepcopy(trainset))
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
        assert (np.unique(validloader.dataset.train_labels) == np.unique(trainbase.dataset.train_labels)).all()
        if one_hot:
            validloader.dataset.train_labels = label_transformer.transform(np.array(validloader.dataset.train_labels).reshape(-1, 1))

    return trainloader, testloader, trainbase, validloader


def get_dataLoder(args, classes, one_hot=True):
    # Data Uplaod
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    root_path = '/Users/changgang/Documents/DATA/Data For Research/CIFAR/'
    if (args.dataset == 'cifar-10'):
        trainset = CIFAR10(root=root_path, train=True, valid=False, classes=classes, transform=data_transform)
    else:
        assert args.dataset == 'cifar-100'
        trainset = CIFAR100(root=root_path, train=True, valid=False, classes=classes, transform=data_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    if one_hot:
        label_transformer = OneHotEncoder(sparse=False).fit(np.array(trainloader.dataset.train_labels).reshape(-1, 1))  #, categories='auto'
        trainloader.dataset.train_labels = label_transformer.transform(np.array(trainloader.dataset.train_labels).reshape(-1, 1))

    return trainloader