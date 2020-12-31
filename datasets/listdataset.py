import torch.utils.data as data
import cv2
import numpy as np
import json

class ListDataset(data.Dataset):
    def __init__(self, root, dataset, img_path_list, label_path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None, datatype=None):

        voidId = 19
        self.root = root
        self.dataset = dataset
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype

        # create id map
        with open('datasets/config.json') as config_file:
            self.labels = json.load(config_file)['labels']
        self.id2trainId = {d['id']: voidId if d['trainId'] == 255 else d['trainId'] for d in self.labels}

    def __getitem__(self, index):
        # img_path = self.img_path_list[index][:-1]

        # print(self.img_path_list)

        img_path = self.img_path_list[index]
        label_path = self.label_path_list[index]

        print('img_path={}'.format(str(img_path)))
        print('label_path={}'.format(str(label_path)))

        # We do not consider other datsets in this work
        assert self.dataset == 'cityscapes'
        assert (self.transform is not None) and (self.target_transform is not None)

        inputs = cv2.imread(str(img_path), 1)
        label_tmp = cv2.imread(str(label_path), 0)  # id map
        # inputs, label = self.loader(img_path, img_path.replace('_img.jpg', '_label.png'))

        # create trainId map
        label = np.zeros_like(label_tmp)
        for k, v in self.id2trainId.items():
            label[label_tmp == k] = v

        if self.co_transform is not None:
            inputs, label = self.co_transform([inputs], label)

        if self.transform is not None:
            image = self.transform(inputs[0])

        label = label.reshape(label.shape[0], label.shape[1], 1)
        print(label.shape)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.img_path_list)
