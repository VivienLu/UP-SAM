import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

"""LAHeart modified from https://github.com/grant-jpg/FUSSNet"""

class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, data_dir, split='train', num=None, patch_size=[128,128,128], SAM=False):
        self.data_dir = data_dir
        self.split = split
        self.SAM = SAM

        tr_transform = Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            # RandomNoise(),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop(patch_size),
            ToTensor()
        ])

        if split == 'train':
            data_path = os.path.join(data_dir,'train.txt')
            self.transform = tr_transform
        else:
            data_path = os.path.join(data_dir,'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        self.image_list = [os.path.join(self.data_dir, item, "mri_norm2.h5") for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        print("{} set: total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        if self.SAM:
            if self.split == 'train':
                # return image_.float().clone().detach(), label_.unsqueeze(0).long().clone().detach()
                return image_.float(), label_.unsqueeze(0).long()
            else:
                return image_.float().clone().detach(), label_.unsqueeze(0).long().clone().detach(), self.image_list[idx]   
        else:
            if self.split == 'train':
                return {'image':image_.float(), 'label':label_.long()}
            else:            
                return {'image':image_.float(), 'label':label_.long(), 'name': self.image_list[idx]}

class CreateNewDataset(Dataset):
    def __init__(self, imgs, plabs, masks, labs, crop_size = (112, 112, 80)):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]
        self.plab = [np.squeeze(plab.cpu().numpy()) for plab in plabs]
        self.mask = [np.squeeze(mask.cpu().numpy()) for mask in masks]
        self.lab = [np.squeeze(lab.cpu().numpy()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            CenterCrop(crop_size),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plab, mask, labs = samples
        return {'image': imgs, 'pseudo':plab.long(), 'mask':plab.long(), 'label': labs.long()}

    def __len__(self):
        return self.num

class MaxCenterCrop(object):
    def __init__(self, scale=16):
        self.output_scale = scale

    def _get_transform(self, label):
        max_v = max(label.shape)
        n = (max_v // self.output_scale)
        output_size = n * self.output_scale

        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= output_size[0] or x.shape[1] <= output_size[1] or x.shape[2] <= output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def _get_transform(self, x):
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        def do_transform(image):
            image = np.rot90(image, k)
            image = np.flip(image, axis=axis).copy()
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def _get_transform(self, x):
        noise = np.clip(self.sigma * np.random.randn(x.shape[0], x.shape[1], x.shape[2]), -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        def do_transform(image):
            image = image + noise
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) if i == 0 else s for i, s in enumerate(samples)]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


if __name__ == '__main__':
 
    data_dir = '../../Datasets/LA_dataset'

    trainset = LAHeart(data_dir, num = 10, split='train')
    testset = LAHeart(data_dir, split='test')
 
    train_sample = trainset[0]
    test_sample = testset[0]
 
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 80 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 20 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])

    trainset = LAHeart(data_dir, num = 10, split='train', SAM=True)
    testset = LAHeart(data_dir, split='test', SAM=True)
 
    train_sample = trainset[0]
    test_sample = testset[0]
 
    print(len(trainset), train_sample[0].shape, train_sample[1].shape) # 80 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(testset), test_sample[0].shape, test_sample[1].shape) # 20 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
