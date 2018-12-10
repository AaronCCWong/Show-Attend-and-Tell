import json, os
import torch
from collections import Counter
from functools import reduce
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_dict = json.load(open(data_path + '/word_dict.json', 'r'))
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        dont_count = (self.word_dict['<start>'], self.word_dict['<eos>'], self.word_dict['<pad>'])
        caption_length = reduce((lambda x, y: x if y in dont_count else x + 1), self.captions[index], 0)

        if self.split_type == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor([caption_length])

        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), \
            torch.tensor(all_captions), torch.tensor([caption_length])

    def __len__(self):
        return len(self.captions)
