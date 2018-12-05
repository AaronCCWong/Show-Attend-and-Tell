import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.transform = transform
        self.split = json.load(open(split_path, 'r'))

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths, self.captions = self.get_split_data(data_path, split_type)

    def __getitem__(self, index):
        img_path = self.img_paths[self.caption_img_idx[index]]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return torch.FloatTensor(img), torch.tensor(self.captions[index])

    def __len__(self):
        return len(self.captions)

    def get_split_data(self, data_path, split_type):
        images = [img for img in self.split['images'] if img['split'] == split_type]

        img_paths, caption_tokens = [], []
        max_length = 0
        for img in images:
            if os.path.isfile(data_path + '/' + img['filepath'] + '/' + img['filename']):
                img_paths.append(data_path + '/' + img['filepath'] + '/' + img['filename'])
            for sen in img['sentences']:
                max_length = max(max_length, len(sen['tokens']))
                self.word_count.update(sen['tokens'])
                caption_tokens.append(sen['tokens'])
                self.caption_img_idx[len(caption_tokens)-1] = len(img_paths)-1

        captions = self.process_caption_tokens(caption_tokens, max_length)

        return img_paths, captions

    def process_caption_tokens(self, caption_tokens, max_length):
        words = [word for word in self.word_count.keys()]
        word_dict = { word: idx + 3 for idx, word in enumerate(words) }
        word_dict['<start>'] = 0
        word_dict['<eos>'] = 1
        word_dict['<unk>'] = 2
        word_dict['<pad>'] = 3

        captions = []
        for tokens in caption_tokens:
            token_idxs = [word_dict[token]
                if word_dict[token] else word_dict['<unk>'] for token in tokens]
            captions.append(
                [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] + \
                [word_dict['<pad>']] * (max_length - len(tokens)))

        return captions
