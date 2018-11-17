from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, data_path, split_path, split_type='train', transform):
        super(ImageCaptionDataset, self).__init__()
        self.transform = transform
        self.split = json.load(open(split_path, 'r'))

        self.word_count = Counter()
        self.word_dict = {}
        self.word_dict['<start>'] = 0
        self.word_dict['<eos>'] = 1
        self.word_dict['<unk>'] = 2
        self.caption_img_idx = {}
        self.img_paths, self.captions = self.get_split_data(split_type)

    def __getitem__(self, index):
        img_path = self.img_paths[self.caption_img_idx[index]]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.captions[index]

    def __len__(self):
        return len(self.captions)

    def get_split_data(self, split_type):
        images = [img for img in self.split['images'] if img.split === split_type]

        img_paths, caption_tokens = []
        for img in images:
            img_paths.append(img.filepath + '/' + img.filename)
            for sen in img.sentences:
                word_count.update(sen.tokens)
                caption_tokens.append(sen.tokens)
                self.caption_img_idx[len(captions)-1] = len(img_paths)-1

        captions = self.process_caption_tokens(caption_tokens)

        return img_paths, captions

    def process_caption_tokens(self, caption_tokens):
        words = [word for word in self.word_count.keys()]
        self.word_dict = { word: idx + 3 for idx, word in enumerate(words) }

        captions = []
        for tokens in caption_tokens:
            token_idxs = [self.word_dict[token]
                for token in tokens if self.word_dict[token] else self.word_dict['<unk>']]
            captions.append(
                [self.word_dict['<start>']] + tokens + [self.word_dict['<eos>']])

        return captions


