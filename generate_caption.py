"""
We use beam search to construct the best sentences following a
similar implementation as the author in
https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py

We also use the same strategy as the author to display visualizations
as in the examples shown in the paper. The strategy used is adapted for
PyTorch from here:
https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
"""

import argparse, json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms


def generate_caption_visualization(encoder, decoder, img_path, word_dict, beam_size=5):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)

    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = [token_dict[word_idx] for word_idx in sentence]

    img = img.squeeze(0)
    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    for idx in range(num_words):
        plt.subplot(w, h, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img)
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alpha[idx, 0].reshape(14, 14), upscale=16, sigma=20)
        else:
            alpha_img = skimage.transform.resize(alpha[idx, 0].reshape(14,14), [img.shape[0], img.shape[1]])
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--model', type=str, help='path to model paramters')
    parser.add_argument('--data-path', type=str, default='data/coco',
                        help='path to data (default: data/coco)')
    args = parser.parse_args()

    word_dict = json.load(open(args.data_path + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder()
    decoder = Decoder(vocabulary_size)

    # encoder.cuda()
    # decoder.cuda()

    encoder.eval()
    decoder.eval()

    generate_caption_visualization(encoder, decoder, args.img_path, word_dict)
