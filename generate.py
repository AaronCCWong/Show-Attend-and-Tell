import argparse
import torch

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms


def generate_caption(encoder, decoder, img_path, beam_size=5):
    """
    We use beam search to construct the best sentences following a
    similar implementation as the author in
    https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
    """
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img_features = encoder(img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--model', type=str, help='path to model paramters')
    args = parser.parse_args()

    encoder = Encoder()
    decoder = Decoder()

    encoder.cuda()
    decoder.cuda()

    encoder.eval()
    decoder.eval()

    generate_caption(encoder, decoder, args.img_path)
