"""
We use beam search to construct the best sentences following a
similar implementation as the author in
https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
"""

import argparse
import torch

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms


def generate_caption(encoder, decoder, img_path, beam_size=5):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    prev_words = torch.zeros(beam_size, 1).long()

    sentences = prev_words
    top_preds = torch.zeros(beam_size, 1)
    alphas = torch.ones(beam_size, 1, img_features.size(1))

    completed_sentences = []
    completed_sentences_alphas = []
    completed_sentences_preds = []

    step = 1
    h, c = decoder.get_init_lstm_state(img_features)

    while True:
        embedding = decoder.embedding(prev_words).squeeze(1)
        context, alpha = decoder.attention(img_features, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        gated_context = gate * context

        lstm_input = torch.cat((embedding, gated_context), dim=1)
        h, c = decoder.lstm(lstm_input, (h, c))
        output = decoder.tanh(decoder.deep_output(h))
        output = top_preds.expand_as(output) + output

        if step == 1:
            top_preds, top_words = output[0].topk(beam_size, 0, True, True)
        else:
            top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
        prev_word_idxs = top_words / output.size(1)
        next_word_idxs = top_words % output.size(1)

        sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
        alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

        incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
        complete = list(set(range(len(next_word_idxs))) - set(incomplete))

        print(sentences[complete])
        break
        # if len(complete) > 0:
        #     completed_sentences.extend(sentences[complete].tolist())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--model', type=str, help='path to model paramters')
    args = parser.parse_args()

    encoder = Encoder()
    decoder = Decoder()

    # encoder.cuda()
    # decoder.cuda()

    encoder.eval()
    decoder.eval()

    generate_caption(encoder, decoder, args.img_path)
