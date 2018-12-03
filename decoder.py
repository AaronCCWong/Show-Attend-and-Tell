import torch
import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.init_h = nn.Linear(512, 512)
        self.init_c = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, 23531)

        self.attention = Attention()
        self.embedding = nn.Embedding(23531, 512)
        self.lstm = nn.LSTMCell(1024, 512)

    def forward(self, img_features, captions):
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long()
        embedding = self.embedding(captions) if self.training else self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, 23531).cuda()
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda()
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.tanh(self.deep_output(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training:
                embedding = self.embedding(output.max(1)[1])
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c
