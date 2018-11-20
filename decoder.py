import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.init_h = nn.Linear(512, 512)
        self.init_c = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

        self.embedding = nn.Embedding(23531, 512)
        self.lstm = nn.LSTMCell(1024, 512)

    def forward(self, img_features, captions):
        c0, h0 = self.get_init_lstm_state(img_features)
        embedding = self.embedding(captions)


    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return c, h
