import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder


def main(args):
    encoder = Encoder()
    decoder = Decoder()

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, 'data/coco/imgs', 'data/coco/dataset.json'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    for epoch in range(1, args.epochs + 1):
        train(epoch, encoder, decoder, optimizer, loss, train_loader)


def train(epoch, encoder, decoder, optimizer, loss, data_loader):
    encoder.eval()
    decoder.train()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        img_features = encoder(imgs)
        decoder(img_features, captions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of the decoder (default: 1e-3)')

    main(parser.parse_args())
