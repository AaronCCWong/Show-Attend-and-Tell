import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder


def main(args):
    train_writer = SummaryWriter()
    validation_writer = SummaryWriter()

    encoder = Encoder()
    decoder = Decoder()

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, 'data/coco/imgs', 'data/coco/dataset.json'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    print('Starting training with {}'.format(args))
    for epoch in range(1, args.epochs + 1):
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, args.alpha_c, args.log_interval, train_writer)
        model_file = 'model/model_' + str(epoch) + '.pth'
        torch.save(decoder.state_dict(), model_file)
        print('Saved model to ' + model_file)



def train(epoch, encoder, decoder, optimizer, cross_entropy_loss, data_loader, alpha_c, log_interval, writer):
    encoder.cuda()
    decoder.cuda()

    encoder.eval()
    decoder.train()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()
        img_features = encoder(imgs)
        optimizer.zero_grad()
        preds, alphas = decoder(img_features, captions)
        targets = captions[:, 1:]

        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0] 

        att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/epoch_{}_loss'.format(epoch), loss.item(), batch_idx)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of the decoder (default: 1e-3)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')

    main(parser.parse_args())
