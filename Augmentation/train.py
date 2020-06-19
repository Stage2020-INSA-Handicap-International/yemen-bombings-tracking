import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import SRCNN, Subpixel, FSRCNN
import datasets
from datasets import TrainDataset, ValDataset
from utils import AverageMeter, calc_psnr
from torchsummary import summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--val-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model', type=str, default="SRCNN", help="SRCNN, FSRCNN, Subpixel")
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    print("Model : {}".format(args.model))
    if args.model == "SRCNN":
        model = SRCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)
    elif args.model == "Subpixel": #TODO MAKE SUBPIXEL WORK
        model = Subpixel(upscale_factor=args.scale).to(device)
        criterion = nn.MSELoss()
        optimizer = optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
    elif args.model == "FSRCNN": #TODO MAKE FSRCNN WORK
        model = FSRCNN(scale_factor=args.scale).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)

    # datasets.data_print(args.train_file)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    val_dataset = ValDataset(args.val_file)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    summary(model, input_size=(1, 32, 32))

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

                del inputs, labels, data

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, '{}_epoch_{}.pth'.format(args.model,epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in val_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

            del inputs, labels, data

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, '{}_best.pth'.format(args.model)))