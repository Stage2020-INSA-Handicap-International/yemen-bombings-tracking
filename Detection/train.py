'''
train.py from standard YOLO script
Change width/height in yolov3.cfg
'''


from __future__ import division

import os
import sys
import argparse
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import Tensor
import torch.optim as optim
from terminaltables import AsciiTable

from torchsummary import summary

from model import *
from datasets import *
from utils import *
from config.parse_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient-accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model-def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data-config", type=str, default="config/yemen.data", help="path to data config file")
    parser.add_argument("--pretrained-weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img-size", type=int, default=4800, help="size of each image dimension")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation-interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute-map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale-training", default=True, help="allow for multi-scale training")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output")
    # os.makedirs("checkpoints")

    # Get data configuration
    data_config = parse_data_config(args.data_config)
    h5_train = data_config["train"]
    h5_valid = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(args.model_def).to(device)
    # summary(model, input_size=(3, args.img_size, args.img_size))
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)

    # Get dataloader
    dataset = DetectDataset(h5_train, augment=True, multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False) #.double()
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % args.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *["YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [("{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += "\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
            del imgs, targets

        '''if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")'''

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_%d.pth" % epoch)