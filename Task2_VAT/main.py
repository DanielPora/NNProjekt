import argparse
import math
import warnings
from copy import deepcopy
import time
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy
from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
from torch.nn import functional

warnings.filterwarnings('ignore')


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # Cross Entropy for training loss
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Learning Rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Early Stopping
    best_test_loss = np.inf
    best_acc = 0
    epochs_without_improv = 0
    stop_crit = args.stop_epochs

    # Other hyperparameters
    lambda_ = 1

    # Output Directory
    # Output Directory
    if args.dataset == "cifar10":
        if args.num_labeled == 4000:
            model_out = f"final_models/test/cifar10/4000/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}.pt"
            writer = SummaryWriter(log_dir=f"./runs/cifar10/4klabels/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}")
        else:
            model_out = f"final_models/test/cifar10/250/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}.pt"
            writer = SummaryWriter(log_dir=f"./runs/cifar10/250labels/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}")
    else:
        if args.num_labeled == 10000:
            model_out = f"final_models/test/cifar100/10000/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}.pt"
            writer = SummaryWriter(log_dir=f"./runs/cifar100/10klabels/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}")
        else:
            model_out = f"final_models/test/cifar100/2500/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}.pt"
            writer = SummaryWriter(log_dir=f"./runs/cifar100/2500labels/lr{args.lr}wd{args.wd}momentum{args.momentum}epsilon{args.vat_eps}")

    file = open(f"/log.txt", "a")

    ############################################################################
    for epoch in range(args.epoch):
        begin = time.perf_counter()
        optimizer.zero_grad()
        model.train()
        print(f"Epoch: {epoch + 1} / {args.epoch}")
        file.write(f"Epoch: {epoch + 1} / {args.epoch}\n")
        train_losses = []
        visualize = True

        # for i in range(args.iter_per_epoch):
        for i in range(len(labeled_loader)):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            if args.base:
                # Don't use VAT training
                predictions = model.forward(x_l)
                loss = criterion(predictions, y_l)
            else:
                # Combine both batches of labeled and unlabeled data
                minibatch = torch.cat((x_l, x_ul), 0)
                # Compute VAT loss and only visualize in the first itertation
                vat_loss = VATLoss(args=args, writer=writer, visualize=visualize, epoch=epoch, dataset=args.dataset)
                visualize = False
                va_loss = vat_loss(model, minibatch)
                # Predict and compute loss of labeled data, add VAT loss to labeled data's loss
                predictions = model.forward(x_l)
                classification_loss = criterion(predictions, y_l)
                loss = classification_loss + lambda_ * va_loss

            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            ####################################################################

        scheduler.step()

        avg_train_loss = sum(train_losses)/len(train_losses)
        print(f"Training Loss: {avg_train_loss}")
        file.write(f"Training Loss: {avg_train_loss}\n")
        writer.add_scalar("Loss / Train Loss per Epoch", avg_train_loss, epoch)

        # Evaluation
        # Accuracy
        model.eval()
        acc = []
        losses = []
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model.forward(imgs)
            acc.append(accuracy(out, labels)[0].item())
            loss = criterion(out, labels)
            losses.append(loss.item())
        avg_acc = sum(acc) / len(acc)
        avg_test_loss = sum(losses) / len(losses)
        print(f"Accuracy: {avg_acc}")
        print(f"Test Loss: {avg_test_loss}")
        file.write(f"Accuracy: {avg_acc}\n")
        file.write(f"Test Loss: {avg_test_loss}\n")
        writer.add_scalar("Accuracy / Test Accuracy per Epoch", avg_acc, epoch)
        writer.add_scalar("Loss / Test Loss per Epoch", avg_test_loss, epoch)
        end = time.perf_counter()
        print("time needed: ", round(end - begin, 3), "sec")
        file.write(f"time needed: {round(end - begin, 3)} sec\n")

        # Early stopping
        # if avg_test_loss < best_test_loss:
        if avg_acc > best_acc:
            best_test_loss = avg_test_loss
            best_acc = avg_acc
            epochs_without_improv = 0
            best_epoch = epoch+1
            best_model_state = deepcopy(model.state_dict())
        elif epochs_without_improv == stop_crit:
            print(f"Early Stopping in epoch {epoch+1} as no improvements were made for {stop_crit} epochs.")
            print(f"Best Test Loss: {best_test_loss}")
            print(f"Best Accuracy: {best_acc}")
            print(f"Best Epoch: {best_epoch}")
            file.write(f"Early Stopping in epoch {epoch+1} as no improvements were made for {stop_crit} epochs.\n")
            file.write(f"Best Test Loss: {best_test_loss}\n")
            file.write(f"Best Accuracy: {best_acc}\n")
            file.write(f"Best Epoch: {best_epoch}\n")
            break
        else:
            epochs_without_improv += 1

    torch.save(best_model_state, f"{model_out}")
    file.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='test batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter")
    parser.add_argument("--vat-iter", default=1, type=int,
                        help="VAT iteration parameter")
    parser.add_argument("--stop-epochs", default=30, type=int,
                        help="Epochs until early stopping")
    parser.add_argument("--base", default=False, type=bool,
                        help="Deactivate VAT Training")
    parser.add_argument("--gamma", type=float, default=0.97,
                        help="learning rate scheduler decrease")
    parser.add_argument("--model-dic", type=str, default="./final_models")
    parser.add_argument("--optim", type=str, default="Adam")
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)