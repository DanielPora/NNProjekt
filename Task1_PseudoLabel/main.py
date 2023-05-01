import argparse
import math
import random
import test
from dataloader import get_cifar10, get_cifar100
import utils

from model.wrn  import WideResNet
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data   import DataLoader
import warnings
import time
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy



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

    # test_erg = test.test_cifar100(test_dataset, "./output/cifar100/10000/SGD_0.03lr_0.001wd_0.5drp_0.99g_baseline_1207ep_err.pt")
    # print("test erg", test_erg[0:4])
    #
    # wait = input()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
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
                                args.num_classes, widen_factor=args.model_width, dropRate=args.dropout)
    model       = model.to(device)


    ############################################################################
    # TODO: SUPPLY your code

    #test_cifar10(test_dataset, "./output/cifar10_250lbl_baseline_err.pt")

    # building the valset
    valset = []
    for i in range(20):
        valset.append(next(unlabeled_loader))

    unlabeled_set = []
    n_ulabel = 0
    while True:
        try:
            unlabeled_set.append(next(unlabeled_loader))
            n_ulabel +=1
        except StopIteration:
            break
    unlabeled_loader = iter(unlabeled_set)

    #constants
    criterion = nn.CrossEntropyLoss()
    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        print("Using Adam optimizer")
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)
        print("Using SGD optimizer")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    softmax = nn.Softmax(dim=1)
    threshold = args.threshold
    stopping = args.stopping  #after how many epochs without improvement early stopping is applied
    lr = args.lr

    # utility
    errors = []  #collect the error rate for the epochs
    n_label = len(labeled_loader)

    n_test = len(test_loader)
    n_iters = n_label * 1


    #naming the model
    if args.base:
        model_name = f"{args.optim}_{args.lr}lr_{args.wd}wd_{args.dropout}drp_{args.gamma}g_baseline_{args.epoch}ep"
    else:
        model_name = f"{args.threshold}thrshld_{args.epoch}ep_test"

    writer = SummaryWriter(f"{args.dataout}{args.dataset}/{args.num_labeled}/{model_name}")

    best_err = 0
    best_val = np.inf
    n_no_progress = 0
    print(f"label batches: {n_label}, size: {len((next(labeled_loader))[1])}")
    print(f"unlabel batches: {n_ulabel}, size: {len((next(unlabeled_loader))[1])}")
    print(f"test batches: {n_test}")
    ############################################################################

    pseudo_set = []
    n_pl_used = 0
    for epoch in range(args.epoch):
        begin = time.perf_counter()
        lr = scheduler.get_last_lr()
        print(f"LR: {lr}")
        optimizer.zero_grad()
        model.train()
        use_pseudo = True
        epochloss = 0
        testloss = 0
        val_loss = 0

        if n_iters > len(pseudo_set):
            iters = n_iters
        else:
            iters = len(pseudo_set)

        # n_iters is by default number of training batches
        for l in range(iters):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)

            x_l, y_l    = x_l.to(device), y_l.to(device)

            try:
                x_p, y_p = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_set)
                x_p, y_p = next(pseudo_iter)
            except UnboundLocalError:
                use_pseudo = False






            ####################################################################
            # TODO: SUPPLY your code
            # Training step

            lab_out = model.forward(x_l)
            labeled_loss = criterion(lab_out, y_l)

            alpha = utils.n_alpha(epoch, args.epoch)
            if use_pseudo:
                x_p, y_p = x_p.to(device), y_p.to(device)
                ps_out = model.forward(x_p)
                ps_loss = criterion(ps_out, y_p)
                loss = labeled_loss / n_iters + alpha * ps_loss / len(pseudo_set)
            else:
                loss = labeled_loss

            loss.backward()
            epochloss += loss.detach().cpu().numpy()

            optimizer.step()

        scheduler.step()


        # determine influence of unlabeled loss


        n_pl_used = 0
        if args.base:
            pass
        else:
            # get pseudo labels

            if alpha > 0:
                pseudo_pic_set = torch.Tensor()
                pseudo_lab_set = torch.IntTensor()
                while True:
                    try:
                        x_ul, _ = next(unlabeled_loader)
                        x_ul = x_ul.to(device)
                    except StopIteration:
                        unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                         batch_size=args.train_batch,
                                                         shuffle=True,
                                                         num_workers=args.num_workers))
                        break
                    #  get pseudo label predictions
                    with torch.no_grad():
                        model.eval()
                        unlabeled_output = model.forward(x_ul)
                        probs = softmax(unlabeled_output)
                        max = torch.max(probs, dim=1)
                        max_values = max.values
                        max_indices = max.indices
                        mask = max_values > threshold
                        pseudo_pic_set = torch.cat((pseudo_pic_set, x_ul[mask].to("cpu")), 0)
                        pseudo_lab_set = torch.cat((pseudo_lab_set, max_indices[mask].to("cpu")), 0)
                        n_pl_used += sum(mask)

                #add to pseudoset
                pseudo_pic_set = torch.split(pseudo_pic_set, args.train_batch)
                pseudo_lab_set = torch.split(pseudo_lab_set, args.train_batch)

                pseudo_set = []

                for index in range(len(pseudo_lab_set)):
                    pseudo_set.append([pseudo_pic_set[index], pseudo_lab_set[index]])
                pseudo_iter = iter(pseudo_set)




        # epoch tests (aka. get the error rate and val loss)
        with torch.no_grad():
            epochloss = epochloss / (n_iters + len(pseudo_set))


            sum_acc = 0
            n = 0
            m = 0
            for item in test_loader:
                n += 1
                model.eval()
                x_t, y_t = item
                x_t, y_t    = x_t.to(device), y_t.to(device)
                out_test = model(x_t)
                testloss += criterion(out_test, y_t)
                sum_acc += (utils.accuracy(out_test, y_t))[0]
            for item in valset:
                m += 1
                x_v, y_v = item
                x_v, y_v = x_v.to(device), y_v.to(device)
                out_val = model(x_v)
                val_loss += criterion(out_val, y_v)


        val_loss = val_loss / m
        err_rate = sum_acc/n
        real_err = 1 - err_rate/100


        # printouts and tensorboard visuals
        if not args.base and alpha > 0:
            writer.add_scalar("Misc/Influence of Pseudo Loss", alpha, epoch + 1)
            writer.add_scalar("Misc/Pseudo labels used", n_pl_used, epoch + 1)
        writer.add_scalar("Misc/Error rate", real_err, epoch + 1)
        writer.add_scalars('Loss', {'Loss/train': epochloss, 'Loss/test': testloss / n, 'Loss/validation': val_loss}, epoch+1)

        err_rate = err_rate.cpu()
        print("Epoch:", epoch+1, "/", args.epoch)
        print("Accuracy:", err_rate.numpy())
        if not args.base:
            print("Train Loss:", epochloss)
            if alpha > 0:
                print(f"pseudolabel used: {n_pl_used}")

        end = time.perf_counter()

        print("time needed: ", round(end-begin, 3), "sek")
        print("#################")
        errors.append(err_rate)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_val_model = deepcopy(model.state_dict())

        if err_rate > best_err:
            best_err = err_rate
            best_err_model = deepcopy(model.state_dict())
            n_no_progress = 0
        else:
            n_no_progress += 1

        if n_no_progress == stopping:
            print(f"Early stopping, since no accuracy progress was made for {stopping} epochs.")
            break

    # after training cleaning, summary and save model
    writer.close()
    torch.save({'model_state_dict': best_val_model,
                'width': args.model_width,
                'depth': args.model_depth,
                'dropout': args.dropout}, f"{args.dataout}{args.dataset}/{args.num_labeled}/{model_name}_val.pt")
    torch.save({'model_state_dict': best_err_model,
                'width': args.model_width,
                'depth': args.model_depth,
                'dropout': args.dropout}, f"{args.dataout}{args.dataset}/{args.num_labeled}/{model_name}_err.pt")



    print("Best Acc:", best_err)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10",
                        type=str, choices=["cifar100", "cifar100"])
    parser.add_argument("--datapath", default="./data/",
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float,
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.4, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.001, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='test batchsize')
    parser.add_argument('--epoch', default=120, type=int,
                        help='total number of epochs to run')
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=8,
                        help="model width for wide resnet")
    parser.add_argument('--base', type=bool, default=False,
                        help='base model without pseudo labels')
    parser.add_argument("--name", type=str, default="testmodel",
                        help='Name of the model')
    parser.add_argument("--stopping", type=int, default=40,
                        help="Early stopping criterion")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="learning rate scheduler decrease")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate for wrn")
    parser.add_argument("--optim", default="SGD",
                        type=str, choices=["SGD", "adam"])


    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    begin_total = time.perf_counter()

    args = parser.parse_args()


    main(args)



    end_total = time.perf_counter()
    print("Total time: ", round(end_total - begin_total, 3), "sek")
