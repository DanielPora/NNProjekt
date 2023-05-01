import torch
import torch.nn as nn
import torchvision.utils
from torch.nn import functional as F


class VATLoss(nn.Module):

    def __init__(self, args, writer, visualize, epoch, dataset):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        self.writer = writer
        self.visualize = visualize
        self.epoch = epoch
        self.dataset = dataset

    def unnormalize(self, minibatch):
        if self.dataset == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
        elif self.dataset == "cifar100":
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        std = torch.tensor(std).view(-1, 1, 1).to(minibatch.device)
        mean = torch.tensor(mean).view(-1, 1, 1).to(minibatch.device)
        return minibatch * std + mean

    def normalize(self, d):
        # Roll out the minibatch images into vectors
        batch_size = d.shape[0]
        image_vectors = d.view(batch_size, -1, *(1 for _ in range(d.dim() - 2)))
        # Divide the minibatch per image by its L2 Norm in order to normalize
        d /= torch.norm(image_vectors, dim=1, keepdim=True) + self.eps  # Avoid division by zero
        return d

    def forward(self, model, x):
        """
            Computes the VATLoss for model and x (=minibatch)
            :param model: Our neural network
            :param x: The minibatch (combined x_l and x_ul)

            :return: Loss which maximizes the KL Divergence of actual predictions and adversarial predicitons
        """
        # Sample gaussian in same shape and device as the minibatch
        r = torch.randn_like(x)
        r = self.normalize(r)
        predictions = model.forward(x)
        predictions = F.softmax(predictions, dim=1)
        for _ in range(self.vat_iter):
            r.requires_grad_()
            # Adds them together, removes ambiguity of broadcasting
            adv_examples = x.add(r, alpha=self.xi)
            adv_predictions = model.forward(adv_examples)
            # KL need log probabilities => log(softmax(predictions))
            adv_predictions = F.log_softmax(adv_predictions, dim=1)
            # Reduction set to batchmean to match mathematical definition of kl_div (see documentation)
            adv_distance = F.kl_div(adv_predictions, predictions, reduction="batchmean")
            # Compute gradient wrt r, set it as new r and normalize it
            adv_distance.backward(retain_graph=True)
            r = r.grad
            r = self.normalize(r)
            model.zero_grad()
        # Scale d and compute final kl divergence
        d = r
        r_adv = d * self.eps
        adv_exp = x.add(r_adv)

        # Visualize the adversarial examples in a grid
        if self.visualize:
            img_grid = torchvision.utils.make_grid(self.unnormalize(adv_exp))
            self.writer.add_image(f"Adversarial Examples/ Epoch: {self.epoch}", img_grid)
            img_grid = torchvision.utils.make_grid(self.unnormalize(x))
            self.writer.add_image(f"Actual Image/ Epoch: {self.epoch}", img_grid)
        adv_predictions = model.forward(adv_exp)
        adv_predictions = F.log_softmax(adv_predictions, dim=1)
        loss = F.kl_div(adv_predictions, predictions, reduction="batchmean")
        return loss

