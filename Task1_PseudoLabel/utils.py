import torch

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def n_alpha(epoch, max_epochs):
    t_1 = max_epochs / 2
    t_2 = max_epochs * 4 / 5
    if epoch <= t_1:
        alpha = 0
    elif t_1 < epoch <= t_2:
        alpha = 3 * (epoch - t_1) / (t_2 - t_1)
    elif epoch > t_2:
        alpha = 3
    return alpha