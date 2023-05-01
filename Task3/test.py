import torch
from model.wrn import WideResNet
from torch.nn.functional import softmax

def test_cifar10(testdataset, filepath = "./output/cifar10/4000/cifar10_4000lbl_baseline_err.pt"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file = torch.load(filepath)
    model_depth = 28
    num_classes = 10
    model_width = 2
    dropout = 0
    state_dict = file

    model = WideResNet(model_depth, num_classes, widen_factor=model_width, dropRate=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    with torch.no_grad():
        model.eval()
        result = []
        for item in testdataset:
            img, target = item
            img = img.to(device)
            img_batched = img[None, :]
            img_pred = model(img_batched)
            logit = softmax(img_pred)[0]
            logit = logit.to('cpu')
            result.append(logit.numpy())

    result = torch.tensor(result)
    return result



def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args:
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape
                [num_samples, 100]. Apply softmax to the logits

    Description:
        This function loads the model given in the filepath and returns the
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc)
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # TODO: SUPPLY the code for this function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file = torch.load(filepath)
    model_depth = 28
    num_classes = 100
    model_width = 2
    dropout = 0
    state_dict = file

    model = WideResNet(model_depth, num_classes, widen_factor=model_width, dropRate=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    with torch.no_grad():
        model.eval()
        result = []
        for item in testdataset:
            img, target = item
            img = img.to(device)
            img_batched = img[None, :]
            img_pred = model(img_batched)
            logit = softmax(img_pred)[0]
            logit = logit.to('cpu')
            result.append(logit.numpy())

    result = torch.tensor(result)
    return result







