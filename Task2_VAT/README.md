#Usage of test.py:

To test a model you must call the test_cifar10 or test_cifar100 function in test.py with the path to the model you 
want to test (Directory of models is final_models). We have two models per dataset/label combination. One for the base 
model (base.pt), without VAT. And one for the VAT model (vat.pt)

#Reproduction of results:

You must run the main.py with the corresponding hyperparameters (learning rate, weight decay, momentum, epsilon, etc.) found in the report. You must also create a directory
where your models are saved, this must be "final_models/test/**dataset**/**number of labels**/".
To deactivate VAT you must set --base True.
Set --optim to "SGD" if you want to use Stochastic Gradient Descent as optimizer, else it will be Adam.