#Usage of test.py:

To test a model you must call the test_cifar10 or test_cifar100 function in test.py with the path to the model you 
want to test (Directory of models is final_models). 

#Reproduction of results:

You must run the main.py with the corresponding hyperparameters (learning rate, weight decay, momentum, epsilon, etc.) found in the report. You must also create a directory
where your models are saved, this must be "final_models/test/**dataset**/**number of labels**/".
To deactivate our method you must set --omega to 0, this will then only use the FM Loss.


The model names are the omega and mu values used for training.