For the cifar 10 we used the following arguments:

optim=adam, lr=0.001, wd=0.001, dropout=0.2, gamma=0.99, epoch=160

for cifar100 we used:
optim=sgd, lr=0.03, wd=0.001, dropout=0.5, gamma=0.99, epoch=120

with an additional argument
base=[True, False], for training the baseline model without pseudolabel

The internal naming convention will save the baseline model in a way, that it's
comprehensible, which values for the arguments were used to create the baseline model.


threshold=[0.95, 0.75, 0.6] respectively for the desired threshold


note: for each model two checkpoints are saved, one with the best validation loss and one with
the highest accuracy on the test set
all training is done with early stopping on accuracy, if no better accuracy was achieved for 35 epochs
we decided to only hand in the models with highest accuracy 
