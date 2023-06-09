Following contains code from parts of assignment5

## model.py 
contains model architecture that is used to train model.(Please note that this is very generic architecture kept very simple for inital assignments)

## utils.py 
contains utility functions for plotting images that may be used as and when needed. Also training and evaluations definition code have been added here. However i believe there should be separate file for this code.

## Session_6.ipynb 
contains funtions from loading dataset to loading train/test loader, also for calling Model.py and utils.py, also to run main training and evaluation loop

We have trained model  on MNIST Dataset. The results are very good and we are able to achieve 99.55% accuracy on test data set within 20 epochs.
Model has used around Total params: 17,130

## Follwing is detailed summary of Model
![Capture](https://github.com/gdeotale/ERA/assets/8176219/e2dca18d-b6ba-4bd4-83dd-0de99fb53b23)

## Following are accuracy numbers for final few epochs
0.001 17
Train: Loss=0.0113 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:21<00:00, 21.58it/s]
Test set: Average loss: 0.0001, Accuracy: 9947/10000 (99.47%)

0.001 18
Train: Loss=0.0059 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:22<00:00, 21.12it/s]
Test set: Average loss: 0.0002, Accuracy: 9948/10000 (99.48%)

0.001 19
Train: Loss=0.0731 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:21<00:00, 21.38it/s]
Test set: Average loss: 0.0001, Accuracy: 9955/10000 (99.55%)


## Following is Train/Test accuracy and loss plot
![Plots](https://github.com/gdeotale/ERA/assets/8176219/81788eb1-ea56-442a-946f-41b682388b47)
