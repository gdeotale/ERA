Following contains code from parts of assignment5

## model.py 
contains model architecture that is used to train model.(Please note that this is very generic architecture kept very simple for inital assignments)

## utils.py 
contains utility functions for plotting images that may be used as and when needed.

## Session_5.ipynb 
contains funtions from loading dataset to loading train/test loader, also for calling Model.py and utils.py, also to run main training and evaluation loop

We have trained model  on MNIST Dataset. The results are very good and we are able to achieve 99.52% accuracy on test data set within 20 epochs.
Model has used around Total params: 593,200

## Follwing is detailed summary of Model
![Capture](https://github.com/gdeotale/ERA/assets/8176219/e2dca18d-b6ba-4bd4-83dd-0de99fb53b23)

## Following are accuracy numbers for final few epochs
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.0397 Batch_id=117 Accuracy=99.06: 100%|██████████| 118/118 [00:22<00:00,  5.24it/s]
Test set: Average loss: 0.0000, Accuracy: 59692/60000 (99.49%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0254 Batch_id=117 Accuracy=99.12: 100%|██████████| 118/118 [00:22<00:00,  5.16it/s]
Test set: Average loss: 0.0000, Accuracy: 59711/60000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0062 Batch_id=117 Accuracy=99.12: 100%|██████████| 118/118 [00:22<00:00,  5.14it/s]
Test set: Average loss: 0.0000, Accuracy: 59701/60000 (99.50%)


## Following is Train/Test accuracy and loss plot
![Plots](https://github.com/gdeotale/ERA/assets/8176219/81788eb1-ea56-442a-946f-41b682388b47)
