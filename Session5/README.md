Following contains code from parts of assignment5

!model.py contains model architecture that is used to train model.(Please note that this is very generic architecture kept very simple for inital assignments)

!utils.py contains utility functions for plotting images that may be used as and when needed.

!Session_5.ipynb contains funtions from loading dataset to loading train/test loader, also for calling Model.py and utils.py, also to run main training and evaluation loop

We have trained model  on MNIST Dataset. The results are very good and we are able to achieve 99.52% accuracy on test data set within 20 epochs.
Model has used around Total params: 593,200

Follwing is detailed summary of Model
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------

Following are accuracy numbers for final few epochs
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.0031 Batch_id=117 Accuracy=99.15: 100%|██████████| 118/118 [00:23<00:00,  5.03it/s]
Test set: Average loss: 0.0000, Accuracy: 59728/60000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0158 Batch_id=117 Accuracy=99.17: 100%|██████████| 118/118 [00:23<00:00,  4.95it/s]
Test set: Average loss: 0.0000, Accuracy: 59732/60000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0582 Batch_id=117 Accuracy=99.15: 100%|██████████| 118/118 [00:24<00:00,  4.86it/s]
Test set: Average loss: 0.0000, Accuracy: 59747/60000 (99.58%)


Following is Train/Test accuracy and loss plot
![Plots](https://github.com/gdeotale/ERA/assets/8176219/81788eb1-ea56-442a-946f-41b682388b47)
