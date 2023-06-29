# Assignment Question

1 Write a new network that has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)

2 total RF must be more than 44

3 one of the layers must use Depthwise Separable Convolution

4 one of the layers must use Dilated Convolution

5 use GAP (compulsory):- add FC after GAP to target #of classes (optional)

6 use albumentation library and apply:
horizontal flip
shiftScaleRotate
coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

7 achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
## Description
Model has used around 198,970 params, i have used dilation twice instead of maxpool, apaart from that i have used strided convolution once instead of maxpool just to illustrate. Model is able to achieve 85.54 around 195-200 epoch and 86.8% accuracy around 295-300 epoch.
Scheduler strategy is cyclic learning rate with triangular2 mode. Following is link to notebook
![](https://github.com/gdeotale/ERA/blob/main/Session9/S9.ipynb)

## Following are sample Cifar training images with augmentation
![trainimage](https://github.com/gdeotale/ERA/assets/8176219/c8f484ba-786e-4938-badb-6cc319e5d526)
## Model Params
![Params](https://github.com/gdeotale/ERA/assets/8176219/b7a58e89-fd1d-4515-86b6-bf300bae1081)
## Following is snippet of final few epochs
![Capture](https://github.com/gdeotale/ERA/assets/8176219/d5d6b7a1-de16-4e3f-90c2-3b9d0e126345)
## Following are train and val curves along with  learning rate
![curves](https://github.com/gdeotale/ERA/assets/8176219/c7912009-cc7d-461c-96e4-50805e2654b2)
## Misclassified images
![misclassified](https://github.com/gdeotale/ERA/assets/8176219/7622a615-1cde-492e-b3d3-f42acdaab555)
## Classified images
![classified](https://github.com/gdeotale/ERA/assets/8176219/bf554580-cd66-4d39-b42b-79b5ea620fa9)
