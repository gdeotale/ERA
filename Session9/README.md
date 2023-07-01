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
Model has used around 198,970 params, i have used dilation twice instead of maxpool, apaart from that i have used strided convolution once instead of maxpool just to illustrate. Model is able to achieve 86.26 around 195-200 epoch and 87.8% accuracy around 295-300 epoch.
Scheduler strategy is cyclic learning rate with triangular2 mode. Following is link to notebook
[Notebook](https://github.com/gdeotale/ERA/blob/main/Session9/S9.ipynb)

## Following are sample Cifar training images with augmentation
![trainimage](https://github.com/gdeotale/ERA/assets/8176219/e40ac5ea-6c7f-49a1-9aab-9f3f51eaec2d)

## Model Params
Following is link to model
[Model](https://github.com/gdeotale/ERA/blob/main/Session9/Net.py)
![Params](https://github.com/gdeotale/ERA/assets/8176219/b7a58e89-fd1d-4515-86b6-bf300bae1081)

## Following is snippet of final few epochs
![Capture0](https://github.com/gdeotale/ERA/assets/8176219/9840dfa4-a430-4433-81ae-ac480b70e1b6)
![Capture](https://github.com/gdeotale/ERA/assets/8176219/1878f19c-ac2a-4f71-8c5e-01bdbd9053d2)

## Following are train and val curves along with  learning rate
![curves](https://github.com/gdeotale/ERA/assets/8176219/2feaeb23-e9c9-4a64-84fa-9523bbcbcea9)
## Misclassified images
![misclassified](https://github.com/gdeotale/ERA/assets/8176219/7622a615-1cde-492e-b3d3-f42acdaab555)
## Classified images
![classified](https://github.com/gdeotale/ERA/assets/8176219/bf554580-cd66-4d39-b42b-79b5ea620fa9)
