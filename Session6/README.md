# Part1
## Following are parts of forward prop and backwardprop computation created on model shown below
![nnimage](https://github.com/gdeotale/ERA/assets/8176219/d880875c-0f58-46af-8649-02383b59978b)

## Breaking computations for above in 6 steps
![computation1](https://github.com/gdeotale/ERA/assets/8176219/8901543b-e020-48dc-9722-2df50f8c2d74)
![computation23](https://github.com/gdeotale/ERA/assets/8176219/6f8cb9d4-06c3-48bb-b02c-e32cce166008)
![computation45](https://github.com/gdeotale/ERA/assets/8176219/c8b8c449-4a64-4116-8a6d-93b07d12f5f0)
![computation6](https://github.com/gdeotale/ERA/assets/8176219/9f5e0163-e081-4354-b21c-0ee499779fc7)

## Following is snippet of computations carried out in attached Backpropogation.csv
Please refer to https://github.com/gdeotale/ERA/blob/main/Session6/BackPropagation.csv for detailed analysis
![computation7](https://github.com/gdeotale/ERA/assets/8176219/feb63f9a-182f-4fbf-97aa-2b73d8b489a5)

## Following are loss curves obtained for different values of lr
Consider name of Image as lr value
![Capture](https://github.com/gdeotale/ERA/assets/8176219/3165d1c0-cdcc-40e1-9f02-d5a3c59c3e06)

# Part2
## Following contains code from parts of assignment5

## model.py 
contains model architecture that is used to train model.(Please note that this is very generic architecture kept very simple for inital assignments)

## utils.py 
contains utility functions for plotting images that may be used as and when needed. Also training and evaluations definition code have been added here. However i believe there should be separate file for this code.

## Session_6.ipynb 
contains funtions from loading dataset to loading train/test loader, also for calling Model.py and utils.py, also to run main training and evaluation loop

We have trained model  on MNIST Dataset. The results are very good and we are able to achieve 99.55% accuracy on test data set within 20 epochs.
Model has used around Total params: 17,130

## Follwing is detailed summary of Model
![Model](https://github.com/gdeotale/ERA/assets/8176219/17eaaafa-9d07-48d4-b334-c0dd7777acbc)

## Following are accuracy numbers for final few epochs
![train_iter](https://github.com/gdeotale/ERA/assets/8176219/483bafb5-1a99-49a5-9f49-8442825bf6e2)


## Following is Train/Test accuracy and loss plot
![Model](https://github.com/gdeotale/ERA/assets/8176219/f8d74223-a12d-4aff-8d78-0c2d6fa484ac)


