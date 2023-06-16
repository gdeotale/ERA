# Code Block1:
## Code 1:
### 1. Target: Getting basic skelton right
### 2. Analysis: here i haven't kept any param count while building model, no batchnorm or augmentation or lr
### 3. Result:
Total params: 2,147,472
Despite all the non usage of params model is able to touch 99.06% test accuracy. following are results of 19th epoch
Train: Loss=0.0383 Batch_id=468 Accuracy=99.76: 100%|██████████| 469/469 [00:22<00:00, 20.90it/s]
Test set: Average loss: 0.0003, Accuracy: 9906/10000 (99.06%)
### 4. File Link: https://github.com/gdeotale/ERA/blob/main/Session7/Basic_skeleton/Basic_skeleton.ipynb

## Code 2:
### 1. Target: stripping code to fit in 8k params
### 2. Analysis: I have kept kernels of size 8,16,12 to restrict size of params. Apart from that maxpool has been used twice.
As expected with decreased number of parametes the accuracy of model does go for toss able to achive 98% validation accuracy in this part of code.
Model does seem to overfit here.
### 3. Result: Total params: 7,720
Train: Loss=0.0269 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:20<00:00, 23.44it/s]
Test set: Average loss: 0.0004, Accuracy: 9828/10000 (98.28%)
### 4. File Link: https://github.com/gdeotale/ERA/blob/main/Session7/Stripped_model/Stripped_model.ipynb

# Code Block2:
## Code 3:
### 1. Target: Improvise on accuracy with stripped model, one sure shot way is to speedup training is adding batch norm in training
### 2. Analysis: Batch norm has been added after every convolution except for last FC layer. It does improve accuracy from 98.28% in previous to 99.20. However it results in overfitting. Also target accuracy has not been met.
### 3. Result:
Total params: 7,888 
Train: Loss=0.0634 Batch_id=468 Accuracy=99.57: 100%|██████████| 469/469 [00:19<00:00, 24.50it/s]
Test set: Average loss: 0.0002, Accuracy: 9920/10000 (99.20%)
### 4. File Link:
https://github.com/gdeotale/ERA/blob/main/Session7/Batch_norm/Batch_norm.ipynb

## Code 4:
### 1. Target: Reduce overfitting by adding dropout and augmentation
### 2. Analysis: We have added different dropout percentage and different augmetnation strategies. Overfitting has been reduced to considerable level and train and val accuracy has been brought in acceptable limit. However the result is one time achievement and that too at 19th epoch
### 3. Results: 0.01 19 Train: Loss=0.0223 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:28<00:00, 16.50it/s] Test set: Average loss: 0.0002, Accuracy: 9923/10000 (99.23%)
### 4. File link: https://github.com/gdeotale/ERA/blob/main/Session7/Avoiding_overfitting/Basic_skeleton.ipynb

# Code Block3:
## Code 5:
### 1. Target : Try to achieve best accuracy multiple times under 15 epochs
### 2. Analysis: Achieved top acuracy of 99.27% in 12 epoch and near 99.22% multiple times after adjusting lr scheduler and batch_size :32
### 3. Result: 
Train: Loss=0.1211 Batch_id=1874 Accuracy=98.86: 100%|██████████| 1875/1875 [00:41<00:00, 45.32it/s]
Test set: Average loss: 0.0007, Accuracy: 9927/10000 (99.27%)
### 4. File Link: https://github.com/gdeotale/ERA/blob/main/Session7/Learning%20Rate/learningrate.ipynb
