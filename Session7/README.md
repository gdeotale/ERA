Code Block1:
Code 1:
Target: getting basic skelton right
Analysis: here i haven't kept any param count while building model, no batchnorm or augmentation or lr
Result:
Total params: 2,147,472
Despite all the non usage of params model is able to touch 99.06% test accuracy. following are results of 19th epoch
lr = 0.01 19
Train: Loss=0.0383 Batch_id=468 Accuracy=99.76: 100%|██████████| 469/469 [00:22<00:00, 20.90it/s]
Test set: Average loss: 0.0003, Accuracy: 9906/10000 (99.06%)

Code 2:
Target: stripping code to fit in 8k params
Analysis: i have kept kernels of size 8,16,12 to restrict size of params. Apart from that maxpool has been used twice.
As expected with decreased number of parametes the accuracy of model does go for toss able to achive 98% validation accuracy in this part of code.
Model does seem to overfit here.
Result:
Total params: 7,720
0.01 19
Train: Loss=0.0269 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:20<00:00, 23.44it/s]
Test set: Average loss: 0.0004, Accuracy: 9828/10000 (98.28%)


Code Block2:
Code 3:
Target: Improvise on accuracy with stripped model, one sure shot way is to speedup training is adding batch norm in training
Analysis: Batch norm has been added after every convolution except for last FC layer.
It does improve accuracy from 98.28% in previous to 99.20. However it results in overfitting. Also target accuracy has not been met.
Result:
Total params: 7,888
0.01 18
Train: Loss=0.0634 Batch_id=468 Accuracy=99.57: 100%|██████████| 469/469 [00:19<00:00, 24.50it/s]
Test set: Average loss: 0.0002, Accuracy: 9920/10000 (99.20%)

Code 4:
Target: Reduce overfitting by adding dropout and augmentation
Analysis: We have added different dropout percentage and different augmetnation strategies
Overfitting has been reduced to considerable level and train and val accuracy has been brought in acceptable limit.
However the result is one time achievement and that too at 19th epoch
Results:
0.01 19
Train: Loss=0.0223 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:28<00:00, 16.50it/s]
Test set: Average loss: 0.0002, Accuracy: 9923/10000 (99.23%)

Code Block3:
Code 5:
Target : Try to achieve best accuracy multiple times under 15 epochs
Analysis:
Achieved top acuracy of 99.27% in 12 epoch and near 99.22% multiple times after adjusting lr scheduler and batch_size
Result:
0.0025 12
Train: Loss=0.1211 Batch_id=1874 Accuracy=98.86: 100%|██████████| 1875/1875 [00:41<00:00, 45.32it/s]
Test set: Average loss: 0.0007, Accuracy: 9927/10000 (99.27%)

0.0025 13
Train: Loss=0.0520 Batch_id=1874 Accuracy=98.95: 100%|██████████| 1875/1875 [00:43<00:00, 43.36it/s]
Test set: Average loss: 0.0007, Accuracy: 9924/10000 (99.24%)

0.000625 14
Train: Loss=0.0012 Batch_id=1874 Accuracy=98.92: 100%|██████████| 1875/1875 [00:44<00:00, 42.40it/s]
Test set: Average loss: 0.0007, Accuracy: 9922/10000 (99.22%)

0.000625 15
Train: Loss=0.0011 Batch_id=1874 Accuracy=98.92: 100%|██████████| 1875/1875 [00:42<00:00, 43.80it/s]
Test set: Average loss: 0.0008, Accuracy: 9922/10000 (99.22%)