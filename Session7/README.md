# Code Block1:
## Code 1:
###1. Target: Getting basic skelton right
###2. Analysis: here i haven't kept any param count while building model, no batchnorm or augmentation or lr
###3. Result:
Total params: 2,147,472
Despite all the non usage of params model is able to touch 99.06% test accuracy. following are results of 19th epoch
Train: Loss=0.0383 Batch_id=468 Accuracy=99.76: 100%|██████████| 469/469 [00:22<00:00, 20.90it/s]
Test set: Average loss: 0.0003, Accuracy: 9906/10000 (99.06%)
###4. File Link: https://github.com/gdeotale/ERA/blob/main/Session7/Basic_skeleton/Basic_skeleton.ipynb

## Code 2:
###1. Target: stripping code to fit in 8k params
###2. Analysis: I have kept kernels of size 8,16,12 to restrict size of params. Apart from that maxpool has been used twice.
As expected with decreased number of parametes the accuracy of model does go for toss able to achive 98% validation accuracy in this part of code.
Model does seem to overfit here.
###3. Result: Total params: 7,720
Train: Loss=0.0269 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:20<00:00, 23.44it/s]
Test set: Average loss: 0.0004, Accuracy: 9828/10000 (98.28%)
###4. File Link: https://github.com/gdeotale/ERA/blob/main/Session7/Stripped_model/Stripped_model.ipynb
