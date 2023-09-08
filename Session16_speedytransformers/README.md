![Screenshot 2023-09-08 132627](https://github.com/gdeotale/ERA/assets/8176219/6f3f894c-d390-46da-9e3d-f775f504af09)

## Strategies followed to improvise accuracy and speed

Speed improvement
1. Batch size was increased from 8 to 32 and num_workers=8 were added in dataloader-> this resulted in major speed boost
2. Mixed precision training was implemented using torch.cuda.amp.autocast() this gave further boost up in speed
3. Weight sharing module was added
    e1, e2, e3 = encoder_blocks
    d1, d2, d3 = decoder_blocks
    encoder_blocks1 = [e1, e2, e3, e3, e2, e1]
    decoder_blocks1 = [d1, d2, d3, d3, d2, d1]
4. the size of token was restricted to 150 in main.py

Loss improvement
1. Onecyle policy resulted in major boost in accuracy
 
## Logs till epoch 20
![Screenshot 2023-09-08 131159](https://github.com/gdeotale/ERA/assets/8176219/c88e1a8e-486d-4a2f-ae36-f11a4c545b8b)
## final few epochs
![Screenshot 2023-09-08 131045](https://github.com/gdeotale/ERA/assets/8176219/cd4f2ae4-b9a9-44ca-b95c-de3369a560e3)
## Tensorboard logs
![Screenshot 2023-09-08 122459](https://github.com/gdeotale/ERA/assets/8176219/96d2c384-6313-4493-bd1d-d525a6447d08)
![Screenshot 2023-09-08 122524](https://github.com/gdeotale/ERA/assets/8176219/540997da-691c-472a-ba84-5401321e3cf5)
![Screenshot 2023-09-08 122540](https://github.com/gdeotale/ERA/assets/8176219/abdbf10d-d255-4570-a1f8-af0f26eac3cf)
![Screenshot 2023-09-08 122441](https://github.com/gdeotale/ERA/assets/8176219/11bd784d-1500-4f25-a7b2-05cdf28dcd87)
