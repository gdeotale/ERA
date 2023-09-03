## Assignment
![Screenshot 2023-09-04 002643](https://github.com/gdeotale/ERA/assets/8176219/0e93990d-ce32-4dc5-a292-a28010d18c00)

## Tensor Board
![training_curve](https://github.com/gdeotale/ERA/assets/8176219/3f24258e-ce95-4e06-b0c7-63c052aec7d0)

## Model
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name    ┃ Type             ┃ Params ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ loss_fn │ CrossEntropyLoss │      0 │
│ 1 │ model   │ Transformer      │ 75.1 M │
└───┴─────────┴──────────────────┴────────┘
Trainable params: 75.1 M                                                                                           
Non-trainable params: 0                                                                                            
Total params: 75.1 M                                                                                               
Total estimated model params size (MB): 300    

## Logs

Epoch 7/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5053/5053 0:22:01 • 0:00:00 4.65it/s loss: 4.1 v_num: 27 val_loss:    
                                                                                  4.559        
                                                                                  
Source:These things being added to my desire of having a good quantity for store, and to secure a constant supply, 
I resolved not to taste any of this crop but to preserve it all for seed against the next season; and in the 
meantime to employ all my study and hours of working to accomplish this great work of providing myself with corn 
and bread.
TARGET:Queste considerazioni aggiunte al mio desiderio d’ingrandire le mie provvigioni e di assicurarmi un costante
vitto per l’avvenire, mi trassero nella risoluzione di lasciare intatto questo secondo ricolto e di serbarlo tutto 
per semenza alla prossima stagione; e d’impiegare intanto l’intero mio studio, le intere ore mie di lavoro alla 
grande impresa di provvedermi così di biade come di pane.
PREDICTED:Questi cose mi un buon buon lavoro per , per un ’ accetta , per essere un ’ altra parte di questa , non 
avendo alcun valore per tutto il più piccolo ricolto ; ma per quanto per me ne i miei compagni , per quanto per me 
ne i miei compagni .

Epoch 8/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5053/5053 0:22:02 • 0:00:00 4.61it/s loss: 3.88 v_num: 27 val_loss:    
                                                                                 4.545  
                                                                                 
Source:One thing that he sincerely confessed was that, living so long in Moscow with nothing but talk and food and 
drink, he was going silly.
TARGET:La cosa che egli confessava più sinceramente era che, vivendo così a lungo a Mosca, passando il tempo 
soltanto a far discorsi, a mangiare e a bere, era divenuto un insensato.
PREDICTED:Un ’ unica cosa che egli voleva dire che , come a Mosca , non voleva parlare di Mosca , ma di nuovo , 
voleva bere e di vino , era semplicemente semplicemente semplicemente .

Epoch 9/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5053/5053 0:21:53 • 0:00:00 4.67it/s loss: 3.64 v_num: 27 val_loss:    
                                                                                 4.555     
                                                                                 
Source:"He would discover many things in you he could not have expected to find?
TARGET:— Egli scoprì presto in voi cose che non aveva sperato di trovare?
PREDICTED:— Non vi sarebbe forse più facile che non avreste trovato ?
`Trainer.fit` stopped: `max_epochs=10` reached.
