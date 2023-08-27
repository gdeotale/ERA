from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
import warnings
warnings.filterwarnings("ignore")
from dataset import YOLODataset
from loss import YoloLoss
import config
from Model.model import YOLOv3
from torchmetrics import Accuracy
import torch

class Yolov3_module(LightningModule):
    def __init__(self, data_dir='PASCAL_VOC', hidden_size=16, learning_rate=2e-4):

        super().__init__()

        # Hardcode some dataset specific attributes
        self.num_classes = 20
        self.train_transform = config.train_transforms
        self.test_transform = config.test_transforms
        self.epoch = self.current_epoch

        # Define PyTorch model
        self.model = YOLOv3()
        self.loss = YoloLoss()
        self.accuracy = Accuracy('multiclass', num_classes=20)
        self.scaled_anchors = (
          torch.tensor(config.ANCHORS)
           * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        )


    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )
        with torch.cuda.amp.autocast():
          out = self(x)
          loss = (
                self.loss(out[0], y0, self.scaled_anchors[0])
                + self.loss(out[1], y1, self.scaled_anchors[1])
                + self.loss(out[2], y2, self.scaled_anchors[2])
            )


        return loss

    def validation_step(self, batch, batch_idx):
     x, y = batch
     y0, y1, y2 = y[0], y[1], y[2]

     with torch.cuda.amp.autocast():
        out = self(x)
        loss = (
            self.loss(out[0], y0, self.scaled_anchors[0])
            + self.loss(out[1], y1, self.scaled_anchors[1])
            + self.loss(out[2], y2, self.scaled_anchors[2])
        )

     self.log("val_loss", loss, prog_bar=True)
     garbage_collection_cuda()
     return loss

    def on_validation_epoch_end(self):
        plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, self.scaled_anchors)
        # Get the learning rate from the optimizer
        optimizer = self.optimizers()
        lr_file = open('lr_file.txt','a')
        current_learning_rate = optimizer.param_groups[0]['lr']
        lr_file.write(str(current_learning_rate)+',')
        lr_file.close()
        print("Current learning rate: "+str(current_learning_rate))
        epoch = self.current_epoch
        print(f"Currently epoch {epoch}")
        print("On Train Eval loader:")
        print("On Train loader:")
        check_class_accuracy(self.model, self.train_dataloader(), threshold=config.CONF_THRESHOLD)
        epoch = self.current_epoch
        if config.SAVE_MODEL:
          save_checkpoint(model, optimizer, filename="checkpoints/"+str(epoch)+f"checkpoint.pth.tar")

        if epoch > 0 and epoch % 5 == 0:
            print("~~~~~~~~~Evaluating for val set~~~~~~~~~~~~~~~")
            check_class_accuracy(self.model, self.test_dataloader(), threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_dataloader(),
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
        garbage_collection_cuda()


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        EPOCHS = config.NUM_EPOCHS // 5
        #lr = [1e-05,0.00014749999999999998,0.000285,0.00042249999999999997,0.00056,0.0006974999999999999,0.0008349999999999999,0.0009725,0.0009756121951219513,0.0009451274390243903,0.0009146426829268294,0.0008841579268292683,0.0008536731707317073,0.0008231884146341463,0.0007927036585365854,0.0007622189024390244,0.0007317341463414634,0.0007012493902439024,0.0006707646341463415,0.0006402798780487805,0.0006097951219512195,0.0005793103658536585,0.0005488256097560976,0.0005183408536585366,0.00048785609756097563,0.00045737134146341476,0.0004268865853658538,0.0003964018292682927,0.00036591707317073173,0.00033543231707317087,0.0003049475609756099,0.0002744628048780488,0.00024397804878048784,0.00021349329268292687,0.000183008536585366,0.00015252378048780492,0.00012203902439024395,9.155426829268298e-05,6.106951219512211e-05,3.058475609756103e-05,1.0000000000005664e-07]
        scheduler = OneCycleLR(
           optimizer,
           max_lr=1E-3,
           steps_per_epoch=len(self.train_dataloader()),
           total_steps = 41,
           epochs=EPOCHS,
           pct_start=0.2,
           div_factor=100,
           three_phase=False,
           final_div_factor=100,
           anneal_strategy='linear'
        )
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Example StepLR configuration
        # Create the MultiStepLR scheduler with the specified learning rate values
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], gamma=1.0)

        return [optimizer], [scheduler]

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
       self.train_dataset = YOLODataset(
          config.DATASET + "/train.csv",
          transform=self.train_transform,
          S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
          img_dir=config.IMG_DIR,
          label_dir=config.LABEL_DIR,
          anchors=config.ANCHORS,
          )
       self.test_dataset = YOLODataset(
          config.DATASET + "/test.csv",
          transform=self.test_transform,
          S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
          img_dir=config.IMG_DIR,
          label_dir=config.LABEL_DIR,
          anchors=config.ANCHORS,
          )

    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    def val_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    def get_optimizer(self):
      return self.optimizers()