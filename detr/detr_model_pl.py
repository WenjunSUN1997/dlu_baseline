import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
import data_class_detr
from pytorch_lightning import Trainer

train_dataloader, val_dataloader= data_class_detr.get_dataloader()


class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        self.id2label = { 0:'pic', 1:'caption', 2:'paragraph', 3:'heading'}
        self.config = DetrConfig(use_pretrained_backbone = False,
                                 num_channels=3,
                                 num_queries=100,
                                 id2label=self.id2label,
                                 num_labels=len(self.id2label))
        self.model = DetrForObjectDetection(self.config)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        with open('train_log.txt', 'a') as file:
            file.write(str(batch_idx) + '\t' + str(loss.item()) + '\n')

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

model = Detr(lr=0.001, lr_backbone=1e-5, weight_decay=1e-4)
trainer = Trainer(gpus=1, max_steps=100, gradient_clip_val=0.1)
trainer.fit(model)