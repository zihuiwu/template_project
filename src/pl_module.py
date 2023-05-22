import torch
from pytorch_lightning import LightningModule


class PytorchLightningModule(LightningModule):
    def __init__(self, cfg, model, loss_fn, metric_fns):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metric_fns'])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, _, label = batch
        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        return {
            'loss': loss,
            'pred': pred
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # log
        _, _, label = batch
        train_loss, pred = outputs['loss'], outputs['pred']
        accuracy_dict = {
            f'train_{self.loss_fn.name}': train_loss.item()
        }
        for metric_fn in self.metric_fns:
            accuracy_dict[f'train_{metric_fn.name}'] = metric_fn(pred, label).item()
        self.log_dict(accuracy_dict, on_step=False, on_epoch=True) # change the default on_step, on_epoch options for "on_train_batch_end" hook

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        image, _, label = batch
        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        return loss, pred

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log
        _, _, label = batch
        val_loss, pred = outputs
        accuracy_dict = {
            f'val_{self.loss_fn.name}': val_loss.item()
        }
        for metric_fn in self.metric_fns:
            accuracy_dict[f'val_{metric_fn.name}'] = metric_fn(pred, label).item()
        self.log_dict(accuracy_dict)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        image, _, label = batch
        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        return loss, pred

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log
        _, _, label = batch
        test_loss, pred = outputs
        accuracy_dict = {
            f'test_{self.loss_fn.name}': test_loss.item()
        }
        for metric_fn in self.metric_fns:
            accuracy_dict[f'test_{metric_fn.name}'] = metric_fn(pred, label).item()
        self.log_dict(accuracy_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **dict(self.cfg.experiment.optimizer_adam))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : scheduler,
                "monitor" : "val_" + self.loss_fn.name
            }
        }