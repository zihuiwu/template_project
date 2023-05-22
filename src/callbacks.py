import torch, os
from sklearn.metrics import confusion_matrix
from pytorch_lightning.callbacks import Callback

from .utils import plot_confusion_matrix
from .data.fashion_mnist import FashionMNISTData


class ValidationCallback(Callback):
    def __init__(self, exp_dir, vis_freq):
        self.vis_freq = vis_freq
        self.save_dir = f'{exp_dir}/val_visualizations'
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if pl_module.current_epoch % self.vis_freq == 0:
            self.preds = []
            self.labels = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.current_epoch % self.vis_freq == 0:
            _, _, label = batch
            _, pred = outputs
            self.preds.append(pred.detach().cpu())
            self.labels.append(label.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.vis_freq == 0:
            plot_confusion_matrix(
                cm=confusion_matrix(y_true=torch.cat(self.labels, dim=0), y_pred=torch.cat(self.preds, dim=0).argmax(1)), 
                target_names=FashionMNISTData.class_names.values(), 
                fname=f'{self.save_dir}/epoch={pl_module.current_epoch}.png',
                normalize=True
            )


class TestCallback(Callback):
    def __init__(self, exp_dir, vis_freq) -> None:
        super().__init__()
        self.vis_freq = vis_freq
        self.save_dir = f'{exp_dir}/val_visualizations'
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass
