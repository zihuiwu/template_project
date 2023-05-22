import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from .callbacks import ValidationCallback, TestCallback

class Experiment:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, pl_module, data_module):
        # checkpoint saving config
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.exp_dir,
            filename=f'{{epoch}}-{{val_{pl_module.metric_fns[0].name}:.2f}}',
            monitor=f'val_{pl_module.metric_fns[0].name}',
            mode=pl_module.metric_fns[0].mode, 
            verbose=False,
        )
        early_stop_callback = EarlyStopping(
            monitor=f'val_{pl_module.metric_fns[0].name}',
            patience=self.cfg.experiment.early_stopping_patience, 
            mode=pl_module.metric_fns[0].mode, 
            verbose=False
        )

        # train model
        trainer = pl.Trainer(
            accelerator='auto', 
            devices=1, 
            logger=WandbLogger(**dict(self.cfg.logger)),
            callbacks=[
                checkpoint_callback, 
                early_stop_callback,
                ValidationCallback(self.cfg.exp_dir, self.cfg.val_epoch_vis_freq),
                TestCallback(self.cfg.exp_dir, self.cfg.test_batch_vis_freq)
            ],
            **dict(self.cfg.experiment.trainer),
        )

        # train/val model
        trainer.fit(pl_module, data_module)

        # test model
        trainer.test(pl_module, data_module)
