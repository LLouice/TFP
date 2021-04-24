import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from config import Config
from data import DataModule
from models.sttn.model import STTransformer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from utils import calc_metrics, get_logger


class TimeCallback(Callback):
    def __init__(self, logger=None):
        self.logger = logger

    def on_epoch_start(self, trainer, pl_module):
        self.tc = time.perf_counter()

    def on_epoch_end(self, trainer, pl_module):
        if self.logger:
            self.logger.debug(
                f"Elapsed Time {time.perf_counter()-self.tc:.4f}")
        self.tc = time.perf_counter()


class Net(pl.LightningModule):
    def __init__(self, hparams, adj, scaler, logger=None):
        super(Net, self).__init__()
        self.hparams = hparams
        args = self.hparams

        self.scaler = scaler
        self._logger = logger

        self.model = STTransformer(adj[0], args.in_dim, args.embed_size,
                                   args.time_num, args.num_layers, args.T_dim,
                                   args.output_T_dim, args.heads, 0,
                                   args.forward_expansion,
                                   dropout=args.dropout,
                                   device=args.device)

        for name, param in self.model.named_parameters():
            if "conv" in name and "weight" in name:
                torch.nn.init.xavier_normal_(param)
            if "bias" in name:
                torch.nn.init.zeros_(param)

    @staticmethod
    def add_model_specific_args(parent_parser):
        _ADJ_CHOICES = [
            'scalap', 'normlap', 'symnadj', 'transition', 'identity'
        ]
        parser = parent_parser.add_argument_group("GWNet")
        parser.add_argument('--device', type=str, default='cuda:0', help='')
        parser.add_argument('--data',
                            type=str,
                            default='data/METR-LA',
                            help='data path')
        parser.add_argument('--adjdata',
                            type=str,
                            default='data/sensor_graph/adj_mx.pkl',
                            help='adj data path')
        parser.add_argument('--adjtype',
                            type=str,
                            default='doubletransition',
                            help='adj type',
                            choices=_ADJ_CHOICES)
        parser.add_argument('--do_graph_conv',
                            action='store_true',
                            help='whether to add graph convolution layer')
        parser.add_argument('--aptonly',
                            action='store_true',
                            help='whether only adaptive adj')
        parser.add_argument('--addaptadj',
                            action='store_true',
                            help='whether add adaptive adj')
        parser.add_argument('--randomadj',
                            action='store_true',
                            help='whether random initialize adaptive adj')
        parser.add_argument('--time_num', type=int, default=288, help='time stamp num')
        parser.add_argument('--T_dim', type=int, default=12, help='')
        parser.add_argument('--output_T_dim', type=int, default=3, help='')
        parser.add_argument('--embed_size',
                            type=int,
                            default=64,
                            help='model hidden dim')
        parser.add_argument('--in_dim',
                            type=int,
                            default=2,
                            help='inputs dimension')
        parser.add_argument('--num_layers', type=int, default=3, help='number of ST Block')
        parser.add_argument('--heads',
                            type=int,
                            default=1,
                            help='attention heads')
        parser.add_argument('--forward_expansion',
                            type=int,
                            default=4,
                            help='dimension for FFN')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.3,
                            help='dropout rate')
        parser.add_argument(
            '--n_obs',
            default=None,
            help='Only use this many observations. For unit testing.')
        parser.add_argument('--apt_size', default=10, type=int)
        parser.add_argument('--cat_feat_gc', action='store_true')
        parser.add_argument('--fill_zeroes', action='store_true')
        parser.add_argument('--checkpoint', type=str, help='')
        return parent_parser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)  # [B, C=1, N ,T']
        predict = self.scaler.inverse_transform(output)
        assert predict.shape[1] == 1
        mae, mape, rmse = calc_metrics(predict.squeeze(1), y, null_val=0.0)
        self.log("train_mae", mae)
        self.log("train_mape", mape)
        self.log("train_rmse", rmse)
        return mae

    def _dev_setp(self, batch, batch_idx):
        x, y = batch
        output = self(x)  #[B, C, N, T]
        predict = self.scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=70.)

        real = torch.unsqueeze(y, dim=1)
        mae, mape, rmse = calc_metrics(predict, real, null_val=0.0)
        return mae, mape, rmse

    def _dev_epoch_end(self, outputs, stage=None):
        if stage:
            maes = []
            mapes = []
            rmses = []
            for (mae, mape, rmse) in outputs:
                maes.append(mae)
                mapes.append(mape)
                rmses.append(rmse)
            if len(maes) > 0:
                self.log(f"{stage}_mae", torch.stack(maes).mean())
                self.log(f"{stage}_mape", torch.stack(mapes).mean())
                self.log(f"{stage}_rmse", torch.stack(rmses).mean())

    def validation_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._dev_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._dev_epoch_end(outputs, "test")

    def configure_optimizers(self):
        lr = self.hparams.lr
        weight_decay = self.hparams.wd

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=lr,
                                     weight_decay=self.hparams.wd)

        if self.hparams.opt == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
            self._logger.info(
                f"use adamw optimizer, lr: {lr}, weight_decay: {weight_decay}")
        elif self.hparams.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
            self._logger.info(
                f"use adami optimizer, lr: {lr}, weight_decay: {weight_decay}")
        elif self.hparams.opt == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        momentum=0.9)
            self._logger.info(
                f"use sgd optimizer, lr: {lr}, momentum: 0.9 weight_decay: {weight_decay}"
            )
        elif self.hparams.opt == "radam":
            from radam import RAdam
            optimizer = RAdam(self.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
            self._logger.info(
                f"use radam optimizer, lr: {lr}, weight_decay: {weight_decay}"
            )

        scheduler = None
        if self.hparams.sched == "exp":
            scheduler = dict(lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.hparams.lr_decay**epoch))

            self._logger.info(
                f"use ExponentialLR, lr: {lr}, gamma: {self.hparams.lr_decay}")
        elif self.hparams.sched == "onecycle":
            scheduler = dict(lr_scheduler=torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=self.hparams.epos,
                steps_per_epoch=self.hparams.step_per_epoch,
                pct_start=0.25,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=1e5,
                last_epoch=-1,
            ),
                             interval="step")
            self._logger.info(f"use OneCycleLR, max_lr: {lr}")
        ret = dict(optimizer=optimizer)
        if scheduler:
            ret.update(scheduler)
        return ret


def main():
    # ------------
    # seed
    # ------------
    torch.set_printoptions(precision=5)
    #seed_everythin(42)

    # ------------
    # args
    # ------------
    config = Config()
    parser = config.get_parser()
    parser = Net.add_model_specific_args(parser)
    # parser = Trainer.add_argparse_args(parser)
    conf = config.get_config(parser=parser)

    logger = get_logger(conf)
    logger.info(f"{'='*42} \n\t {conf} \n\t {'='*42}")

    # ------------
    # data
    # ------------
    dm = DataModule(conf, logger)
    dm.prepare_data()
    dm.setup()
    dl_trn = dm.train_dataloader()
    dl_val = dm.val_dataloader()

    # calc step_per_epoch
    step_per_epoch = len(dl_trn)
    conf.step_per_epoch = step_per_epoch

    # ------------
    # model
    # ------------
    model = Net(conf, dm.supports, dm.scaler, logger)

    # "checkpoint"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), f"runs/ckpts/{conf.exp}"),
        filename="{epoch}-{val_mae:.4f}",
        save_top_k=1,
        verbose=True,
        monitor='val_mae',
        mode='min',
    )

    # "tensorboard logger"
    tb_logger = TensorBoardLogger(
        "runs/logs",
        name=conf.exp,
    )

    early_stopping = EarlyStopping(monitor='val_mae',
                                   patience=5,
                                   strict=False,
                                   verbose=False,
                                   mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step',
                                     log_momentum=True)

    if conf.tpu:
        gpus = None
    else:
        gpus = conf.gpus if not conf.nogpu else None

    distributed_backend = None
    if len(conf.gpus.split(",")) > 1 or conf.gpus == "-1":
        distributed_backend = "ddp"

    trainer = Trainer(
        fast_dev_run=conf.dbg,
        gpus=gpus,
        tpu_cores=8 if conf.tpu else None,
        max_epochs=conf.epos,
        check_val_every_n_epoch=conf.check_val,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
        # progress_bar_refresh_rate=conf.pb_rate,
        distributed_backend=distributed_backend,
        logger=tb_logger,
        gradient_clip_val=3.0,
        callbacks=[early_stopping],
        # resume_from_checkpoint="ckpts/foo.ckpt"
        # limit_train_batches=4,
        # limit_val_batches=4,
    )

    if conf.lr_find:
        lr_finder = trainer.lr_find(model)
        # "Inspect results"
        # fig = lr_finder.plot(); fig.show(); fig.savefig("lr_finder.png")
        suggested_lr = lr_finder.suggestion()
        logger.info("suggested_lr: ", suggested_lr)
    else:
        trainer.fit(model, dl_trn, dl_val)
        logger.success("training finish!")
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
