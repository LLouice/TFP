import os
import time

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # , LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

from config import Config
from data import DataModule
from utils import get_logger, calc_metrics
from model import GWNet


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
    def __init__(self, hparams, aptinit, supports, scaler, logger=None):
        super(Net, self).__init__()
        self.hparams = hparams
        args = self.hparams

        self.scaler = scaler
        self._logger = logger

        self.model = GWNet(args.N,
                           args.device,
                           dropout=args.dropout,
                           supports=supports,
                           do_graph_conv=args.do_graph_conv,
                           addaptadj=args.addaptadj,
                           aptinit=aptinit,
                           in_dim=args.in_dim,
                           apt_size=args.apt_size,
                           out_dim=args.seq_length,
                           residual_channels=args.nhid,
                           dilation_channels=args.nhid,
                           skip_channels=args.nhid * 8,
                           end_channels=args.nhid * 16,
                           cat_feat_gc=args.cat_feat_gc)

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
        parser.add_argument('--seq_length', type=int, default=12, help='')
        parser.add_argument('--nhid',
                            type=int,
                            default=40,
                            help='Number of channels for internal conv')
        parser.add_argument('--in_dim',
                            type=int,
                            default=2,
                            help='inputs dimension')
        parser.add_argument('--N',
                            type=int,
                            default=207,
                            help='number of nodes')
        parser.add_argument('--batch_size',
                            type=int,
                            default=64,
                            help='batch size')
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
        inp = F.pad(x, (1, 0, 0, 0))
        output = self(inp).transpose(1, 3)  #[B, C, N, T] ?
        predict = self.scaler.inverse_transform(output)
        assert predict.shape[1] == 1
        mae, mape, rmse = calc_metrics(predict.squeeze(1), y, null_val=0.0)
        self.log("train/mae", mae)
        self.log("train/mape", mape)
        self.log("train/rmse", rmse)

        # TODO: Add clip
        # if self.clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        return mae

    def _dev_setp(self, batch, batch_idx, stage=None):
        x, y = batch
        inp = F.pad(x, (1, 0, 0, 0))
        output = self(inp).transpose(1, 3)  #[B, C, N, T] ?
        predict = self.scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=70.)

        real = torch.unsqueeze(y, dim=1)
        mae, mape, rmse = calc_metrics(predict, real, null_val=0.0)
        if stage:
            self.log(f"{stage}/mae", mae)
            self.log(f"{stage}/mape", mape)
            self.log(f"{stage}/rmse", rmse)
        return mae, mape, rmse

    def _dev_epoch_end(self, outputs):
        log = dict()
        return log

        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["val/" + k] = v
        if self._logger:
            self._logger.info(tb_log)
        return dict(log=tb_log)

    def validation_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["val/" + k] = v
        if self._logger:
            self._logger.info(tb_log)
        return dict(log=tb_log)

    def test_step(self, batch, batch_idx):
        self._dev_setp(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["test/" + k] = v
        return dict(log=tb_log)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.wd)

        if self.hparams.opt == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.wd)
        elif self.hparams.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.lr,
                                         weight_decay=self.hparams.wd)
        elif self.hparams.opt == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        weight_decay=self.hparams.wd,
                                        momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.hparams.lr_decay ** epoch)
        return dict(optimizer=optimizer, lr_scheduler=scheduler)


def main():
    # ------------
    # seed
    # ------------
    torch.set_printoptions(precision=5)
    seed_everything(42)

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

    # ------------
    # model
    # ------------
    model = Net(conf, dm.aptinit, dm.supports, dm.scaler, logger)

    # "checkpoint"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "runs/ckpts"),
        filename=conf.exp + "/{epoch}-{val_acc:.4f}",
        save_top_k=1,
        verbose=True,
        monitor='val/mse',  # TODO: the name ?
        mode='min',
    )

    # "tensorboard logger"
    tb_logger = TensorBoardLogger(
        "runs/logs",
        name=conf.exp,
    )

    early_stopping = EarlyStopping(monitor='val/mse',
                                   patience=5,
                                   strict=False,
                                   verbose=False,
                                   mode='min')
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
        # early_stop_callback=early_stopping,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=conf.pb_rate,
        distributed_backend=distributed_backend,
        logger=tb_logger,
        callbacks=[early_stopping],
        # resume_from_checkpoint="ckpts/foo.ckpt"
    )

    if conf.lr_find:
        lr_finder = trainer.lr_find(model)
        # "Inspect results"
        # fig = lr_finder.plot(); fig.show(); fig.savefig("lr_finder.png")
        suggested_lr = lr_finder.suggestion()
        logger.info("suggested_lr: ", suggested_lr)
    else:
        trainer.fit(model, dm)
        logger.success("training finish!")
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
