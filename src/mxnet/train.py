# import numpy as np
# import pandas as pd
import time

from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler

import utils
from model import GWNet
from config import Config
from data import DataModule
from loss import MAELoss
from metric import get_mae_metric, get_mape_metric, get_rmse_metric
from callback import MyGradientUpdateHandler, MyMetricHandler

npx.set_np()


def main():
    config = Config()
    parser = config.get_parser()
    parser = GWNet.add_model_specific_args(parser)
    conf = config.get_config(parser)
    print(conf)

    # data
    dm = DataModule(conf)
    dm.prepare_data()
    dm.setup()
    dl_trn = dm.train_dataloader()
    dl_val = dm.val_dataloader()
    dl_test = dm.test_dataloader()

    # model
    model = GWNet(
        num_nodes=conf.num_nodes,
        dropout=0.3,
        # supports=dm.supports,
        supports=None,
        do_graph_conv=True,
        addaptadj=True,
        aptinit=None,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        cat_feat_gc=conf.cat_feat_gc,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        apt_size=10)

    model.initialize(init=init.Xavier(), ctx=npx.cpu())
    print(model)

    # for x, y in dl_trn:
    #     model(x)
    #     print(model.summary(x))
    #     break

    # model.hybridize()

    # for x, y in dl_trn:
    #     model(x)
    #     break

    loss_fn = MAELoss(scaler=dm.scaler)
    # loss_fn = MAELoss()

    trainer = gluon.Trainer(model.collect_params(), 'adamw',
                            {'learning_rate': conf.lr})  # Trainer

    est = estimator.Estimator(net=model,
                              loss=loss_fn,
                              val_metrics=[
                                  get_mae_metric(dm.scaler),
                                  get_mape_metric(dm.scaler),
                                  get_rmse_metric(dm.scaler)
                              ],
                              trainer=trainer,
                              context=npx.cpu())

    # ignore warnings for nightly test on CI only
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    # Magic line
    est.fit(train_data=dl_trn,
            val_data=dl_val,
            epochs=1,
            event_handler=[
                MyGradientUpdateHandler(),
                MyMetricHandler(get_mae_metric(dm.scaler, -1000)),
                MyMetricHandler(get_mape_metric(dm.scaler, -1001)),
                MyMetricHandler(get_rmse_metric(dm.scaler, -1002)),
            ])


if __name__ == "__main__":
    main()
    #TODO clip wd lr lr_decay true evaluate
