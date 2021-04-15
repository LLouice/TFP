# import numpy as np
# import pandas as pd
import time

from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.contrib.estimator.event_handler import TrainBegin, TrainEnd, EpochEnd, CheckpointHandler

import utils
from model import GWNet
from config import Config
from data import DataModule
from loss import MAELoss
from metric import get_mae_metric, get_mape_metric, get_rmse_metric

npx.set_np()


# from util import calc_tstep_metrics
# from exp_results import summary
class Learner():
    pass


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

    loss_fn = MAELoss(scaler=dm.scaler)
    # loss_fn = MAELoss()

    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': conf.lr})  # Trainer

    # for x, y in dl_trn:
    #     print(x.shape)
    #     print(y.shape)
    #     with autograd.record():
    #         loss = loss_fn(model(x, ), y)
    #     loss.backward()
    #     print("loss:", loss)
    #     break

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
    # est.fit(train_data=dl_trn, epochs=1)


if __name__ == "__main__":
    main()
    # parser = util.get_shared_arg_parser()
    # parser.add_argument('--epochs', type=int, default=100, help='')
    # parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    # parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    # parser.add_argument('--save', type=str, default='experiment', help='save path')
    # parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    # parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')

    # args = parser.parse_args()
    # t1 = time.time()
    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)
    # print(args)
    # # pickle_save(args, f'{args.save}/args.pkl')
    # main(args)
    # # t2 = time.time()
    # # mins = (t2 - t1) / 60
    # # print(f"Total time spent: {mins:.2f} seconds")
