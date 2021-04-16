import numpy as np
from mxnet import metric
from mxnet.metric import CompositeEvalMetric


def mae_metric(scaler):
    def _f(labels, preds):
        labels = labels.transpose(0, 3, 2, 1)
        preds = scaler.inverse_transform(labels)

        null_val = 0.0
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)

        mask = mask.astype("float32")
        mask /= np.mean(mask)

        mae = np.abs(preds - labels)
        return np.mean(np.nan_to_num(mae * mask))

    return _f


def mape_metric(scaler):
    def _f(labels, preds):
        labels = labels.transpose(0, 3, 2, 1)
        preds = scaler.inverse_transform(labels)

        null_val = 0.0
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)

        mask = mask.astype("float32")
        mask /= np.mean(mask)

        mae = np.abs(preds - labels)
        mape = mae / labels
        return np.mean(np.nan_to_num(mape * mask))

    return _f


def rmse_metric(scaler):
    def _f(labels, preds):
        labels = labels.transpose(0, 3, 2, 1)
        preds = scaler.inverse_transform(labels)

        null_val = 0.0
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)

        mask = mask.astype("float32")
        mask /= np.mean(mask)

        mse = (preds - labels)**2
        rmse = np.sqrt(mse)
        return np.mean(np.nan_to_num(rmse * mask))

    return _f


def get_mae_metric(scaler):
    return metric.np(mae_metric(scaler))


def get_mape_metric(scaler):
    return metric.np(mape_metric(scaler))


def get_rmse_metric(scaler):
    return metric.np(rmse_metric(scaler))


def get_val_metrics(scaler):
    val_metrics = CompositeEvalMetric()
    val_metrics.add(get_mae_metric(scaler))
    val_metrics.add(get_mape_metric(scaler))
    val_metrics.add(get_rmse_metric(scaler))
    return val_metrics


# class MAE(EvalMetric):

#     def __init__(self, name='mae',
#                  output_names=None, label_names=None):
#         super(MAE, self).__init__(
#             name, output_names=output_names, label_names=label_names,
#             has_global_stats=True)

#     def update(self, labels, preds):
#         """Updates the internal evaluation result.

#         Parameters
#         ----------
#         labels : list of `NDArray`
#             The labels of the data.

#         preds : list of `NDArray`
#             Predicted values.
#         """
#         labels, preds = check_label_shapes(labels, preds, True)

#         labels = labels.transpose(0, 3, 2, 1)
#         preds = scaler.inverse_transform(labels)

#         null_val = 0.0
#         if np.isnan(null_val):
#             mask = ~np.isnan(labels)
#         else:
#             mask = np.not_equal(labels, null_val)

#         mask = mask.astype("float32")
#         mask /= np.mean(mask)

#         mae = np.abs(preds - labels)
#         mae = np.mean(np.nan_to_num(mae * mask))

#         self.sum_metric += mae
#         self.global_sum_metric += mae
#         self.num_inst += 1 # numpy.prod(label.shape)
#         self.global_num_inst += 1 # numpy.prod(label.shape)
