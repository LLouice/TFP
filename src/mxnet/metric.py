from mxnet import np, use_np, metric


@use_np
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


@use_np
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


@use_np
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
