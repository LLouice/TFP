from mxnet import np, use_np
from mxnet.gluon.loss import Loss


@use_np
class MAELoss(Loss):
    def __init__(self, weight=None, batch_axis=0, scaler=None, **kwargs):
        super(MAELoss, self).__init__(weight, batch_axis, **kwargs)
        self._weight = weight
        self._batch_axis = batch_axis
        self.scaler = scaler

    def forward(self, preds, labels):
        labels = labels.transpose(0, 3, 2, 1)
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
        #TODO transform clip norm

        mask = np.not_equal(labels, 0.0)

        mask = mask.astype("float32")
        mask /= np.mean(mask)

        mae = np.abs(preds - labels)
        mae = mae * mask
        return np.mean(np.nan_to_num(mae))
