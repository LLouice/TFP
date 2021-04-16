from mxnet import nd
from mxnet.gluon.loss import Loss


class MAELoss(Loss):
    def __init__(self, weight=None, batch_axis=0, scaler=None, **kwargs):
        super(MAELoss, self).__init__(weight, batch_axis, **kwargs)
        self._weight = weight
        self._batch_axis = batch_axis
        self.scaler = scaler

    def hybrid_forward(self, F, preds, labels):
        labels = F.transpose(labels,axes=(0, 3, 2, 1))
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
        #TODO transform clip norm

        mask = F.not_equal(labels, 0.0)

        mask = mask.astype("float32")
        mask /= F.mean(mask)

        mae = F.abs(preds - labels)
        mae = mae * mask
        # TODO nan_to_num
        return F.mean(mae, axis=(1,2,3))
