from mxnet.gluon.contrib.estimator.event_handler import BatchEnd, EpochEnd, MetricHandler, GradientUpdateHandler
from mxnet.metric import Loss as metric_loss


class MyGradientUpdateHandler(GradientUpdateHandler):
    """
    just add ignore_stale_grad to the default GradientUpdateHandler
    ----------
    """
    def __init__(self, priority=-2000):
        self.priority = priority
        super(MyGradientUpdateHandler, self).__init__(priority)

    def batch_end(self, estimator, *args, **kwargs):
        print(f"call MyGradientUpdateHandler {self.priority}")
        loss = kwargs['loss']
        batch_size = 0
        if not isinstance(loss, list):
            loss = [loss]
        if isinstance(loss, list):
            for l in loss:
                batch_size += l.shape[0]

        estimator.trainer.step(batch_size, ignore_stale_grad=True)


class MyMetricHandler(MetricHandler):
    def __init__(self, metrics, priority=-1000):
        super(MyMetricHandler, self).__init__(metrics, priority)

    # override
    def batch_end(self, estimator, *args, **kwargs):
        print(f"call MyMetricHandler {self.priority}")
        pred = kwargs['pred']
        label = kwargs['label']
        loss = kwargs['loss']
        for metric in self.metrics:
            if isinstance(metric, metric_loss):
                # metric wrapper for loss values
                metric.update(0, loss.as_nd_ndarray())
            else:
                metric.update(label, pred)
