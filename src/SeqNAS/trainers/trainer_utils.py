class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, metric, name, metric_type):
        self.metric_type = metric_type
        self.metric = metric
        self.name = name
        self.history = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.last_value = 0
        if self.metric_type == "full":
            self.metric.reset()

    def _reset(self):
        self.history.append(self.avg)
        """Reset all statistics"""
        value = 0
        self.last_value = value
        self.avg = value
        self.sum = value
        self.count = 0
        if self.metric_type == "full":
            self.metric.reset()

    def _update_avg(self, val, n=1):
        """Update statistics"""
        self.last_value = val
        if self.name == "loss":
            val = val.detach().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self, outputs, batch):
        if self.metric_type == "average":
            val = self.metric(outputs, batch)

            bs = outputs["preds"].shape[0]
            self._update_avg(val, bs)

        if self.metric_type == "full":
            self.avg = self.metric(outputs, batch)
