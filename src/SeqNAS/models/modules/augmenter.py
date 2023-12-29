class Augmenter:
    def __init__(
        self,
        cat_features=[],
        real_features=[],
        cat_transforms=[],
        real_transforms=[],
        seq_len=0,
    ):
        self.cat_features = cat_features
        self.real_features = real_features
        self.cat_transforms = cat_transforms
        self.real_transforms = real_transforms
        self.qeq_len = seq_len

    def __call__(self, x):
        for f_name in x:
            if f_name in self.cat_features:
                for t in self.cat_transforms:
                    x[f_name] = t(x[f_name], self.qeq_len)

            if f_name in self.real_features:
                for t in self.real_transforms:
                    x[f_name] = t(x[f_name], self.qeq_len)

        return x
