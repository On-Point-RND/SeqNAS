import torch
import torch.nn as nn
from ..search_spaces.basic_ops import Repeat, LayerChoice
from ..search_spaces.multi_trail_model import SinglePathRandom, RepeatRandom
from ..search_spaces.omnimodels import omnimodel


def init_random_model(classes=10, dim_in=3, mid_dim=9):
    @omnimodel([(LayerChoice, SinglePathRandom), (Repeat, RepeatRandom)])
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv_first = nn.Sequential(
                LayerChoice(
                    [
                        nn.Conv2d(dim_in, mid_dim, 7, 1, 7 // 2),
                        nn.Conv2d(dim_in, mid_dim, 3, 1, 3 // 2),
                    ]
                ),
                nn.ReLU(),
            )

            self.conv_second = nn.Sequential(
                LayerChoice(
                    [
                        nn.Conv2d(mid_dim, mid_dim * 2, 7, 1, 7 // 2),
                        nn.Conv2d(mid_dim, mid_dim * 2, 5, 1, 5 // 2),
                    ]
                ),
                nn.ReLU(),
            )

            self.last = Repeat(
                nn.Sequential(
                    nn.Conv2d(mid_dim * 2, mid_dim * 2, 5, 1, 5 // 2), nn.ReLU()
                ),
                [1, 3],
            )

            self.adapt = torch.nn.AdaptiveMaxPool2d(1)
            self.out = nn.Linear(mid_dim * 2, classes)

        def forward(self, x):
            x = self.conv_first(x)
            x = self.conv_second(x)
            x = self.last(x)
            x = self.adapt(x)
            b = x.shape[0]
            return self.out(x.reshape(b, -1))

    model = Model()
    model.run_replacement()
    # graph = torch.fx.Tracer().trace(model)
    # traced = torch.fx.GraphModule(model, graph)
    # traced.recompile()
    # print(traced.code)
    return model
