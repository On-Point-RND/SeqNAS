# Search Spaces Functionality:

Provides flexible way to replace some *place holder* samplers in `SeqNAS/search_spaces/basic_ops.py`<br/>
with *actual* samplers from `SeqNAS/search_spaces/multi_trail_model.py` in model class decorator.<br/>

There is an example below:

```python
@register_model("FlexibleTransformer")
@omnimodel(
    [
        (LayerChoice, SinglePathRandom),
        (Repeat, RepeatRandom),
        (RepeatParallel, ParallelRepsRandom),
        (LayerSkip, SkipSampler),
    ]
)
class FlexibleTransformer(nn.Module):
    def __init__(...
```

First *place holder* samplers (like LayerChoice) in tuples will be replaced by subsequent ones.

There is a table of samplers matching below:

| *Place holder*   | *Actual sampler*                                                                                                          |
|:-----------------|:--------------------------------------------------------------------------------------------------------------------------|
| `LayerChoice`    | `SinglePathRandom` <br/>`DiffLayerSoftMax`<br/> `IdentityLayer`<br/> `SinglePathRandomSimpleFirst`<br/> `DiffLayerGumbel` |
| `LayerSkip`      | `SkipSampler`                                                                                                             |
| `RepeatParallel` | `ParallelRepsRandom`                                                                                                      |
| `Repeat`         | `RepeatRandom`<br/> `ParallelRepsRandom`                                                                                  |
| `Cell`           | `RandomRNN`                                                                                                               |

