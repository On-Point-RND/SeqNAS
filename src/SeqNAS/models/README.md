# Models

This module contains models and model skeletons for searching.
Models can be accessed by model name instead of explicitly importing them.
To do it one shoud properly register the model:

```python
    from SeqNAS.models import register_model

    @register_model("MyAwesomeModel")
    class MyModel(nn.Module):
        # model code
```

After this registratin `MyModel` can be accessed by its name "MyAwesomeModel"
in training scripts, in the `get_model` function in `pipeline` module or explicitly:

```python
    from SeqNAS.models import MODEL_REGISTRY

    model = MODEL_REGISTRY["MyAwesomeModel"](args)
```

