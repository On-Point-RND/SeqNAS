# Something similar to pytorch lighting functionality

Trainer takes:
- model,
- dataloaders (train / val),
- logger,
- metrics,
- configuration.

Train performs:
- training,
- validation,
- model checkpointing.

Useage:
1. Specify config file.
2. Instantiate dataloader, logger and model.
3. (optionally) Setup knowledge distillation.
4. (optionally) Load the checkpoint.
5. Train the model for the desired number of steps.

