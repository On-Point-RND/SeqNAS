# Sample configs description

There are default configs in this directory.

## Environment config - env.yaml

Environment config is shown below.

```yaml
PROJECT:
  ROOT:
    "./"
    # this parameter should be in __BASE__ config
  DEFAULT_DIRS:
    [
      "nash_logging",
      "search_optimizers",
      "search_spaces",
      "trainers",
      "configs",
      "experiments_src",
      "models",
    ]

EXPERIMENT:
  DIR: "./experiments"
TENSORBOARD_SETUP:
  FLUSH_EVERY_N_MIN: 1
  LOG_PARAMS: True
  LOG_PARAMS_EVERY_N_ITERS: 1

HARDWARE:
  GPU: 0
  WORKERS: 4
```
Description:
- PROJECT - main project directories
- EXPERIMENT - directory where experiment results will be saved
- TENSORBOARD_SETUP - configs for tensorboard
- HARDWARE - gpus/dataloaders numbers
  - GPU - number of gpus on which computation is made
  - WORKERS - number of dataloader workers per single gpu

It is recommended to **not change** anything in this config. You can change only `GPU` and `WORKERS`
parameters if you don't want to set them through command line (`--gpu_num`, `--worker_count`).

## Experiment configs

Example of experiment config is shown below.

```yaml
search_method:
  name: RandomSearcher
  params:
    arches_count: 100
trainer:
  loss: "WeightCrossEntropyLoss"
  metrics: [ "accuracy", "AmexMetric"]
  epochs: 5
  scoring_metric: "AmexMetric"
  optimizer: "FUSEDLAMB"
  optim_params:
    lr: 0.001
    weight_decay: 0.001
  use_amp: true
dataset:
  dataset_type: "WebSequenceDataset"
  data_path: "/data/amex/train.parquet"
  batch_size: 256
  dataset_params:
    index_columns: [ "customer_ID" ]
    sort_columns: [ "S_2" ]
    target_column: "target"
    classification: true
    test_dataset: "/data/amex/test.parquet"
    categorical_columns: ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    numerical_columns: ['P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41', 'B_3', 'D_42', 'D_43', 'D_44', 'B_4', 'D_45',
               'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49', 'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52',
               'P_3', 'B_10', 'D_53', 'S_5', 'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13',
               'R_5', 'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_65', 'D_135', 'D_136',
               'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'S_12', 'R_6', 'S_13', 'B_21', 'D_69', 'B_22', 'R_26', 'R_27',
               'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74', 'D_75', 'D_76', 'B_24', 'R_7', 'D_77',
               'B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10', 'R_11', 'B_27', 'D_81', 'D_82',
               'S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'S_18', 'D_86', 'D_134',
               'D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21', 'B_33', 'D_89',
               'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22', 'S_23', 'S_24', 'S_25',
               'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36', 'B_37', 'D_143', 'D_144', 'D_145',
               'D_108', 'D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40', 'S_27', 'D_113', 'D_115', 'D_141', 'D_142',
               'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125',  'D_127', 'D_138', 'D_139', 'D_140',
               'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131', 'D_132', 'D_133', 'D_137', 'R_28']
    seq_len_limit: 13
    seq_len_trunc_type: "last"
    min_partition_num: 4
    partition_size_mbytes: 64
    split_sizes:
      test: 0.
      train: 0.8
      val: 0.2
    seed: 42
model_name: FlexibleTransformer
model:
  hidden_size: 32
  output_size: 2
  embeddings_hidden: 8
  dropout: 0.3
  tricks: "choice"
```

Description:
- search_method - block contains search method info
- trainer - block contains trainer info
- dataset - block contains info about processing dataset by *webdataset*
- model_name - model name
- model - model parameters

Parameter combinations for *search_method* are listed below:

```yaml
search_method:
  name: RandomSearcher
  params:
    arches_count: 100 # number of architectures that will be computed
```

```yaml
search_method:
  name: Bananas
  params:
    num_trials: 20 # number of bananas iterations (when select new best arches and compute them)
    computed_arches_dir: null # if you want to continue training you can pass directory with already computed arches
    bananas_config: "/home/dev/examples/sample_configs/bananas.yaml" # path to bananas config
```

```yaml
search_method:
  name: Hyperband
  params:
    max_total_epochs: 1000 # maximum number of epoches during searching
    epochs_growth_factor: 3 # parameter of hyperband
    max_epochs_per_arch: 5 # maximum number of epoches per architecture
```

```yaml
search_method:
  name: DiffSearcher
  params: { } # no parameters
```

```yaml
search_method:
  name: PTSearcher
  params: { } # no parameters
```

Description of *trainer* parameters is shown below:

```yaml
trainer:
  loss: "WeightCrossEntropyLoss" # supported default losses: [CrossEntropyLoss, WeightCrossEntropyLoss, MSELoss, MAELoss]
  metrics: [ "accuracy", "AmexMetric"] # supported default metrics: [f1_macro, f1_weighted, accuracy, auc, r2]
  epochs: 5 # number of epoches per architecture (not used for hyperband)
  scoring_metric: "AmexMetric" # scoring metric to select best architecture (choose one out of metrics above)
  optimizer: "FUSEDLAMB" # supported optimizers: [FUSEDSGD, FUSEDADAM, FUSEDNOVOGRAD, FUSEDLAMB]
  optim_params:
    lr: 0.001
    weight_decay: 0.001
  use_amp: true # use Mixed Precision or not
```

Description of *dataset* parameters is shown below:

```yaml
dataset:
  dataset_type: "WebSequenceDataset" # only WebSequenceDataset currently type supported
  data_path: "/data/amex/train.parquet" # path to train data. supports [.csv, .csv.zip, .parquet] formats
  batch_size: 256 # batch size
  dataset_params:
    index_columns: [ "customer_ID" ] # name of ID column
    sort_columns: [ "S_2" ] # name of time column
    target_column: "target"
    classification: true # set *false* if regression
    test_dataset: "/data/amex/test.parquet" # path to test data. Don't fill this field if you don't have test data
    categorical_columns: [...]
    numerical_columns: [...]
    seq_len_limit: 13 # maximum sequence length
    seq_len_trunc_type: "last" # get last seq_len_limit values. Set *first* if you want to keep first values
    min_partition_num: 4 # minimum number of parts into which the dataset will be divided
    partition_size_mbytes: 64
    split_sizes:
      test: 0. # set 0. if you have test dataset
      train: 0.8
      val: 0.2
    seed: 42 # seed fixes data splitting
```

Model name and search method matching is shown below:

| Model name        | Searcher                           |
|:------------------|:-----------------------------------|
| `FlexibleTransformer` | RandomSearcher, Bananas, Hyperband |
| `FlexibleTransformerDecoder`    | RandomSearcher, Bananas, Hyperband |
| `EncoderDecoderModelDiff`         | DiffSearcher                       |
| `EncoderDecoderModelPT`         | PTSearcher                         |

Description of main *model* parameters is shown below:

```yaml
model:
  hidden_size: 32 # hyperparameter
  output_size: 2 # number of target classes
  embeddings_hidden: 8 # hyperparameter
  num_embeddings_hidden: "auto" # auto by default or int value, optional.
  dropout: 0.3 # hyperparameter
  tricks: "choice" # possible values: [choice, all_or_nothing, off]
```

To see all parameters find init model method

## Optuna config - hpo_config.yaml

Optuna config is shown below.

```yaml
sampler_method: TPESampler

# hyperparams for tuning
hyperparams:
  train_params:
    batch_size:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ 256 ]
    optimizer:
      trial_method: suggest_categorical
      trial_method_params:
        choices: ['FUSEDSGD', 'FUSEDADAM', 'FUSEDNOVOGRAD', 'FUSEDLAMB']
    optimizer_params:
      lr:
        trial_method: suggest_categorical
        trial_method_params:
          choices: [ 1e-4, 1e-3, 1e-2 ]
      weight_decay:
        trial_method: suggest_categorical
        trial_method_params:
          choices: [ 1e-4 ]
      beta1:
        trial_method: suggest_categorical
        trial_method_params:
          choices: [ 0.7, 0.8, 0.9 ]
  model_params:
    hidden_size:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ 32, 64 ]
    num_embeddings_hidden:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ 8, 16 ]
    embeddings_hidden:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ 8, 16 ]
    dropout:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ 0.0, 0.1, 0.2, 0.3, 0.5, 0.7 ]
    augmentations:
      trial_method: suggest_categorical
      trial_method_params:
        choices: [ None ]
```

There is so easy to understand interface. Several advices:
- Choose only single `batch_size` to decrease search space
- Use only `FUSEDADAM` and `FUSEDLAMB` optimizers. In our experiments they achieved better results than the others
- There is no sense to search many values of `hidden_size`, `num_embeddings_hidden` and `embeddings_hidden` because after some value score stops to increase and locates on plateau
- In our experiments the use of `augmentations` didn't give good score. That's why there is `None` by default

## Bananas config - bananas.yaml

Bananas config is shown below.

```yaml
initial_step: 100
candidates_to_seed: 100
candidates_per_step: 15
predictor_objective: MAE
acquisition_function: ITS
predictor_ensembles: 5
predictor_iters: 100
predictor_lr: 0.01
candidate_generation:
  type: random
```

Bananas consists of the following steps:
1. Runs random search on `initial_step` architectures
2. Trains Catboost on founded architectures
3. Generates new `candidates_to_seed` architectures and estimates score on them by Catboost
4. Gets the best `candidates_per_step` architectures and make search on them
5. Repeats 2-4 steps `num_trials` times
