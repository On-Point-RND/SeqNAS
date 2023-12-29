# Webdataset description

Webdataset was taken from [this repo](https://github.com/webdataset/webdataset) and modified to work with data sequences. <br/>
It splits data into partitions and allow to work with big datasets. <br/>

Main using class is **WebSequenceDataset** in `SeqNAS/datasets/webdataset/sequence_dataset/dataset.py` <br/> 
For dataset creation used *config.yaml* file (or dict in such format). <br/>
When dataset is created there are data and *info.yaml* file are created.

Example of config for dataset and info file you can find in following files:
- `SeqNAS/datasets/webdataset/config_example.yaml`
- `SeqNAS/datasets/webdataset/info_example.yaml`

# Config parameters for data preprocessing: 

* **data_path**: path to source dataset in \[.csv.zip, .csv, .parquet\] format
* **index_columns**: List of index columns (supports only single index)
* **sort_columns**: List of time columns (supports only single)
* **target_column**: target column
* **save_path**: path where preprocessed data will be saved
* **local_path**: always equal to save_path (appendix from original webdataset)
* **classification**: whether classification task or regression
* **val_dataset**: path to validation dataset if it exists
* **test_dataset**: path to test dataset if it exists
* **categorical_columns**: categorical columns
* **numerical_columns**: numerical columns
* **skip_columns**: skip columns
* **categorical_cardinality_limit**: limit on different categorical values
* **export_index_name**: index name after preprocessing
* **seq_len_limit**: max sequence length
* **seq_len_trunc_type**: if "last" - keep values from the end, if "first" - keep values from the beginning
* **min_partition_num**: minimum number of partitions into which the dataset will be split
* **partition_size_mbytes**: size of single partition
* **split_sizes**: sizes for splitting into train/val/test
* **seed**: value that fixes splitting dataset in different launches


## Preprocessing relusts
After *create_dataset* method is called **info.yaml** will be saved in **save_path** and contains following parameters:

* **categorical_cardinality**: cardinalities of categorical features
* **categorical_columns**: list of categorical features
* **dataset_dir**: path to preprocessed dataset
* **numerical_columns**: list of numerical features
* **target_cardinality**: cardinality of target
* **target_column**: target column
* **test_dataset**: list of url partitions of test dataset in the form of bracket notation
* **train_dataset**: list of url partitions of train dataset in the form of bracket notation
* **val_dataset**: list of url partitions of val dataset in the form of bracket notation

