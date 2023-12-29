# Dataset preprocess

Before running main scripts you have to preprocess your dataset to the appropriate format for *webdataset*. <br/>
Details about this format you can find in [examples/sample_configs/README.md](../sample_configs/README.md)
There is information about used datasets below.

# Requires pre-processing 

## UCR datasets (ElectricDevices and Insect)

Use commands below to download and preprocess any UCR dataset:
```shell
cd /home/dev
wget https://timeseriesclassification.com/Downloads/ElectricDevices.zip
python examples/dataset_preprocess/ucr_prepr.py --ucr_zip=ElectricDevices.zip --save_folder=/data/ElectricDevices
rm ElectricDevices.zip
```

```shell
wget https://timeseriesclassification.com/Downloads/InsectSound.zip
python examples/dataset_preprocess/ucr_prepr.py --ucr_zip=InsectSound.zip --save_folder=/data/InsectSound
rm InsectSound.zip
```


# Does not require pre-processing 

## Amex

Dataset is already in necessary format. <br/>
Download [Amex dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format?select=train.parquet). Create `/data/amex` directory and move *train.parquet* and *test.parquet* to it.

## Alpha

Dataset is taken from [this hackathon](https://boosters.pro/championship/alfabattle2/data/vtoraya_zadacha), but links for downloading data and sending results don't work.

## VTB

TBA

