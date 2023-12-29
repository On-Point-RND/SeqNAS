from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="SeqNAS",
    version="1.0.0",
    description="A pytorch neural architecture search library for sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igudav/SeqNAS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"SeqNAS": ["recommendation/data/default_priors.pkl"]},
    python_requires=">=3.7, <4",
    install_requires=[
        "omegaconf==2.3.0",
        "webdataset==0.1.103",
        "torchmetrics==0.10.3",
        "pyarrow==7.0.0",
        "pyspark==3.3.2",
        "catboost==1.1.1",
        "iopath==0.1.10",
        "deflate_dict==1.0.9",
        "dict_hash==1.1.26",
        "mysqlclient==2.1.1",
        "optuna==3.1.0",
    ],
)
