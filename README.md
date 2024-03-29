This is the official PyTorch implementation of the IEEE Acccess (2024) paper [SeqNAS: Neural Architecture Search for Event Sequence Classification](https://ieeexplore.ieee.org/abstract/document/10379671). Citation:
```bibtex
@ARTICLE{udovichenko2024seqnas,
  author={Udovichenko, Igor and Shvetsov, Egor and Divitsky, Denis and Osin, Dmitry and Trofimov, Ilya and Sukharev, Ivan and Glushenko, Anatoliy and Berestnev, Dmitry and Burnaev, Evgeny},
  journal={IEEE Access}, 
  title={SeqNAS: Neural Architecture Search for Event Sequence Classification}, 
  year={2024},
  volume={12},
  number={},
  pages={3898-3909},
  keywords={Computer architecture;Task analysis;Benchmark testing;Training;DNA;Artificial neural networks;Transformers;Event detection;Knowledge management;NAS;temporal point processes;event sequences;RNN;transformers;knowledge distillation;surrogate models},
  doi={10.1109/ACCESS.2024.3349497}
}
```

## Contents:
- [Usage examples](./examples/README.md)
    -  [Dataset preprocesssing](./examples/dataset_preprocess/README.md)
    -  [Configs](./examples/sample_configs/README.md)
- [Datloaders](./src/SeqNAS/datasets/README.md)
- [Models](./src/SeqNAS/models/README.md)
- [Logging](./src/SeqNAS/nash_logging/README.md)
- [Search spaces](./src/SeqNAS/search_spaces/README.md)
- [Trainer](./src/SeqNAS/trainers/README.md)



<hr>

# Quick Start                                                                                                                                                                                            
### Install as a package

```bash
git clone https://github.com/igudav/SeqNAS.git
pip3 install . 
```

### Use in a docker container

Without installing as a package

```bash
git clone https://github.com/igudav/SeqNAS.git
cd docker
docker-compose build && docker-compose up
```

## Datasets
Our datasets are available [here](https://disk.yandex.ru/d/N2TzBBTo8Ac7lQ). 

A dataset with evaluated architectures is [here](nas-bench-event-sequences.zip).
The dataset contains JSON files with experiment results for each dataset using random search and SeqNAS.
These JSON file are not intended to be human readable, but here is the commented (JSONC) example of an architectures dataset item:

```jsonc
{
    "arch": {  // encoded architecture
        "position_encoding": 0,  // 1 if used

        // entries related to the encoder
        "encoder": 1,  // whether the arch contains an encoder
        "encoder.layer.layers": [1, 1, 1, 1],  // 4 layers
        // the combination of operations used (see Figure 5 in the paper)
        "encoder.layer.layers.ops.0": [1, 0, 0, 0, 0, 0, 0],
        // A number of heads in the MHA if used. 2 heads in the 1-st layer
        "encoder.layer.layers.ops.0.ops.0.split_layer.layers.0.heads": [1, 1, 0, 0, 0, 0, 0, 0],
        "encoder.layer.layers.ops.1": [0, 1, 0, 0, 0, 0, 0],  // no MHA
        "encoder.layer.layers.ops.2": [0, 0, 0, 0, 1, 0, 0],
        "encoder.layer.layers.ops.2.ops.4.split_layer.layers.0.heads": [1, 1, 0, 0, 0, 0, 0, 0],
        "encoder.layer.layers.ops.3": [0, 0, 0, 0, 1, 0, 0],
        "encoder.layer.layers.ops.3.ops.4.split_layer.layers.0.heads": [1, 1, 0, 0, 0, 0, 0, 0],

        // entries related to the decoder
        "decoder": 1,
        "decoder.layer.layers.repeated_layer": [1, 1],  // 2 layers
        // a number of heads in different attentions in the layers
        "decoder.layer.layers.repeated_layer.ops.0.self_attention.heads": [1, 1, 1, 1, 1, 1, 1, 1],
        "decoder.layer.layers.repeated_layer.ops.0.enc_attention.heads": [1, 1, 0, 0, 0, 0, 0, 0],
        "decoder.layer.layers.repeated_layer.ops.1.self_attention.heads": [1, 0, 0, 0, 0, 0, 0, 0],
        "decoder.layer.layers.repeated_layer.ops.1.enc_attention.heads": [1, 1, 1, 1, 0, 0, 0, 0],

        // entries related to the decoder
        "head.layers.1": 1  // whether the temporal dropout is used
        // How to agregate hidden states. Supported options are: mean, max, mean + max.
        "head.layers.2": [0, 1, 0],
    },
    "score": 0.7103096842765808,  // performance metric value
    // architecture code used to train the Predictor-model (see Algorithm 1 in the paper)
    "arch_code": [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]
}
```
