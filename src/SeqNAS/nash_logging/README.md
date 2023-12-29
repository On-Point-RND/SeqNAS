# Logger Functionality:

Main using class is **LoggerUnited** in `SeqNAS/nash_logging/common.py`
Logger supports multiprocessing

Logger takes:
- Environment config file (default file `examples/sample_configs/env.yaml`)
- tensorboard online logger

Logger performs:
- printing to stdout
- printing to log files
- saves tensorboard logs
- saving custom txt and torch files

Usage:
1. Specify config file.
2. Run `examples/main_examples/search.py` or `examples/main_examples/train_final.py` file, logger will dump txt to <br/> 
and tensorboard logs to experiment folder.

