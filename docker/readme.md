# Container for the SeqNAS

To build and run the container you need to install [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Build
To build the SeqNAS image run:
```bash
docker compose build
```
## Run container
To run the container execute the following command: 
```bash
docker compose up
```
You need to pass some envirinment variables or edit the [env-file](.env).
Variables are descripted there.

## Experiments
Для запуска экспериментов можно воспользоваться командой j\ для создания нового контейнера
To run experiments you can use the `docker-compose run` command to run new container or `docker exec` to run in the existing one.
Examples (workdir is mounted to `/rnas`):
```bash
docker-compose run --rm --name your_container_name -w /rnas jupyter-server python -m SeqNAS search --env_cfg examples/sample_configs/env.yaml --exp_cfg examples/sample_configs/experiment/amex_random.yaml --worker_count=2 --gpu_num=0,1
```
or
```bash
docker exec -itw /rnas docker-jupyter-server-1 python -m SeqNAS search --env_cfg examples/sample_configs/env.yaml --exp_cfg examples/sample_configs/experiment/amex_random.yaml --worker_count=2 --gpu_num=0,1```
