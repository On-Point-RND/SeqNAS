docker run -it --add-host=docker:172.17.0.1  --rm --memory=64g --memory-swap=32g --cpuset-cpus=40-60 --gpus '"device=0,1,2,"' --memory=16g  --shm-size=1gb  --name rnas  -v "$PWD:/home/dev" -v "/home/xaosina/dev/data_main/NAS:/data"  rnas:v1.2.1

