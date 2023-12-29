import yaml


def read_yaml(path):
    with open(path, "rb") as inp:
        cfg = yaml.load(inp, Loader=yaml.FullLoader)
    return cfg


def write_yaml(cfg, path):
    with open(path, "w") as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
