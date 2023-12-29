def disp_arch(arch):
    k_pad = max(len(k) for k in arch.keys()) + 2
    v_pad = max(len(str(v)) for v in arch.values()) + 2

    for k in sorted(arch):
        print(f"{k:.<{k_pad}}{str(arch[k]):.>{v_pad}}")


def data_to_device(batch, device):
    for key in batch["model_input"].keys():
        batch["model_input"][key] = batch["model_input"][key].to(device)
    batch["time"] = batch["time"].to(device)
    return batch


def print_params(params_dict):
    print("Params:")
    for param_name in params_dict.keys():
        print(f" - {param_name}: {params_dict[param_name]}")


def print_arch(arch):
    print("Arch:")
    for key in arch.keys():
        print(f" - layer: {key}: {arch[key]}")
