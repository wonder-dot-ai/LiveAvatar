import json
import os
import argparse
import yaml

args = None


def parse_hp_string(hp_string):
    result = {}
    for pair in hp_string.split(","):
        if not pair:
            continue
        key, value = pair.split("=")
        try:
            ori_value = value
            value = float(value)
            if "." not in str(ori_value):
                value = int(value)
        except ValueError:
            pass

        if value in ["true", "True"]:
            value = True
        if value in ["false", "False"]:
            value = False
        if "." in key:
            keys = key.split(".")
            keys = keys
            current = result
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            result[key.strip()] = value
    return result


def parse_args_for_training_config(training_config_path):
    training_config = {}
    if training_config_path:
        with open(training_config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Apply YAML config values not already defined by argparse
        for key, value in yaml_config.items():
            if not hasattr(training_config, key):
                training_config[key] = value
            elif getattr(training_config, key) is None:
                training_config[key] = value

    return training_config


def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")

    # Define argparse parameters
    parser.add_argument("--exp_path", type=str, help="Path to save the model.")
    parser.add_argument("--input_file", type=str, help="Path to inference txt.")
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("-hp", "--hparams", type=str, default="")

    args = parser.parse_args()

    # Load YAML config if --config is provided
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Apply YAML config values not already defined by argparse
        for key, value in yaml_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            elif getattr(args, key) is None:
                setattr(args, key, value)

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun
    args.device = f"cuda:{args.local_rank}"
    args.num_nodes = int(os.getenv("NNODES", "1"))
    debug = args.debug
    if not os.path.exists(args.exp_path):
        args.exp_path = f"checkpoints/{args.exp_path}"

    if hasattr(args, "reload_cfg") and args.reload_cfg:
        # Reload config file
        conf_path = os.path.join(args.exp_path, "config.json")
        if os.path.exists(conf_path):
            print("| Reloading config from:", conf_path)
            args = reload(args, conf_path)
    if len(args.hparams) > 0:
        hp_dict = parse_hp_string(args.hparams)
        for key, value in hp_dict.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            else:
                # if key == 'debug':
                #     import pdb;pdb.set_trace()
                if isinstance(value, dict):
                    ori_v = getattr(args, key)
                    ori_v.update(value)
                    setattr(args, key, ori_v)
                else:
                    setattr(args, key, value)
    # args.debug = debug
    dict_args = convert_namespace_to_dict(args)
    if args.local_rank == 0:
        print(dict_args)
    return args


def reload(args, conf_path):
    """Reload config file without overwriting existing parameters."""
    with open(conf_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    # Apply YAML config values not already defined by argparse
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        elif getattr(args, key) is None:
            setattr(args, key, value)
    return args


def convert_namespace_to_dict(namespace):
    """Convert argparse.Namespace to dict, handling non-serializable objects."""
    result = {}
    for key, value in vars(namespace).items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, OverflowError):
            result[key] = str(value)
    return result
