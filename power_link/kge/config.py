"""Tiny JSON-config helper used to load pretraining checkpoints' hyper-params."""

import json


class Config:
    """argparse.Namespace-like wrapper around a flat JSON config dict."""

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def load_config(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return Config(config_dict)
