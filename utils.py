import yaml

def read_config_(config):
    with open(config, "r") as f:
        return yaml.safe_load(f)