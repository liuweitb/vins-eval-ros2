import yaml


def load_camera_yaml(path: str) -> dict:
    with open(path) as f:
        lines = f.read().splitlines()
    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]
    return yaml.safe_load("\n".join(lines)) or {}
