from importlib import import_module

# Registry of available algorithms
REGISTERED = {
    "ppo": "ppo_agent.PPOAgent",
    # "rppo": "rppo_agent.RPPOAgent",
    # "qrdqn": "qrdqn_agent.QRDQNAgent",
}

def get_agent_class(name: str):
    name = name.lower()
    try:
        module_path, cls_name = REGISTERED[name].rsplit(".", 1)
        module = import_module(f".{module_path}", package=__name__)
        return getattr(module, cls_name)
    except KeyError as e:
        raise ValueError(f"Unknown algorithm '{name}'. "
                         f"Available: {list(REGISTERED)}") from e
