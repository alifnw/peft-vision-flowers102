"""Explicit method settings and parameter boundaries for the native paper models."""

METHODS = ("linear", "lora", "ssf", "convpass")


def method_settings(config, method):
    if method not in METHODS or method not in config["methods"]:
        raise ValueError("Method must be supported and listed in the configuration")
    settings = {"ssf": False, "lora_bottleneck": 0,
                "ft_attn_module": None, "ft_mlp_module": None}
    if method == "lora":
        rank = config["lora_rank"]
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        settings["lora_bottleneck"] = rank
    elif method == "ssf":
        settings["ssf"] = True
    elif method == "convpass":
        chosen = config["convpass"]
        if chosen["bottleneck"] not in (8, 16):
            raise ValueError("Use a ConvPass bottleneck from the paper's VTAB grid")
        if chosen["scale"] not in (0.01, 0.1, 1, 10, 100):
            raise ValueError("Use a ConvPass scale from the paper's VTAB grid")
        if not isinstance(chosen["xavier_init"], bool):
            raise ValueError("ConvPass initialization flag must be a boolean")
        settings.update(ft_attn_module="convpass", ft_attn_mode="parallel", ft_attn_ln="after",
                        ft_mlp_module="convpass", ft_mlp_mode="parallel", ft_mlp_ln="after",
                        convpass_bottleneck=chosen["bottleneck"],
                        convpass_scaler=chosen["scale"],
                        convpass_xavier_init=chosen["xavier_init"],
                        convpass_init="lora_xavier")
    return settings


def is_method_parameter(name, method):
    if method not in METHODS:
        raise ValueError("Unsupported method")
    parts = name.split(".")
    if method == "lora":
        return "lora" in parts
    if method == "ssf":
        return parts[-1].startswith(("ssf_scale", "ssf_shift"))
    if method == "convpass":
        return any(part in ("ft_attn_module", "ft_mlp_module") for part in parts)
    return False


def expected_method_parameters(config, method, width=768, depth=12):
    method_settings(config, method)
    if method == "lora":
        return depth * 4 * width * config["lora_rank"]
    if method == "ssf":
        return depth * 22 * width + 4 * width
    if method == "convpass":
        rank = config["convpass"]["bottleneck"]
        per_module = 2 * width * rank + 9 * rank ** 2 + 2 * rank + width
        return 2 * depth * per_module
    return 0
