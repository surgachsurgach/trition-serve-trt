def str2bool(v):
    if not isinstance(v, str):
        raise ValueError("Boolean value expected.")
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise ValueError(f"Cannot convert {v} to bool.")
