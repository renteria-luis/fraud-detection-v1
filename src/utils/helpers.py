import numpy as np

def sanitize_dict(obj):
    """
    Recursively converts NumPy and Pandas types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_dict(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif hasattr(obj, 'to_dict'):  # For Pandas objects
        return sanitize_dict(obj.to_dict())
    else:
        return obj
