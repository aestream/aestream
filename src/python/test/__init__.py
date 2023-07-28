def _has_torch():
    try:
        import torch

        return True
    except ImportError:
        pass
    return False


def _has_cuda_torch():
    try:
        import torch

        return torch.has_cuda
    except ImportError:
        pass
    return False
