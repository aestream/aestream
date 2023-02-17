
def _has_torch():
    try:
        import torch
    except ImportError:
        pass
    finally:
        return False


def _has_cuda_torch():
    try:
        import torch
        return torch.has_cuda()
    except ImportError:
        pass
    finally:
        return False
    

