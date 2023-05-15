import torch

def GetCuda():
    """
    Checks if the system has a CUDA-capable GPU.

    Returns:
        bool: True if a CUDA device is available, False otherwise.
    """
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cuda_string = f"CUDA version: {cuda_version}"
        print(cuda_string)
        return True
    else:
        print("Warning: CUDA is not available. Running on CPU.")
        return False



GetCuda()