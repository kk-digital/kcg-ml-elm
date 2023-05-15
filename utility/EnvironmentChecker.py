import os

def GetEnvironment():
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'KAGGLE_URL_BASE' in os.environ:
        return 'kaggle'
    else:
        return 'local'

x = GetEnvironment()
if x == 'colab':
    print(f"Environment is {x}")
elif x == 'kaggle':
    print(f"Environment is {x}")
else:
    print(f"Environment is {x}")