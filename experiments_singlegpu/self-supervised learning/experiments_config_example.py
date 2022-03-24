"""
    Example of moco training config file
"""
dataset = dict(
    img_size = 224,
    normalization = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
)

moco = dict(
    moco_dim=128,
    moco_k=65536,
    moco_m=0.999,
    moco_t=0.2,
    mlp=True,
)
