dataset = dict(
    img_size = 224,
    normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

moco = dict(
    moco_dim=128,
    moco_k=4096,
    moco_m=0.999,
    moco_t=0.2,
    mlp=True,
)
