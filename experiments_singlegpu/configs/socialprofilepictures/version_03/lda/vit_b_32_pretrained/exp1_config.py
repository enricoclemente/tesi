dataset = dict(
    version=3,
    randomize_metadata=True,
    img_size = 224,
    normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
)

model="vit_b_32"
model_path=""
