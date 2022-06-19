dataset = dict(
    version=3,
    randomize_metadata=True,
    img_size = 224,
    normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
)

features_layer = "layer4"
model="resnet18"
model_path=""

n_cluster_values = dict(
    start=24,
    end=30
)

umap = dict(
    fix_n_components = False,
    n_epochs = 1000
)

calculate_acc = True