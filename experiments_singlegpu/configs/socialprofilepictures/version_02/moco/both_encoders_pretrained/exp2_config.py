dataset = dict(
    version=2,
    randomize_metadata=False,
    img_size = 224,
    normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
    train_padding = False,
    train_resize = True,
    test_padding = False,
    test_resize = True,
)

model = "resnet18"
model_path = "/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_02/moco/both_encoders_pretrained/exp2/checkpoints/checkpoint_last.pth.tar"

moco = dict(
    moco_dim=128,
    moco_k=4096,
    moco_m=0.999,
    moco_t=0.2,
    mlp=True,
    query_encoder_pretrained=True,
    key_encoder_pretrained=True
)
