dataset = dict(
    dataset_name = "socialprofilepictures",
    version=3,
    randomize_metadata=True,
    img_size = 224,
    normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
    train_padding = False,
    train_resize = True,
    test_padding = False,
    test_resize = True,
)

model = "resnet18"
model_path = "/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/moco/both_encoders_pretrained/exp2/checkpoints/checkpoint_last.pth.tar"

moco = dict(
    moco_dim=128,
    moco_k=32768,
    moco_m=0.999,
    moco_t=0.2,
    mlp=True,
    query_encoder_pretrained=True,
    key_encoder_pretrained=True
)

training = dict(
    epochs = 150,
    batch_size = 256,
    lr = 0.06,
    cosine_lr_decay = True,
    keep_lr = False,
    schedule_lr_decay = [],
)

optimizer = dict(
    momentum = 0.9,
    wd = 1e-4,
)
