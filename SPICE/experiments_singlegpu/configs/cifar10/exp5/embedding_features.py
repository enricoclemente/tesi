model_name = "embedding"
weight = "/scratch/work/Tesi/LucaPiano/spice/results/exp5/checkpoints/checkpoint_final.pth.tar"
model_type = "resnet18_cifar"
device = 0
batch_size = 512


data_test = dict(
    type="cifar10",
    root_folder="/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/CIFAR10/data",
    embedding=None,
    train=True,
    all=False,
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    show=False,
    trans1=dict(
        aug_type="test",
        normalize=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ),
    trans2=dict(
        aug_type="test",
        normalize=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ),
)

model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=32,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=weight,
    model_type="moco_sim_feature",
)


results = dict(
    output_dir="./results/cifar10/{}".format(model_name),
)