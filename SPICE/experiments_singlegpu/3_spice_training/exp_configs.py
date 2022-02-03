batch_size = 5000
target_sub_batch_size = 100
train_sub_batch_size = 128
test_batch_size = 100
num_repeat = 8

dataset = dict(
    img_size=32,
    normalization = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
)
show_images = False

moco = dict(
    moco_dim=128,
    moco_k=65536,
    moco_m=0.999,
    moco_t=0.2,
    mlp=True,
)

features_dim = 512
num_head = 10
num_cluster = 10
clustering_head = [
    dict(classifier=dict(type="mlp", num_neurons=[features_dim, features_dim, num_cluster], last_activation="softmax"),
        feature_conv=None,
        num_cluster=num_cluster,
        loss_weight=dict(loss_cls=1),
        iter_start=100,
        iter_up=100,
        iter_down=100,
        iter_end=100,
        ratio_start=1.0,
        ratio_end=1.0,
        center_ratio=0.5,
        )]*num_head,    # multiple heads