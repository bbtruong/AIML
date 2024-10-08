import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # model architecture
    parser.add_argument('--model', default='ResNet50', choices=('ResNet18', 'ResNet50'))
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='baseline')

    # data loading
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    # training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=False)

    # optimizer and learning rate schedule
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default = 'ADAM', choices=('SGD', 'ADAM'))
    parser.add_argument('--decay_type', default='cosine_warmup', choices=('step', 'step_warmup', 'cosine_warmup'))

    args = parser.parse_args()
    return args
