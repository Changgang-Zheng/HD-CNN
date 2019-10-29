use_cuda = True

model_dir = './logging/model_dir/'
var_dir = './logging/var_dir/'
log_dir = './logging/log_dir/'

mean = {
    'cifar-10': (0.4914, 0.4822, 0.4465),
    'cifar-100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar-10': (0.2023, 0.1994, 0.2010),
    'cifar-100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
