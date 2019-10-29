"""
@author: Wei Han
Arrange information for complex scenes via dynamic clustering

Notes:
    Some utilities to make codes simple and clear.
    use tf.py_func or autograph
"""

import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable

import math
import numpy as np
import config as cf


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def get_optim(params, args, mode='preTrain', **kwargs):
    if mode == 'preTrain':
        lr = args.pretrain_lr
        # params = net.parameters()
        # for p in net.diverter.parameters():
        #     p.requires_grad = False
    elif mode == 'fineTune':
        lr = args.finetune_lr
        # params = net.parameters()
    else:
        raise NotImplementedError

    if args.opt_type == 'sgd':
        lr = get_learning_rate(lr, kwargs['epoch'])
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-3)
    elif args.opt_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-3)
    else:
        raise NotImplementedError

    return optimizer, lr


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


def get_learning_rate(init, epoch):
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    else:
        optim_factor = 0
    learning_rate = init*math.pow(0.2, optim_factor)
    #
    # learning_rate = init * (0.96 ** epoch)
    #
    # if epoch < 80:
    #     learning_rate = init
    # elif epoch < 120:
    #     learning_rate = init * 0.1
    # else:
    #     learning_rate = init * 0.01
    return learning_rate


# Strategy for numCluster
def get_numCluster(data):
    numInstance = data.shape[0]
    assert numInstance>= 10
    if numInstance > 20000:
        numClusters = 20
    elif numInstance > 15000:
        numClusters = 18
    elif numInstance > 10000:
        numClusters = 15
    elif numInstance > 5000:
        numClusters = 12
    elif numInstance > 1000:
        numClusters = 8
    elif numInstance > 300:
        numClusters = 6
    elif numInstance > 50:
        numClusters = 3
    else:
        numClusters = 2
    return numClusters


def get_all_assign(net, trainloader):
    required_data = []
    required_targets = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        batch_required_data = net.coarse(net.share(inputs))
        batch_required_targets = targets
        # _, batch_required_data = torch.max(batch_required_data, 1)
        # _, targets = torch.max(targets, 1)

        batch_required_data = batch_required_data.data.cpu().numpy() if cf.use_cuda else batch_required_data.data.numpy()
        batch_required_targets = batch_required_targets.data.cpu().numpy() if cf.use_cuda else batch_required_targets.data.numpy()
        required_data = stack_or_create(required_data, batch_required_data, axis=0)
        required_targets = stack_or_create(required_targets, batch_required_targets, axis=1)

    return required_data, required_targets


def stack_or_create(all_array, local_array, axis=1):
    if not torch.is_tensor(local_array):
        if len(all_array) != 0:
            if axis==1:
                all_array = np.hstack((all_array, local_array))
            else:
                all_array = np.vstack((all_array, local_array))
        else:
            all_array = local_array
    else:
        if len(all_array) != 0:
            all_array = torch.cat((all_array, local_array), axis)
        else:
            all_array = local_array
    return all_array


def adjust_learning_rate(optimizer, **kwargs):
    for param_group in optimizer.param_groups:
        if 'rate' in kwargs.keys():
            param_group['lr'] = param_group['lr'] * kwargs['rate']
        elif 'lr' in kwargs.keys():
            param_group['lr'] = kwargs['lr']
        else:
            raise AttributeError