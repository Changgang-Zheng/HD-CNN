# Newest 2018.11.23 9:53:00

from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import config as cf
import numpy as np

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt

from dataset import get_all_dataLoders, get_dataLoder
from models import HD_CNN, Clustering
from losses import pred_loss, consistency_loss
from utils import get_optim, get_all_assign


parser = argparse.ArgumentParser(description='HD_CNN in PyTorch')
parser.add_argument('--dataset', default='cifar-100', type=str, help='Determine which dataset to be used')
parser.add_argument('--num_superclasses', default=9, type=int, help='The number of cluster centers')
parser.add_argument('--num_epochs_pretrain', default=200, type=int, help='The number of pre-train epoches')
parser.add_argument('--num_epochs_train', default=150, type=int, help='The number of train epoches')
parser.add_argument('--pretrain_batch_size', default=128, type=int, help='The batch size of pretrain')
parser.add_argument('--train_batch_size', default=256, type=int, help='The batch size of train')
parser.add_argument('--min_classes', default=2, type=int, help='The minimum of classes in one superclass')

parser.add_argument('--opt_type', default='sgd', type=str, help='Determine the type of the optimizer')
parser.add_argument('--pretrain_lr', default=0.1, type=float, help='The learning rate of pre-training')
parser.add_argument('--finetune_lr', default=0.001, type=float, help='The learning rate of inner model')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The probability of to keep')
parser.add_argument('--weight_consistency', default=1e1, type=float, help='The weight of coarse category consistency')
parser.add_argument('--gamma', default=5, type=float, help='The weight for u_k')#1.25

parser.add_argument('--resume_coarse', default=True, type=bool, help='resume coarse from checkpoint')
parser.add_argument('--resume_fines', default=True, type=bool, help='resume coarse & fines from checkpoint')
parser.add_argument('--resume_model', default=True, type=bool, help='resume the whole model from checkpoint')
args = parser.parse_args()

# Hyper Parameter settings
cf.use_cuda = torch.cuda.is_available()

trainloader, testloader, pretrainloader, validloader = get_all_dataLoders(args, valid=True, one_hot=False)
args.num_classes = 10 if args.dataset == 'cifar-10' else 100


def show(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

# Model
print('\nModel setup')
net = HD_CNN(args)
if cf.use_cuda:
    net.cuda()
    for i in range(args.num_superclasses):
        net.fines[i].cuda()
    cudnn.benchmark = True

function = Clustering(args)

# Pre-Training
def pretrain_coarse(epoch):
    net.share.train()
    net.coarse.train()

    param = list(net.share.parameters())+list(net.coarse.parameters())
    optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)

    print('\n==> Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(pretrainloader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU setting
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net.coarse(net.share(inputs)) # Forward Propagation

        loss = pred_loss(outputs, targets)
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        _, predicted = torch.max(outputs.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('Pre-train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (pretrainloader.dataset.train_data.shape[0]//args.pretrain_batch_size)+1,
                           loss.item(), 100.*correct.item()/num_ins))
        sys.stdout.flush()


def pretrain_fine(epoch, fine_id):
    net.fines[fine_id].train()

    optimizer, lr = get_optim(net.fines[fine_id].parameters(), args, mode='preTrain', epoch=epoch)

    print('==> Epoch #%d, LR=%.4f' % (epoch, lr))
    required_train_loader = get_dataLoder(args, classes=net.class_set[fine_id], mode='preTrain')
    predictor = net.fines[fine_id]
    for batch_idx, (inputs, targets) in enumerate(required_train_loader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets).long()

        outputs = predictor(net.share(inputs)) # Forward Propagation
        loss = pred_loss(outputs, targets)
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        num_ins = targets.size(0)
        _, outputs = torch.max(outputs, 1)
        correct = outputs.eq(targets.data).cpu().sum()
        acc = 100.*correct.item()/num_ins

        sys.stdout.write('\r')
        sys.stdout.write('Pre-train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (required_train_loader.dataset.train_data.shape[0]//args.pretrain_batch_size)+1,
                           loss.item(), acc))
        sys.stdout.flush()


def fine_tune(epoch):
    net.share.train()
    net.coarse.train()
    for i in range (args.num_superclasses):
        net.fines[i].train()

    param = list(net.share.parameters()) + list(net.coarse.parameters())
    for k in range(args.num_superclasses):
        param += list(net.fines[k].parameters())
    optimizer, lr = get_optim(param, args, mode='fineTune', epoch=epoch)

    print('\n==> fine-tune Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets).long()

        outputs, coarse_outputs = net(inputs, return_coarse=True)

        tloss = pred_loss(outputs, targets)
        closs = consistency_loss(coarse_outputs, t_k, weight=args.weight_consistency)
        loss = tloss + closs
        loss.backward()  # Backward Propagation
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct.item()/num_ins

        sys.stdout.write('\r')
        sys.stdout.write('Finetune Epoch [%3d/%3d] Iter [%3d/%3d]\t\t tloss: %.4f closs: %.4f Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_train, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                           tloss.item(), closs.item(), loss.item(), acc))
        sys.stdout.flush()


def test(epoch):
    net.share.eval()
    net.coarse.eval()
    for i in range(args.num_superclasses):
        net.fines[i].eval()

    num_ins = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings

        inputs, targets = Variable(inputs), Variable(targets).long()
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        num_ins += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100.*correct/num_ins
    print("\nValidation Epoch #%d\t\tAccuracy: %.2f%%" % (epoch, acc))
    return acc


def independent_test(epoch, mode=999):
    net.share.eval()
    net.coarse.eval()
    for i in range(args.num_superclasses):
        net.fines[i].eval()

    if mode == 999:
        required_loader = testloader
    else:
        required_loader = get_dataLoder(args, classes=net.class_set[mode], mode='Test')

    num_ins = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(required_loader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings

        inputs, targets = Variable(inputs), Variable(targets).long()
        if mode == 999:
            outputs = net.coarse(net.share(inputs))
        else:
            outputs = net.fines[mode](net.share(inputs))

        _, predicted = torch.max(outputs.data, 1)
        num_ins += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / num_ins
    print("\nValidation Epoch #%d\t\tAccuracy: %.2f%%" % (epoch, acc))
    return acc


# Pre-train
if not args.resume_model:
    if not args.resume_coarse:
        print('\n==> Pretrain the coarse model')
        best_acc = 0
        for epoch in range(args.num_epochs_pretrain):
            print('Pre-train Coarse')
            pretrain_coarse(epoch)
            acc = independent_test(epoch, mode=999)

            if acc >= best_acc:
                print('\nSaving Best model...\t\t\tTop1 = %.2f%%' % (acc))
                save_point = cf.var_dir + args.dataset
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(net.share.state_dict(), save_point + '/share_params.pkl')
                torch.save(net.coarse.state_dict(), save_point + '/coarse_params.pkl')
                best_acc = acc

    # No matter resume or not
    # Load the best parameter
    print('\n\n==> Get the best coarse from checkpoint')
    map_location = 'cuda' if cf.use_cuda else 'cpu'
    net_share_params = torch.load(cf.var_dir + args.dataset + '/share_params.pkl', map_location=map_location)
    net_coarse_params = torch.load(cf.var_dir + args.dataset + '/coarse_params.pkl', map_location=map_location)
    net.share.load_state_dict(net_share_params)
    net.coarse.load_state_dict(net_coarse_params)

    if not args.resume_fines:
        print('\n==> Doing the spectural clusturing')
        coarse_outputs, coarse_target = get_all_assign(net, validloader)
        coarse_output = np.argmax(coarse_outputs, 1)
        F = function.confusion_matrix(coarse_output, coarse_target)
        D = (1/2)*((np.identity(args.num_classes)-F)+np.transpose(np.identity(args.num_classes)-F))
        cluster_result = function.spectral_clustering(D, K=args.num_superclasses, gamma=10)


        print('\n==> Get Confusion coefficient')
        P_d = np.zeros((args.num_classes, args.num_superclasses))
        u_kj = np.zeros((args.num_classes, args.num_superclasses))
        for superclass in range(args.num_superclasses):
            P_d[cluster_result == superclass, superclass] = 1
        B_d_ik = np.dot(coarse_outputs, P_d)
        net.P_d = torch.from_numpy(P_d)
        for class_id in range(args.num_classes):
            u_kj[class_id] = np.mean(B_d_ik[validloader.dataset.train_labels == class_id], 0)
        u_t = 1 / (args.num_superclasses * args.gamma)
        P_o = (u_kj >= u_t).astype(np.float32)
        net.P_o = torch.from_numpy(P_o)
        for k in range(args.num_superclasses):
            net.class_set[k] = np.where(P_o[:,k] == 1)[0].astype(int)
        if cf.use_cuda:
            net.P_d = net.P_d.cuda()
            net.P_o = net.P_o.cuda()


        print('\n==> Get coarse category consistency')
        from collections import Counter
        t_k = torch.zeros((args.num_superclasses))
        if cf.use_cuda: t_k = t_k.cuda()
        c = Counter(trainloader.dataset.train_labels)
        for k in range(args.num_superclasses):
            t_k[k] = sum([c[i] for i in net.class_set[k]])
        t_k = t_k/torch.sum(t_k)

        print('\nSaving Parameters...' )
        state = {
            'net.P_d': net.P_d,
            'net.P_o': net.P_o,
            'net.class_set': net.class_set,
            't_k': t_k,
        }
        save_point = cf.model_dir + args.dataset
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + '/parameters.pkl')


        print('\n\n==> train fine independents')
        best_acc = np.zeros((args.num_superclasses,))
        for fine_id in range(args.num_superclasses):
            for epoch in range(args.num_epochs_pretrain):
                print('\nPre-train Fine #%d'%fine_id)
                pretrain_fine(epoch, fine_id)
                acc = independent_test(epoch, mode=fine_id)

                if acc >= best_acc[fine_id]:
                    print('\nSaving Best model...\t\t\tTop1 = %.2f%%' % (acc))
                    net_params = net.fines[fine_id].state_dict()
                    save_point = cf.var_dir + args.dataset
                    if not os.path.isdir(save_point):
                        os.mkdir(save_point)
                    torch.save(net_params, save_point + '/fine_%d.pkl'%fine_id)
                    best_acc[fine_id] = acc

    else:
        # If resume, load the corresponding parameters
        print('\n\n==> Get parameters from checkpoint')
        map_location = 'cuda' if cf.use_cuda else 'cpu'
        checkpoint = torch.load(cf.model_dir + args.dataset + '/parameters.pkl', map_location=map_location)
        net.P_d = checkpoint['net.P_d']
        net.P_o = checkpoint['net.P_o']
        net.class_set = checkpoint['net.class_set']
        t_k = checkpoint['t_k']

    # No matter resume or not
    # Load the best parameter
    print('\n\n==> Get the best fines from checkpoint')
    for fine_id in range(args.num_superclasses):
        map_location = 'cuda' if cf.use_cuda else 'cpu'
        net_fine_params = torch.load(cf.var_dir + args.dataset + '/fine_%d.pkl'%fine_id, map_location=map_location)
        net.fines[fine_id].load_state_dict(net_fine_params)


    print('\n\n==> fine_tune the model')
    best_acc = 0
    for epoch in range(1, args.num_epochs_train + 1):
        fine_tune(epoch)
        acc = test(epoch)

        if acc >= best_acc:
            print('\nSaving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            save_point = cf.model_dir + args.dataset
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(net, save_point + '/over_all_model.pkl')
            best_acc = acc

# No matter resume or not
# Load the best whole model
map_location = 'cuda' if cf.use_cuda else 'cpu'
net = torch.load(cf.model_dir + args.dataset + '/over_all_model.pkl', map_location=map_location)


# Final test
net.share.eval()
net.coarse.eval()
for i in range(args.num_superclasses):
    net.fines[i].eval()

num_ins = 0
correct = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    if cf.use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
    inputs, targets = Variable(inputs), Variable(targets).long()

    outputs = net(inputs)

    _, predicted = torch.max(outputs.data, 1)
    num_ins += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum().item()
acc = 100. * correct / num_ins
print("\nThe final performance @Accuracy: %.2f%%" % acc)