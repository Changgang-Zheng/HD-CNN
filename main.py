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

from dataset import get_pretrain_dataLoders, get_dataLoder
from models import HD_CNN, clustering
from losses import diverter_loss, pred_loss, recon_loss, center_loss, fine_tuning_loss
from utils import get_optim, get_hms, adjust_learning_rate
from copy import deepcopy

#============================================================================= remember to chage the data address: 'root_path' in dataset.py

parser = argparse.ArgumentParser(description='HD_CNN in PyTorch')
parser.add_argument('--dataset', default='cifar-10', type=str, help='Determine which dataset to be used ================ cifar-10/cifar-100')
parser.add_argument('--meta_method', default='kmeans', type=str, help='Determine which method to cluste')
parser.add_argument('--num_superclass', default=2, type=int, help='The number of cluster centers ======================= 2/9')
parser.add_argument('--express_dim', default=128, type=int, help='The dimensionality of express vector')

parser.add_argument('--num_epochs_pretrain', default=1, type=int, help='The number of pre-train steps ================== when final test, make it bigger')
parser.add_argument('--num_test', default=3, type=int, help='The number of test batch steps in a epoch ================= when final test, make it bigger')
parser.add_argument('--num_epochs_outer', default=150, type=int, help='The number of encoder training steps')
parser.add_argument('--num_steps_inner', default=100, type=int, help='The number of predictor training steps')
parser.add_argument('--num_fine_classes', default=10, type=int, help='The number of fine classes ======================= 10/100')
parser.add_argument('--train_batch_size', default=128, type=int, help='The batch size of train')
parser.add_argument('--min_classes', default=2, type=int, help='The minimum of classes in one superclass')


parser.add_argument('--opt_type', default='sgd', type=str, help='Determine the type of the optimizer')
parser.add_argument('--pretrain_lr', default=0.1, type=float, help='The learning rate of pre-training')
parser.add_argument('--inner_lr', default=0.1, type=float, help='The learning rate of inner model')
parser.add_argument('--outer_lr', default=0.1, type=float, help='The learning rate of outer model')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The probability of to keep')
parser.add_argument('--weight_recon', default=1e-6, type=float, help='The weight of reconstruction loss')
parser.add_argument('--weight_class_center', default=1e-2, type=float, help='The weight of class center loss')
parser.add_argument('--weight_branch_center', default=1e-2, type=float, help='The weight of meta-learner loss')
parser.add_argument('--stepsize_class_center', default=1e-3, type=float, help='The step size of class center update')

# parser.add_argument('--display_step', default=20, type=int, help='The number of training steps to display and log')
# parser.add_argument('--with_pretrain', default=True, type=bool, help='Whether should do the pre-train or not')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--with_pretrain', default=True, type=bool, help='Pre-train or not')
parser.add_argument('--with_test', default=True, type=bool, help='test or not')
parser.add_argument('--testOnly', default=False, type=bool, help='Test mode with the saved model')
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

# Hyper Parameter settings
cf.use_cuda = torch.cuda.is_available()
best_acc = 0

trainloader, testloader, trainbase, validloader = get_pretrain_dataLoders(args, valid=True, one_hot=True)
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
function = clustering(args)

# Pre-Training
def pretrain_clustering(epoch, mode, cluster_result = None):
    net.share.train()
    net.croase.train()

    train_loss = 0
    optimizer_share, lr = get_optim(net.share, args, mode='preTrain', epoch=epoch)
    optimizer_croase, lr = get_optim(net.croase, args, mode='preTrain', epoch=epoch)

    if mode == 'clustering' :
        Data = enumerate(trainloader)
        print('\ntrain data-loader activated')
    else:
        print('---------------- Warning! no mode is activated ----------------\n')
    print('=> pre-train Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, targets) in Data:
        if batch_idx>=args.num_test:
            break
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer_share.zero_grad()
        optimizer_croase.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net.croase.independent(net.share.encoder(inputs)) # Forward Propagation
        if batch_idx==0:
            total_outputs=outputs
            total_targets=targets
        else:
            total_outputs=torch.cat((total_outputs,outputs), 0)
            total_targets=torch.cat((total_targets,targets), 0)

        loss = pred_loss(outputs, targets)

        loss.backward()  # Backward Propagation
        optimizer_share.step() # Optimizer update
        optimizer_croase.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, targets = torch.max(targets.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('Pre-train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                           loss.item(), 100.*correct.item()/num_ins))
        sys.stdout.flush()

    print('\n=> valid epoch begining for clustering')
    if mode == 'clustering':
        clustering_data = enumerate(validloader)
        print('valid data-loader activated')
        for batch_idx, (inputs, targets) in clustering_data:
            if batch_idx>=args.num_test:
                break
            if cf.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            optimizer_share.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = net.croase.independent(net.share.encoder(inputs)) # Forward Propagation
            if batch_idx==0:
                total_outputs=outputs
                total_targets=targets
            else:
                total_outputs=torch.cat((total_outputs,outputs), 0)
                total_targets=torch.cat((total_targets,targets), 0)
            loss = pred_loss(outputs, targets)

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets.data, 1)
            num_ins = targets.size(0)
            correct = predicted.eq(targets.data).cpu().sum()
            acc = 100.*correct.item()/num_ins
            sys.stdout.write('\r')
            sys.stdout.write('valid epoch begining [%3d/%3d] Iter [%3d/%3d]\t\t Accuracy: %.3f%%'
                             %(epoch, args.num_epochs_pretrain, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                               acc))
            sys.stdout.flush()

        print('\nSaving model...\t\t\tTop1 = %.2f%%' % (acc))
        share_params = net.share.state_dict()
        croase_params = net.croase.state_dict()
        save_point = cf.model_dir + args.dataset
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(share_params, save_point + '/share_params.pkl')
        torch.save(croase_params, save_point + '/croase_params.pkl')
    return total_outputs , total_targets


def pretrain_fine(epoch, cluster_result = None, u_kj = None):
    save_point = cf.model_dir + args.dataset
    net.share.train()
    net.croase.train()
    for i in range (args.num_superclass):
        net.fines[i].train()

    train_loss = 0
    optimizer_share, lr = get_optim(net.share, args, mode='preTrain', epoch=epoch)
    optimizer_croase, lr = get_optim(net.croase, args, mode='preTrain', epoch=epoch)
    optimizer_fine = {}
    for k in range (args.num_superclass):
        for para in list(net.fines[k].parameters())[:-9]:
            para.requires_grad=False
        optimizer_fine[k], lr = get_optim(net.fines[k], args, mode='preTrain', epoch=epoch)
    if epoch == 1 :
        print('\nprevious model activated')
        net.share.load_state_dict(torch.load(save_point + '/share_params.pkl'))
        net.croase.load_state_dict(torch.load(save_point + '/croase_params.pkl'))
        for i in range (args.num_superclass):
            net.fines[i].load_state_dict(torch.load(save_point + '/croase_params.pkl'))

    print('\ntrain data-loader activated')
    Data = enumerate(trainloader)
    for batch_idx, (inputs, targets) in Data:
        if batch_idx>=args.num_test:
            break
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer_share.zero_grad()
        optimizer_croase.zero_grad()
        for i in range (args.num_superclass):
            optimizer_fine[i].zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        share = net.share.encoder(inputs)
        outputs = net.croase.independent(share) # Forward Propagation
        fine_out = {}
        fine_target={}
        fine_result={}
        # ==================== divide the fine result =====================
        for k in range (args.num_superclass):
            fine_out[k]=[]
            fine_target[k]=[]
            fine_result[k] = net.fines[k].independent(share)
        for i in range (np.shape(targets)[0]):
            for j in range (args.num_fine_classes):
                if j == torch.max(targets, 1)[1][i]:
                    for k in range (args.num_superclass):
                        if cluster_result[j] == k or u_kj[j, k] >= u_t:
                            if np.shape(fine_out[k])[0] == 0:
                                fine_out[k] = torch.reshape(fine_result[k][i,:],[1,10])
                                fine_target[k] = torch.reshape(targets[i,:],[1,10])
                            else:
                                fine_out[k] = torch.cat((fine_out[k],torch.reshape(fine_result[k][i,:],[1,10])),0)
                                fine_target[k] = torch.cat((fine_target[k],torch.reshape(targets[i,:],[1,10])),0)


        fine_loss={}
        for k in range (args.num_superclass):
            fine_loss[k]= pred_loss(fine_out[k], fine_target[k])
            if k == 0:
                loss = fine_loss[k]
            else:
                loss += fine_loss[k]
        loss.backward()  # Backward Propagation
        for k in range (args.num_superclass):
            optimizer_fine[k].step() # Optimizer update
        train_loss += loss.item()
        for k in range (args.num_superclass):
            if k == 0:
                predicted = torch.max(fine_out[k].data, 1)[1]
                targets = torch.max(fine_target[k].data, 1)[1]
            else:
                predicted = torch.cat((predicted , torch.max(fine_out[k].data, 1)[1]),0)
                targets = torch.cat((targets , torch.max(fine_target[k].data, 1)[1]),0)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct.item()/num_ins
        sys.stdout.write('\r')
        sys.stdout.write('Pre-train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                           loss.item(), acc))
        sys.stdout.flush()
    return acc

def fine_tune(epoch, cluster_result = None, u_kj = None):
    save_point = cf.model_dir + args.dataset

    net.share.train()
    net.croase.train()
    for i in range (args.num_superclass):
        net.fines[i].train()

    train_loss = 0
    optimizer_share, lr = get_optim(net.share, args, mode='preTrain', epoch=epoch)
    optimizer_croase, lr = get_optim(net.croase, args, mode='preTrain', epoch=epoch)
    optimizer_fine = {}
    for i in range (args.num_superclass):
        optimizer_fine[i], lr = get_optim(net.fines[i], args, mode='preTrain', epoch=epoch)

    if epoch == 1 :
        print('\nprevious model activated')
        net.load_state_dict(torch.load(save_point + '/over_all_model.pkl'))
        for i in range (args.num_superclass):
            net.fines[i].load_state_dict(torch.load(save_point + '/fine'+str(i)+'.pkl'))
    print('\ntrain data-loader activated')
    Data = enumerate(trainloader)
    for batch_idx, (inputs, targets) in Data:
        if batch_idx>=args.num_test:
            break
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer_share.zero_grad()
        optimizer_croase.zero_grad()
        for i in range (args.num_superclass):
            optimizer_fine[i].zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        share = net.share.encoder(inputs)
        outputs = net.croase.independent(share) # Forward Propagation
        fine_out = {}
        for i in range (args.num_superclass):
            fine_out[i] = net.fines[i].independent(share) # output of each fine layers
        # ==================== prepare B_o_ik =======================
        B_o_ik = np.zeros((np.shape(outputs)[0],args.num_superclass))
        for i in range (np.shape(outputs)[0]):
            for j in range (args.num_fine_classes):
                for k in range (args.num_superclass):
                    if cluster_result[j] == k or u_kj[j, k] >= u_t:
                        B_o_ik[i,k] += outputs[i,j]
        # ================== prepare fine_result ====================
        fine_result = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
        for i in range (np.shape(fine_out[1])[0]):
            result_upper = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
            result_lower = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
            for k in range (args.num_superclass):
                result_upper[i,:] += B_o_ik[i,k]*fine_out[k][i,:]
                result_lower[i,:] += fine_out[k][i,:]
            for j in range (args.num_fine_classes):
                fine_result[i,j]= result_upper[i,j]/result_lower[i,j]
        # ====================== prepare t_k ========================
        t_k = torch.zeros(args.num_superclass)
        total = 0
        for k in range (args.num_superclass):
            for i in range (np.shape(fine_out[1])[0]):
                for j in range (args.num_fine_classes):
                    if cluster_result[j] == k or u_kj[j, k] >= u_t:
                        t_k[k] += 1
        t_k = t_k/torch.sum(t_k)
        # =================== finish preperation ====================
        loss = fine_tuning_loss( fine_result, torch.max(targets, 1)[1], t_k, B_o_ik, args )
        loss.backward()  # Backward Propagation
        for i in range (args.num_superclass):
            optimizer_fine[i].step() # Optimizer update
        train_loss += loss.item()
        _, predicted = torch.max(fine_result.data, 1)
        _, targets = torch.max(targets.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct.item()/num_ins
        sys.stdout.write('\r')
        sys.stdout.write('Pre-train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                           loss.item(), acc))
        sys.stdout.flush()
    return acc




def test(epoch, cluster_result = None, u_kj = None):
    train_loss = 0
    save_point = cf.model_dir + args.dataset
    if epoch == 1 :
        print('\nprevious model activated')
        net.load_state_dict(torch.load(save_point + '/over_all_model.pkl'))
        for i in range (args.num_superclass):
            net.fines[i].load_state_dict(torch.load(save_point + '/fine'+str(i)+'.pkl'))
    print('\ntest data-loader activated')
    Data = enumerate(testloader)
    for batch_idx, (inputs, targets) in Data:
        if batch_idx>=args.num_test:
            break
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings

        inputs, targets = Variable(inputs), Variable(targets)
        share = net.share.encoder(inputs)
        outputs = net.croase.independent(share) # Forward Propagation
        fine_out = {}
        for i in range (args.num_superclass):
            fine_out[i] = net.fines[i].independent(share) # output of each fine layers
        # ==================== prepare B_o_ik =======================
        B_o_ik = np.zeros((np.shape(outputs)[0],args.num_superclass))
        for i in range (np.shape(outputs)[0]):
            for j in range (args.num_fine_classes):
                for k in range (args.num_superclass):
                    if cluster_result[j] == k or u_kj[j, k] >= u_t:
                        B_o_ik[i,k] += outputs[i,j]
        # ================== prepare fine_result ====================
        fine_result = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
        for i in range (np.shape(fine_out[1])[0]):
            result_upper = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
            result_lower = torch.zeros(np.shape(fine_out[1])[0],args.num_fine_classes)
            for k in range (args.num_superclass):
                result_upper[i,:] += B_o_ik[i,k]*fine_out[k][i,:]
                result_lower[i,:] += fine_out[k][i,:]
            for j in range (args.num_fine_classes):
                fine_result[i,j]= result_upper[i,j]/result_lower[i,j]
        # ====================== prepare t_k ========================
        t_k = torch.zeros(args.num_superclass)
        total = 0
        for k in range (args.num_superclass):
            for i in range (np.shape(fine_out[1])[0]):
                for j in range (args.num_fine_classes):
                    if cluster_result[j] == k or u_kj[j, k] >= u_t:
                        t_k[k] += 1
        t_k = t_k/torch.sum(t_k)
        # =================== finish preperation ====================
        loss = fine_tuning_loss( fine_result, torch.max(targets, 1)[1], t_k, B_o_ik, args )
        train_loss += loss.item()
        _, predicted = torch.max(fine_result.data, 1)
        _, targets = torch.max(targets.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct.item()/num_ins
        sys.stdout.write('\r')
        sys.stdout.write('test Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch, args.num_epochs_pretrain, batch_idx+1, (trainloader.dataset.train_data.shape[0]//args.train_batch_size)+1,
                           loss.item(), acc))
        sys.stdout.flush()
    return acc

# Pre-train
if args.with_pretrain:
    print('\n========== Training the original model ==========')
    for epoch in range(1, args.num_epochs_pretrain + 1):
        total_outputs , total_targets = pretrain_clustering(epoch,'clustering')
        if epoch == args.num_epochs_pretrain :
            print('\n========== Doing the spectural clusturing ==========')
            total_output = np.argmax(total_outputs.detach(), 1)
            total_target = np.argmax(total_targets.detach(), 1)
            #============================ clustering ==================================
            F = function.acc(total_output, total_target)
            D = (1/2)*((np.identity(args.num_fine_classes)-F)+np.transpose(np.identity(args.num_fine_classes)-F))
            cluster_result = function.spectral_clustering(D, K=args.num_superclass, gamma=10)
        #============================ clustering Finished =============================

    #============================= Overlappin
    # g coarse categories ======================
    print('\n\n========== Overlapping coarse categories ==========')
    u_kj=np.zeros((args.num_fine_classes, args.num_superclass))
    B_d_ik = np.zeros((np.shape(total_outputs)[0],args.num_superclass))
    B_o_ik = np.zeros((np.shape(total_outputs)[0],args.num_superclass))
    u_t= 1/(args.num_superclass*5)

    for i in range (np.shape(total_outputs)[0]):
        for j in range (args.num_fine_classes):
            for k in range (args.num_superclass):
                if cluster_result[j] == k:
                    B_d_ik[i,k] += total_outputs[i,j]
    for j in range (args.num_fine_classes):
        for k in range (args.num_superclass):
            B_sum = 0
            S_jf = 0
            for n in range (np.shape(total_outputs)[0]):
                if total_output[n]== j:
                    B_sum += B_d_ik[n,k]
                    S_jf += 1
            u_kj[j,k] = B_sum/S_jf
    for i in range (np.shape(total_outputs)[0]):
        for j in range (args.num_fine_classes):
            for k in range (args.num_superclass):
                if cluster_result[j] == k or u_kj[j, k] >= u_t:
                    B_o_ik[i,k] += total_outputs[i,j]


    print('\n\n============ train fine independents ============')
    for epoch in range(1, args.num_epochs_pretrain + 1):
        acc = pretrain_fine(epoch, cluster_result, u_kj)
        if acc > best_acc:
            print('\nSaving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            net_params = net.state_dict()
            save_point = cf.model_dir + args.dataset
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(net_params, save_point + '/over_all_model.pkl')
            for i in range (args.num_superclass):
                net_fine = net.fines[i].state_dict()
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(net_fine, save_point + '/fine'+str(i)+'.pkl')
            best_acc = acc

    print('\n\n================ fine_tune the model =================')
    for epoch in range(1, args.num_epochs_pretrain + 1):
        acc = fine_tune(epoch, cluster_result, u_kj)
        if acc > best_acc:
            print('\nSaving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            net_params = net.state_dict()
            save_point = cf.model_dir + args.dataset
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(net_params, save_point + '/over_all_model.pkl')
            for i in range (args.num_superclass):
                net_fine = net.fines[i].state_dict()
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(net_fine, save_point + '/fine'+str(i)+'.pkl')
            best_acc = acc
    print('\n\n================ pre-train finished ==================')


if args.with_test:
    print('\n\n================== test the model ====================')
    for epoch in range(1, args.num_test + 1):
        test(epoch, cluster_result, u_kj)