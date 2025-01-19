from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import torchvision.models as models

def regroup_median(input, target, step, b_s):
    loss_v = -torch.sum(F.log_softmax(input, dim=1) * target, dim=1)
    # print(loss_v)
    bsz = input.size(0)

    if bsz != b_s:
        index = np.random.randint(0, bsz, b_s - bsz)
        s_loss = loss_v[index]
        loss_v = torch.cat([loss_v, s_loss], dim=0)

    item = -(loss_v ** 2 + loss_v)

    loss_norm = torch.exp(item.detach())
    loss_norm = loss_norm / (loss_norm.sum())

    loss = []

    for i in range(b_s):
        select_index = torch.multinomial(loss_norm, step * 8, replacement=False)
        select_loss = loss_v[select_index].detach()
        select_loss = select_loss.view(8, step)
        # select_loss_median,_ = torch.median(select_loss, dim=1)
        select_loss_mean = torch.mean(select_loss, dim=1)
        select_loss = torch.cat([loss_v[i].detach().view(1), select_loss_mean])
        select_loss = torch.median(select_loss)
        if loss_v[i] <= select_loss or loss_v[i] == 0:
            loss.append(loss_v[i].view(1))

        else:
            tmp = (select_loss / loss_v[i].detach()) * loss_v[i]
            loss.append(tmp.view(1))
    loss = torch.cat(loss, dim=0)
    # loss=torch.median(group)

    return loss.mean()


parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0., type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0., type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='', type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='data/webvision/', type=str, help='path to dataset')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer,optimizer2, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        # with torch.no_grad():
            # label co-guessing of unlabeled samples
        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()
        targets_x=labels_x
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b
        # print('point')
        torch.cuda.empty_cache()
        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx = -torch.sum(torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0))


        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter,warm_step) * Lu
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        sys.stdout.write('\r')
        sys.stdout.write('Webvision | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                         % (epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

def regroup_median_matrix(loss_data,label, step, b_s,loss_m,label_m,s_index):
    loss_v = loss_data
    loss = []

    for i in range(b_s):
        index = torch.nonzero(label_m == label[i]).squeeze()
        loss_x = loss_m[index]
        size=loss_x.size(0)
        step=int(size*0.2)
        # print(loss_x)
        loss_norm=torch.exp(-loss_x**2-loss_x)
        loss_norm=loss_norm/(loss_norm.sum()+0.00000001)
        # print(loss_norm)
        loss_x_i_1=loss_m[s_index[i]]
        select_index = torch.multinomial(loss_norm, step*4, replacement=False)
        select_loss = loss_x[select_index].detach()
        select_loss = select_loss.view(4, step)
        # select_loss_median,_ = torch.median(select_loss, dim=1)
        select_loss_mean = torch.mean(select_loss, dim=1)
        select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])
        select_loss = torch.median(select_loss)
        l_weight=select_loss/(loss_x_i_1+0.0000001)
        if loss_v[i].detach() <= loss_v[i].detach()*l_weight or loss_v[i] == 0:
            loss.append(loss_v[i].view(1))

        else:
            tmp = l_weight* loss_v[i]
            loss.append(tmp.view(1))
    loss = torch.cat(loss, dim=0)
    # loss=torch.median(group)

    return loss.mean()
def warmup(epoch, net,net2, optimizer,optimizer2, dataloader,loss_m,label_m):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CE(outputs, labels)
        # targets = F.one_hot(labels, 50)
        batch_size=loss.size(0)
        if epoch==0:
            loss=torch.mean(loss)
        else:
            loss = regroup_median_matrix(loss,labels,100,batch_size,loss_m,label_m,index)
        # penalty = conf_penalty(outputs)
        L = loss  # + penalty

        L.backward()
        optimizer.step()
        optimizer2.step()
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                         % (args.id, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()
def filter_data(net,dataloader):
    loss_m = []
    label=[]

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = CE(outputs, labels)
            loss_m.append(loss.detach())
            label.append(labels)
    loss_m=torch.cat(loss_m,dim=0)
    label=torch.cat(label,dim=0)
    # for i in range(14):
    #     index=torch.nonzero(label==i).squeeze()
    #     loss_data.append(loss_m[index])
    # loss_data=torch.cat(loss_data,dim=0)
    return loss_m,label
class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            # fix the error 'RuntimeError: result type Float can't be cast to the desired output type Long'
            # print(param.type())
            if param.type() == 'torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)

def test(epoch, net1, net2, test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            acc_meter.add(outputs, targets)
    accs = acc_meter.value()
    return accs


def eval_train(epoch, model,model2,loss_data):
    model.eval()

    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset))
    prob=torch.zeros(len(eval_loader.dataset))
    label = []

    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            outputs2 = model2(inputs)
            loss = CE(outputs, targets)

            pred_prob=torch.softmax(outputs,dim=1)
            pred_prob2 = torch.softmax(outputs2, dim=1)
            pred_label=torch.argmax(pred_prob,dim=1)
            pred_label2 = torch.argmax(pred_prob2, dim=1)
            loss_data[index]=loss
            label.append(targets)
            for b in range(inputs.size(0)):
                if pred_label[b]==targets[b] and pred_label2[b]==targets[b]:
                    if pred_prob[b,int(targets[b])]>0. and pred_prob2[b,int(targets[b])]>0.:
                        prob[n]=pred_prob[b,int(targets[b])]+0.2
                else:
                    prob[n]=0.
                losses[n] = loss[b]


                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    label=torch.cat(label,dim=0)
    return prob.numpy(),losses.numpy(),loss_data,label


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model(ema=False):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, args.num_class)
    # model = InceptionResNetV2(num_classes=args.num_class)
    
    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    
    return model


stats_log = open('./checkpoint/%s' % (args.id) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s' % (args.id) + '_acc.txt', 'w')

warm_step=200

loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_workers=5,data_number=0, root_dir=args.data_path,
                                         log=stats_log, num_class=args.num_class)
# loader_filter = dataloader.webvision_dataloader(batch_size=25, num_workers=5,data_number=100, root_dir=args.data_path,
#                                          log=stats_log, num_class=args.num_class)

print('| Building net')
net1 = create_model()
net2 = create_model(ema=True)
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = WeightEMA(net1, net2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer1,args.num_epochs)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
eval_loader = loader.run('eval_train')
loss_m=torch.zeros(len(eval_loader.dataset)).cuda()
for epoch in range(args.num_epochs + 1):


    eval_loader = loader.run('eval_train')

    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')
    if epoch==0:
        prob1,all_loss[0],loss_m,label_m=eval_train(epoch,net1,net2,loss_m)

    if epoch < warm_step:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1,net2, optimizer1,optimizer2, warmup_trainloader,loss_m,label_m)

    else:
        pred1 = (prob1 > args.p_threshold)


        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
        train(epoch, net1, net2, optimizer1,optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net1

    web_acc = test(epoch, net1, net2, web_valloader)
    # imagenet_acc = test(epoch,net1,net2,imagenet_valloader)
    scheduler.step()
    # print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%)\n" % (
        epoch, web_acc[0], web_acc[1]))
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%)\n' % (epoch, web_acc[0], web_acc[1]))
    test_log.flush()
    web_acc = test(epoch, net1, net1, web_valloader)
    # imagenet_acc = test(epoch,net1,net2,imagenet_valloader)

    # print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%)\n" % (
        epoch, web_acc[0], web_acc[1]))
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%)\n' % (epoch, web_acc[0], web_acc[1]))
    test_log.flush()
    web_acc = test(epoch, net2, net2, web_valloader)
    imagenet_acc = test(epoch,net2,net2,imagenet_valloader)

    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    # print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%)\n" % (
    #     epoch, web_acc[0], web_acc[1]))
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%) \n' % (epoch, web_acc[0], web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()
    print('\n==== net 1 evaluate training data loss ====')
    prob1, all_loss[0],loss_m,label_m= eval_train(epoch,net1,net2, loss_m)
    # print('\n==== net 2 evaluate training data loss ====')
    # prob2, all_loss[1] = eval_train(net2, all_loss[1])
    # torch.save(all_loss, './checkpoint/%s.pth.tar' % (args.id))

