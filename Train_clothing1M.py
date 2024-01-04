from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture


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


def linear_rampup(current, warm_up=20, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0., type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0., type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--sigma', default=2., type=float, help='weight for unsupervised loss')
parser.add_argument('--data_path', default='/clothing1m/', type=str,
                    help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=2000, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
step = 5
b_s = args.batch_size


# Training
def train(epoch, net, net2, optimizer, optimizer2, labeled_trainloader, unlabeled_trainloader):
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
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()
        targets_x = labels_x
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx = -torch.sum(torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0))

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:]) ** 2)
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter, step) * Lu
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        sys.stdout.write('\r')
        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                         % (epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()


def regroup_median_matrix(loss_data, label, step, b_s, loss_m, label_m, s_index):
    loss_v = loss_data
    loss = []

    for i in range(b_s):
        index = torch.nonzero(label_m == label[i]).squeeze()
        loss_x = loss_m[index]
        loss_norm = torch.exp(-loss_x ** 2 - loss_x)
        loss_norm = loss_norm / (loss_norm.sum())

        loss_x_i_1 = loss_m[s_index[i]]
        select_index = torch.multinomial(loss_norm, step * 8, replacement=False)
        select_loss = loss_x[select_index].detach()
        select_loss = select_loss.view(8, step)
        # select_loss_median,_ = torch.median(select_loss, dim=1)
        select_loss_mean = torch.mean(select_loss, dim=1)
        select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])
        select_loss = torch.median(select_loss)
        l_weight = select_loss / (loss_x_i_1 + 0.0000001)
        if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
            loss.append(loss_v[i].view(1))

        else:
            tmp = l_weight * loss_v[i]
            loss.append(tmp.view(1))
    loss = torch.cat(loss, dim=0)
    # loss=torch.median(group)

    return loss.mean()


def warmup(net, net2, optimizer, optimizer2, dataloader):
    net.train()
    for batch_idx, (inputs, labels, path, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        b_s = outputs.size(0)
        z = torch.randn(b_s).cuda()
        labels = F.one_hot(labels, args.num_class)
        mask = z * args.sigma + 1.0
        S = torch.sum(torch.softmax(outputs, dim=1) * labels, dim=1)
        loss_v = -torch.sum(torch.log_softmax(outputs, dim=1) * labels, dim=1)

        # loss_t2 = -2 * ((S - 1) / (S + 1)+ 1/3*((S - 1) / (S + 1))**3+1/5*((S - 1) / (S + 1))**5)
        loss_t2 = -2 * ((S - 1) / (S + 1) + 1 / 2 * ((S - 1) / (S + 1)) ** 2)

        loss_r = (loss_t2 - loss_t2.mean()) / torch.sqrt(loss_t2.var()) + loss_t2.mean()
        # loss_r=loss_t2
        cov = torch.mean(loss_r * loss_v) - torch.mean(loss_r) * torch.mean(loss_v)
        alpha = cov / loss_r.var()
        # if epoch<10:
        #     loss=loss_v.mean()
        # else:
        loss_vr = loss_v - alpha.detach() * (loss_r)
        loss_w = loss_vr.detach() / (loss_v.detach() + 0.00000001)
        loss_w[loss_v < loss_vr.mean()] = 1.
        loss_tmp = loss_w * loss_v
        loss = loss_tmp * mask

        # L = loss.mean()

        penalty = conf_penalty(outputs)
        L = loss.mean()
        L.backward()
        optimizer.step()
        optimizer2.step()
        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                         % (batch_idx + 1, args.num_batches, loss.mean().item(), penalty.item()))
        sys.stdout.flush()


def val(net, val_loader, k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
    if acc > best_acc[k - 1]:
        best_acc[k - 1] = acc
        print('| Saving Best Net%d ...' % k)
        save_point = './checkpoint/%s_net%d.pth.tar' % (args.id, k)
        torch.save(net.state_dict(), save_point)
    return acc


def test(net1, net2, test_loader):
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

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    return acc


def data_filter(epoch, model, loader):
    loss_m = []
    label = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, path, index) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            loss_m.append(loss)
            label.append(targets)
    loss_m = torch.cat(loss_m, dim=0)
    label = torch.cat(label, dim=0)
    return loss_m, label


def eval_train(epoch, model, model2):
    model.eval()
    num_samples = args.num_batches * args.batch_size
    losses = torch.zeros(num_samples)
    prob = torch.zeros(num_samples)
    paths = []
    n = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, path, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            outputs2 = model2(inputs)
            loss = CE(outputs, targets)
            pred_prob = torch.softmax(outputs, dim=1)
            pred_prob2 = torch.softmax(outputs2, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_label2 = torch.argmax(pred_prob2, dim=1)

            for b in range(inputs.size(0)):
                if pred_label[b] == targets[b] and pred_label2[b] == targets[b]:
                    if pred_prob[b, int(targets[b])] > 0. and pred_prob2[b, int(targets[b])] > 0.:
                        prob[n] = pred_prob[b, int(targets[b])] + 0.2
                else:
                    prob[n] = 0.
                losses[n] = loss[b]

                paths.append(path[b])
                n += 1

            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())

    return prob.numpy(), losses.numpy(), paths


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


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


def create_model(ema=False):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()

    return model


log = open('./checkpoint/%s.txt' % args.id, 'w')
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=5,
                                        num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model(ema=True)
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = WeightEMA(net1, net2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 200, 1e-6)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
max_acc1 = 0
max_acc2 = 0
max_net1 = net1
max_net2 = net2
best_acc = [0, 0]
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    # if epoch >= 40:
    #     lr /= 10
    # for param_group in optimizer1.param_groups:
    #     param_group['lr'] = lr

    if epoch < step:  # warm up
        train_loader = loader.run('warmup')
        # loss_m,label_m=data_filter(epoch,net1,train_loader)
        print('Warmup Net1')
        warmup(net1, net2, optimizer1, optimizer2, train_loader)

    else:
        pred2 = (prob2 > args.p_threshold)

        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, loss, paths=paths2)  # co-divide
        train(epoch, net1, net2, optimizer1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net1

    val_loader = loader.run('val')  # validation
    test_loader = loader.run('test')
    acc1 = val(net1, test_loader, 1)
    acc2 = val(net2, test_loader, 2)
    scheduler.step()
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n' % (epoch, acc1, acc2))
    log.flush()
    if acc1 >= max_acc1 or acc2 >= max_acc2:
        if acc1 >= max_acc1:
            max_acc1 = acc1
            max_net1 = net1
        if acc2 >= max_acc2:
            max_acc2 = acc2
            max_net2 = net2

        acc = test(max_net1, max_net2, test_loader)
        log.write('Test Accuracy:%.2f\n' % (acc))
        log.flush()
        test_acc1 = test(max_net1, max_net1, test_loader)
        log.write('Test Accuracy:%.2f\n' % (test_acc1))
        log.flush()
        test_acc2 = test(max_net2, max_net2, test_loader)
        log.write('Test Accuracy:%.2f\n' % (test_acc2))
        log.flush()
    print('\n==== net 2 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval_train')
    prob2, loss, paths2 = eval_train(epoch, net1, net2)

# test_loader = loader.run('test')
# net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar' % args.id))
# net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar' % args.id))
# acc = test(net1, net2, test_loader)
#
# log.write('Test Accuracy:%.2f\n' % (acc))
# log.flush()
