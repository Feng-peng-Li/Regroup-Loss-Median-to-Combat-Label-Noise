import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def Median_filter(input, target):
    input = -F.log_softmax(input, dim=-1)
    loss_v = torch.sum(input * target, dim=1)
    bs = input.size(0)

    for i in range(bs):
        index = np.random.randint(0, bs, 5)
        s_loss = loss_v[index]
        loss_v[i] = s_loss.median()
    return loss_v.mean()


def regroup_median_matrix(loss_data, label, step,n, b_s, loss_m, label_m, s_index):
    loss_v = loss_data
    loss = []

    for i in range(b_s):
        index = torch.nonzero(label_m == label[i]).squeeze()

        loss_x = loss_m[index]

        loss_norm = torch.exp(-loss_x ** 2 - loss_x)
        loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
        # print(loss_norm)
        loss_x_i_1 = loss_m[s_index[i]]
        select_index = torch.multinomial(loss_norm, step * n, replacement=False)
        select_loss = loss_x[select_index].detach()
        select_loss = select_loss.view(n, step)
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
    return loss


def filter(model, train_loader, ceriation):
    label_m = []
    label_p = []
    loss_m = torch.zeros(50000).cuda()
    with torch.no_grad():
        for i, (images, labels, index) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if labels.dim() > 1:
                loss_m[index] = -torch.mean(torch.sum(torch.log_softmax(output, dim=1) * labels, dim=1), dim=0).float()
            else:
                loss_m[index] = ceriation(output, labels.long())
            label_m.append(labels)
            label_p.append(torch.argmax(torch.softmax(output, dim=1), dim=1))
    label_m = torch.cat(label_m, dim=0)
    label_p = torch.cat(label_p, dim=0)
    return loss_m, label_m, label_p.cpu().numpy()


def train(model, train_loader, optimizer, optimizer2, ceriation, epoch,step,n):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    loss_m, label_m, _ = filter(model, train_loader, ceriation)
    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        logist = model(images)
        b_s = logist.size(0)
        loss_data = ceriation(logist, labels.long())
        loss = regroup_median_matrix(loss_data, labels, step,n, b_s, loss_m, label_m, index).mean()
        acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item()



def evaluate(model, eva_loader, ceriation, prefix, ignore=-1):
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist = model(images)

            loss = ceriation(logist, labels.long()).mean()
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return losses.avg, top1.avg.to("cpu", torch.float).item()


def evaluateWithBoth(model1, model2, eva_loader, prefix):
    top1 = AverageMeter('Acc@1', ':3.2f')
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist1 = model1(images)
            logist2 = model2(images)
            logist = (F.softmax(logist1, dim=1) + F.softmax(logist2, dim=1)) / 2
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return top1.avg.to("cpu", torch.float).item()


def predict(predict_loader, model):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _, _ in predict_loader:
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1 = model(images1)
                logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()
