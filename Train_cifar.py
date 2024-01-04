import os
import os.path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100

from networks.ResNet import ResNet18, ResNet34
from common.tools import getTime, evaluate, predict_softmax, train
from common.NoisyUtil import Train_Dataset, dataset_split, Semi_Unlabeled_Dataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=5e-4)
parser.add_argument('--num_epochs', default=300, type=int)

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=0.9, type=float, help='data number percent')
parser.add_argument('--noise_type', default='instance', type=str)
parser.add_argument('--noise_rate', default=0.4, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--step', default=200, type=float, help='weight for unsupervised loss')
parser.add_argument('--n', default=6, type=float, help='weight for unsupervised loss')
parser.add_argument('--gpuid', default=0, type=float, help='weight for unsupervised loss')


args = parser.parse_args()
print(args)
os.system('nvidia-smi')

args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True
torch.cuda.set_device(args.gpuid)

def create_model(name="resnet18", input_channel=3, num_classes=10,ema=False):
    if(name == "resnet18"):
        model = ResNet18(num_classes)
    else:
        print("create ResNet34")
        model = ResNet34(num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    model.cuda()
    return model






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

if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':

    args.num_class = 10
    args.model_name = "resnet18"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':

    args.num_class = 100
    args.model_name = "resnet34"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)

train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, _ = dataset_split(train_set.data, np.array(train_set.targets), args.noise_rate, args.noise_type, args.data_percent, args.seed, args.num_class, False)
train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0, pin_memory=True, drop_last=True)
val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8,pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False,num_workers=0, pin_memory=True)

model = create_model(name=args.model_name, num_classes=args.num_class)
model1=create_model(name=args.model_name, num_classes=args.num_class,ema=True)
model2=create_model(name=args.model_name, num_classes=args.num_class,ema=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
ceriation = nn.CrossEntropyLoss(reduction='none').cuda()
train_ceriation = ceriation
optimizer2 = WeightEMA(model, model1)
best_val_acc = 0
best_test_acc = 0

for epoch in range(args.num_epochs):
    train(model, train_loader, optimizer, optimizer2, ceriation, epoch, args.step, args.n)

    scheduler.step()
    _, val_acc = evaluate(model, val_loader, ceriation, "Val Acc:")
    if val_acc>=best_val_acc:
        _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
        best_test_acc = test_acc


print(getTime(), "Best Test Acc:", best_test_acc)
