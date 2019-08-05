import os
import torch
import numpy as np
import torchvision
from models import *
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from wavelet_haar import wavelet_transform
import torchvision.transforms as transforms


class Runner(object):
    def __init__(self, root_path='/home/ubuntu/data1.5TB/cifar',
                 model=WaveletVGG, wavelet_level=0, batch_size=128, lr=0.1, name="vgg"):
        self.root_path = root_path

        self.model = model
        self.wavelet_level = wavelet_level

        self.batch_size = batch_size
        self.lr = lr
        self.name = "{}_{}".format(name, self.wavelet_level)
        self.checkpoint_path = Tools.new_dir("./checkpoint/{}".format(self.name))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = self._build(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_loader, self.test_loader = self._data()
        pass

    def info(self):
        Tools.print("model={} wavelet level={} batch size={} lr={} name={}".format(
            str(self.model), self.wavelet_level, self.batch_size, self.lr, self.name))
        pass

    def _data(self):
        Tools.print('==> Preparing data..')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.CIFAR10(self.root_path, train=True, download=True, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=True, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        return _train_loader, _test_loader

    def _wavelet_data(self, inputs, wavelet_level=1):
        result = {"Input": inputs}

        if self.wavelet_level == 0:
            return result

        inputs = np.asarray(inputs)

        # 获得小波数据
        wavelets = []
        for input_data in inputs:
            wavelet_one = []
            for input_data_channel in input_data:
                wavelet = wavelet_transform(input_data_channel, wavelet_level)
                wavelet_one.append(wavelet)
                pass

            wavelet_result = {}
            for key in wavelet_one[0]:
                shape = wavelet_one[0][key].shape
                wavelet_result_key = np.zeros(shape=(len(wavelet_one), shape[0], shape[1]), dtype=np.float32)
                for wavelet_one_index, wavelet_one_i in enumerate(wavelet_one):
                    wavelet_result_key[wavelet_one_index, :, :] = wavelet_one_i[key]
                    pass
                wavelet_result[key] = wavelet_result_key
                pass

            wavelets.append(wavelet_result)
            pass

        for key in wavelets[0]:
            shape = wavelets[0][key].shape
            wavelet_result_key = np.zeros(shape=(len(wavelets), shape[0], shape[1], shape[2]), dtype=np.float32)
            for input_index, input_i in enumerate(wavelets):
                wavelet_result_key[input_index, :, :, :] = input_i[key]
                pass
            result[key] = torch.Tensor(wavelet_result_key)
            pass

        return result

    def _build(self, model):
        Tools.print('==> Building model..')
        net = model(level=self.wavelet_level)

        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            pass
        return net

    def _change_lr(self, epoch, total_epoch=300):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < total_epoch // 3:
            __change_lr(self.lr)
        elif total_epoch // 3 <= epoch < total_epoch * 2 // 3:
            __change_lr(self.lr / 10)
        elif total_epoch * 2 // 3 <= epoch:
            __change_lr(self.lr / 100)

        pass

    def resume(self, is_resume):
        if is_resume and os.path.isdir(self.checkpoint_path):
            Tools.print('==> Resuming from checkpoint..')
            checkpoint = torch.load('{}/ckpt.t7'.format(self.checkpoint_path))
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        pass

    def train(self, epoch, change_lr=False, total_epoch=300):
        print()
        Tools.print('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch, total_epoch)

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = self._wavelet_data(inputs, self.wavelet_level)
            inputs, targets = {key: inputs[key].to(self.device) for key in inputs}, targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / len(self.train_loader), 100. * correct / total, correct, total))
        pass

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs = self._wavelet_data(inputs, self.wavelet_level)
                inputs, targets = {key: inputs[key].to(self.device) for key in inputs}, targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass
            pass

        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / len(self.test_loader), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            Tools.print('Saving..')
            state = {'net': self.net.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            torch.save(state, '{}/ckpt.t7'.format(self.checkpoint_path))
            self.best_acc = acc
            pass
        Tools.print("best_acc={} acc={}".format(self.best_acc, acc))
        pass

    pass


class Runner2(object):
    def __init__(self, root_path='/home/ubuntu/data1.5TB/cifar',
                 model=WaveletVGG2, wavelet_level=0, batch_size=128, lr=0.1, name="vgg"):
        self.root_path = root_path

        self.model = model
        self.wavelet_level = wavelet_level

        self.batch_size = batch_size
        self.lr = lr
        self.name = "{}_{}".format(name, self.wavelet_level)
        self.checkpoint_path = Tools.new_dir("./checkpoint/{}".format(self.name))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.best_acc = 0
        self.start_epoch = 0

        self.net = self._build(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_loader, self.test_loader = self._data()
        pass

    def info(self):
        Tools.print("model={} wavelet level={} batch size={} lr={} name={}".format(
            str(self.model), self.wavelet_level, self.batch_size, self.lr, self.name))
        pass

    def _data(self):
        Tools.print('==> Preparing data..')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.CIFAR10(self.root_path, train=True, download=True, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=True, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        return _train_loader, _test_loader

    def _wavelet_data(self, inputs, wavelet_level=1, detach=False):
        result = {"Input": inputs}

        if self.wavelet_level == 0:
            return result

        inputs = np.asarray(inputs.detach() if detach else inputs)

        # 获得小波数据
        wavelets = []
        for input_data in inputs:
            wavelet_one = []
            for input_data_channel in input_data:
                wavelet = wavelet_transform(input_data_channel, wavelet_level)
                wavelet_one.append(wavelet)
                pass

            wavelet_result = {}
            for key in wavelet_one[0]:
                shape = wavelet_one[0][key].shape
                wavelet_result_key = np.zeros(shape=(len(wavelet_one), shape[0], shape[1]), dtype=np.float32)
                for wavelet_one_index, wavelet_one_i in enumerate(wavelet_one):
                    wavelet_result_key[wavelet_one_index, :, :] = wavelet_one_i[key]
                    pass
                wavelet_result[key] = wavelet_result_key
                pass

            wavelets.append(wavelet_result)
            pass

        for key in wavelets[0]:
            shape = wavelets[0][key].shape
            wavelet_result_key = np.zeros(shape=(len(wavelets), shape[0], shape[1], shape[2]), dtype=np.float32)
            for input_index, input_i in enumerate(wavelets):
                wavelet_result_key[input_index, :, :, :] = input_i[key]
                pass
            _result_key = torch.Tensor(wavelet_result_key)
            result[key] = _result_key.to(self.device) if detach else _result_key
            pass

        return result

    def _build(self, model):
        Tools.print('==> Building model..')
        net = model(level=self.wavelet_level, wavelet_fn=self._wavelet_data)

        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            pass
        return net

    def _change_lr(self, epoch, total_epoch=300):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < total_epoch // 3:
            __change_lr(self.lr)
        elif total_epoch // 3 <= epoch < total_epoch * 2 // 3:
            __change_lr(self.lr / 10)
        elif total_epoch * 2 // 3 <= epoch:
            __change_lr(self.lr / 100)

        pass

    def resume(self, is_resume):
        if is_resume and os.path.isdir(self.checkpoint_path):
            Tools.print('==> Resuming from checkpoint..')
            checkpoint = torch.load('{}/ckpt.t7'.format(self.checkpoint_path))
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        pass

    def train(self, epoch, change_lr=False, total_epoch=300):
        print()
        Tools.print('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch, total_epoch)

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / len(self.train_loader), 100. * correct / total, correct, total))
        pass

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass
            pass

        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / len(self.test_loader), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            Tools.print('Saving..')
            state = {'net': self.net.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            torch.save(state, '{}/ckpt.t7'.format(self.checkpoint_path))
            self.best_acc = acc
            pass
        Tools.print("best_acc={} acc={}".format(self.best_acc, acc))
        pass

    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1

    runner = Runner2(model=WaveletVGG2, wavelet_level=3, batch_size=128, lr=0.01, name="WaveletVGG2_64")
    runner.info()
    runner.resume(is_resume=False)

    total_epoch = 300
    for _epoch in range(runner.start_epoch, total_epoch):
        runner.train(_epoch, change_lr=True, total_epoch=total_epoch)
        runner.test(_epoch)
        pass

    pass
