import numpy as np
import torch
from utils.utils import AverageMeter, accuracy
from onnxruntime.training import ORTModule

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, scaler, use_ort):
        self.model = ORTModule(model) if use_ort else model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.use_ort = use_ort

    def train(self, data_loader, epoch, amp, log_interval, result_dict):
        total_loss = 0
        count = 0
        
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            if amp == 'Yes':
                with torch.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            if len(labels.size()) > 1:
                labels = torch.argmax(labels, axis=1)

            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            self.optimizer.zero_grad()

            # Backward pass
            if amp == 'Yes':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.tolist()
            count += labels.size(0)

            # Logging the result at set interval
            if batch_idx % log_interval == 0:
                _s = str(len(str(len(data_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.sampler), 100 * count / len(data_loader.sampler)),
                    'train_loss: {: >4.2e}'.format(total_loss / count),
                    'train_accuracy : {:.2f}%'.format(top1.avg)
                ]
                print(', '.join(ret))

        self.scheduler.step()
        result_dict['train_loss'].append(losses.avg)
        result_dict['train_acc'].append(top1.avg)

        return result_dict