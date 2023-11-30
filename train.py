import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import ResNet18,ResNet34
from config import get_parser

if __name__ == '__main__':
    
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(0)

    train_loader = get_dataloader_all(args)
    n_class = 7
    model = ResNet34()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    n_epochs = 200
    lr = 0.01
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        right_sample = 0
        total_sample = 0
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train()
        for data, target in tqdm(train_loader):

            data = data.to(device)
            target = target.to(device) 

            optimizer.zero_grad()
            output = model(data).to(device)
            pred = torch.argmax(output,dim=1)

            right_sample += torch.sum(pred==target)
            total_sample += target.shape[0]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.sampler)
        print('Epoch: {} \tTraining Loss: {:.6f}\tTraing Acc: {:6f}'.format(
            epoch, train_loss, right_sample/total_sample))

    torch.save(model.state_dict(), 'Checkpoints/resnet.pth')
