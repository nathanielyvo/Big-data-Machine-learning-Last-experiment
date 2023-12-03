import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import ResNet18,ResNet34
import pandas as pd
import warnings
from config import get_parser
warnings.filterwarnings("ignore")

from datasets import *


if __name__ == '__main__':

    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_reverse_list = ['dog','elephant','giraffe','guitar','horse','house','person']

    test_loader = get_dataloader_test(args)

    n_class = 7
    model = ResNet34()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class) 
    state_dict = torch.load('Checkpoints/resnet.pth')
    model.load_state_dict(state_dict)

    model.to(device)
    ID = []
    LABEL = []
    model.eval()
    
    with torch.no_grad():
        for index,data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output,dim=1).cpu().numpy().tolist()
            ID += torch.tensor(index).numpy().tolist()
            LABEL += pred
    for i in range(len(LABEL)):
        LABEL[i] = label_reverse_list[LABEL[i]]

    res = pd.DataFrame([ID,LABEL]).T
    res.columns = ['ID','label']
    res.to_csv('result.csv',index=False)

    print('test finish!\tpredict file save in result.csv')
