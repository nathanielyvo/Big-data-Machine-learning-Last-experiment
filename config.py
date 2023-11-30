import argparse

def get_parser():#参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)  
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--batchsize', type=int, default=32)
    return parser.parse_args()
