import argparse
import torch
import yaml
from utils.utils import loaddata
from utils.model import DPSC


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    args = parser.parse_args()


    data_path = './diff_data/' + args.dataset + '_diff_' + args.pretrained + '.pt'
    data, label = loaddata(data_path)
    with open('./config.yaml') as config_file:
        config_all = yaml.load(config_file, yaml.Loader)
    config = config_all[args.dataset]

    net = DPSC(n_all=len(label), label=label, alpha=config['alpha'], gamma1=config['gamma1'],
               gamma2=config['gamma2'], dim=config['dim'], ro=config['ro'])
    _ = net.train(data=data, epochs=config['epoch'], lr=args.lr)

if __name__ == "__main__":
    main()