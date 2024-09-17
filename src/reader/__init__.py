import logging

from .ntu_reader import NTU_Reader
# from dataset.graphs import Graph
import sys
sys.path.append("D:/publication1/ss_GCN/src/dataset/graphs")
from src.dataset.graphs import Graph


__generator = {
    'ntu': NTU_Reader,
}

def create(args):
    dataset = args.dataset.split('-')[0]
    dataset_args = args.dataset_args[dataset]
    graph = Graph(dataset)
    connect_joint = graph.connect_joint
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, connect_joint, **dataset_args)
