import argparse
import os
import os.path as osp
import numpy as np
from shutil import copy2

import torch
from torch import optim
from data_loader import data_loaders
import model.model as model_arch
import model.loss as model_loss
import model.metric as model_metric
from trainer import training, evaluating
from utils import Params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='b01', help='config filename')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--logging', type=bool, default=True, help='logging')
    parser.add_argument('--stage', type=int, default=1, help='1.Training, 2. Testing')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--eval', type=str, default='test', help='dataset')
    parser.add_argument('--pred', type=str, default='pred', help='dataset')

    args = parser.parse_args()
    return args


def data_loading(hparams, stage=1, data_tag='test'):
    data_config = hparams.data
    data_set = data_config['data_set']
    data_dir = data_config['data_dir']
    num_workers = data_config['num_workers']
    data_name = data_config['data_name']
    k_shot = data_config['k_shot']

    if stage == 1:
        batch_size = hparams.batch_size
        split_train = 'train'
        shuffle_train = True
        train_loader = getattr(data_loaders, data_set)(
            batch_size=batch_size,
            data_dir=data_dir,
            split=split_train,
            shuffle=shuffle_train,
            num_workers=num_workers,
            data_name=data_name,
            k_shot=k_shot
        )

        split_val = 'valid'
        shuffle_val = False
        valid_loader = getattr(data_loaders, data_set)(
            batch_size=batch_size,
            data_dir=data_dir,
            split=split_val,
            shuffle=shuffle_val,
            num_workers=num_workers,
            data_name=data_name,
            k_shot=k_shot
        )
        return train_loader, valid_loader
    else:
        batch_size = hparams.batch_size
        shuffle_test = False
        test_loader = getattr(data_loaders, data_set)(
            batch_size=batch_size,
            data_dir=data_dir,
            split=data_tag,
            shuffle=shuffle_test,
            num_workers=num_workers,
            data_name=data_name,
            k_shot=k_shot
        )
        return test_loader


def get_network_paramcount(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


def train(hparams, checkpt, train_loader, valid_loader, exp_dir):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])
    model.to(device)
    epoch_start = 1
    if checkpt is not None:
        model.load_state_dict(checkpt['state_dict'])
        learning_rate = checkpt['cur_learning_rate']
        epoch_start = checkpt['epoch'] + 1

    # loss & metrics
    loss = getattr(model_loss, hparams.loss)
    metrics = [getattr(model_metric, met) for met in hparams.metrics]

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_info = dict(hparams.optimizer)
    optimizer = getattr(optim, optimizer_info['type'])(trainable_params, **optimizer_info['args'])
    if checkpt is not None:
        optimizer.load_state_dict(checkpt['optimizer'])

    # lr scheduler
    lr_scheduler = None if not hparams.lr_scheduler else getattr(optim.lr_scheduler, hparams.lr_scheduler)
    
    # count number of parameters in the mdoe
    num_params = get_network_paramcount(model)
    print('Number of parameters: {}'.format(num_params))

    # train model
    training.train_driver(model, checkpt, epoch_start, optimizer, lr_scheduler, \
        train_loader, valid_loader, loss, metrics, hparams, exp_dir)


def evaluate(hparams, test_loader, exp_dir, data_tag):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])
    model.to(device)
    checkpt = torch.load(exp_dir + '/' + hparams.best_model, map_location=device)
    model.load_state_dict(checkpt['state_dict'])

    # metrics
    metrics = [getattr(model_metric, met) for met in hparams.metrics]
    
    # evaluate model
    evaluating.evaluate_driver(model, test_loader, metrics, hparams, exp_dir, data_tag)


def predict(hparams, eval_loader, pred_loader, exp_dir, data_tag):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])
    model.to(device)
    checkpt = torch.load(exp_dir + '/' + hparams.best_model, map_location=device)
    model.load_state_dict(checkpt['state_dict'])

    # metrics
    metrics = [getattr(model_metric, met) for met in hparams.metrics]

    # evaluate model
    evaluating.prediction_driver(model, eval_loader, pred_loader, metrics, hparams, exp_dir, data_tag)


def main(hparams, checkpt, stage=1, eval_tag='test', pred_tag='pred'):
    # directory path to save the model/results
    exp_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', hparams.exp_name, hparams.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    if stage == 1:
        copy2(json_path, exp_dir)
        # copy model to exp_dir

        # load data
        train_loader, valid_loader = data_loading(hparams, stage)

        # start training
        train(hparams, checkpt, train_loader, valid_loader, exp_dir)
    elif stage == 2:
        # load data
        data_loader = data_loading(hparams, stage, eval_tag)

        # start testing
        evaluate(hparams, data_loader, exp_dir, eval_tag)
    elif stage == 3:
        # load data
        eval_loader = data_loading(hparams, stage, eval_tag)
        pred_loader = data_loading(hparams, stage, pred_tag)

        # start testing
        predict(hparams, eval_loader, pred_loader, exp_dir, pred_tag)


if __name__ == '__main__':
    args = parse_args()

    # fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # filename of the params
    fname_config = args.config + '.json'
    # read the params file
    json_path = osp.join(osp.dirname(osp.realpath('__file__')), "config", fname_config)
    hparams = Params(json_path)
    torch.cuda.set_device(hparams.device)

    # check for a checkpoint passed in to resume from
    if args.checkpt != 'None':
        exp_path = 'experiments/{}/{}/{}'.format(hparams.exp_name, hparams.exp_id, args.checkpt)
        if os.path.isfile(exp_path):
            print("=> loading checkpoint '{}'".format(args.checkpt))
            checkpt = torch.load(exp_path, map_location=device)
            print('checkpoint: ', checkpt.keys())
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpt, checkpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpt))
            exit(0)
    else:
        checkpt = None
    
    if args.stage == 1:
        print('Stage 1: begin training ...')
        main(hparams, checkpt, stage=args.stage)
        print('Training completed!')
        print('--------------------------------------')
    elif args.stage == 2:
        print('Stage 2: begin evaluating ...')
        main(hparams, checkpt, stage=args.stage, eval_tag=args.eval)
        print('Evaluating completed!')
        print('--------------------------------------')
    elif args.stage == 3:
        print('Stage 3: begin predicting ...')
        main(hparams, checkpt, stage=args.stage, eval_tag=args.eval, pred_tag=args.pred)
        print('Evaluating completed!')
        print('--------------------------------------')
    else:
        print('Invalid stage option!')