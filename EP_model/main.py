import argparse
import os
import os.path as osp
import numpy as np
from shutil import copy2

import torch
from torch import optim
from data_loader import data_loaders, mesh2graph
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
    parser.add_argument('--stage', type=int, default=1, help='1.Training, 2. Testing')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--tag', type=str, default='test', help='dataset')

    args = parser.parse_args()
    return args


def data_loading(hparams, stage=1):
    data_config = hparams.data
    data_set = data_config['data_set']
    data_dir = data_config['data_dir']
    num_workers = data_config['num_workers']
    data_names = data_config['data_names']
    signal_type = data_config['signal_type']
    num_meshes = data_config['num_meshes']
    seq_len = data_config['seq_len']
    k_shot = data_config.get('k_shot')

    if stage == 1:
        train_loaders, valid_loaders = {}, {}
    elif stage == 2:
        eval_loaders = {}
    elif stage == 3:
        eval_loaders, pred_loaders = {}, {}
    
    for data_name, num_mesh in zip(data_names, num_meshes):
        if stage == 1:
            batch_size = hparams.batch_size
            train_loader = getattr(data_loaders, data_set)(
                batch_size=batch_size,
                data_dir=data_dir,
                split='train',
                shuffle=True,
                num_workers=num_workers,
                data_name=data_name,
                signal_type=signal_type,
                num_mesh=num_mesh,
                seq_len=seq_len,
                k_shot=k_shot
            )

            valid_loader = getattr(data_loaders, data_set)(
                batch_size=batch_size,
                data_dir=data_dir,
                split='valid',
                shuffle=False,
                num_workers=num_workers,
                data_name=data_name,
                signal_type=signal_type,
                num_mesh=num_mesh,
                seq_len=seq_len,
                k_shot=k_shot
            )
            train_loaders[data_name] = train_loader
            valid_loaders[data_name] = valid_loader
        elif stage == 2:
            batch_size = hparams.batch_size
            eval_tags = data_config['eval_tags']
            eval_loader_tag = {}

            for eval_tag in eval_tags:
                eval_loader = getattr(data_loaders, data_set)(
                    batch_size=batch_size,
                    data_dir=data_dir,
                    split=eval_tag,
                    shuffle=False,
                    num_workers=num_workers,
                    data_name=data_name,
                    signal_type=signal_type,
                    num_mesh=num_mesh,
                    seq_len=seq_len,
                    k_shot=k_shot
                )
                eval_loader_tag[eval_tag] = eval_loader
            eval_loaders[data_name] = eval_loader_tag
        elif stage == 3:
            batch_size = hparams.batch_size
            eval_tags = data_config['eval_tags']
            pred_tags = data_config['pred_tags']
            eval_loader_tag = {}
            pred_loader_tag = {}

            for eval_tag, pred_tag in zip(eval_tags, pred_tags):
                eval_loader = getattr(data_loaders, data_set)(
                    batch_size=batch_size,
                    data_dir=data_dir,
                    split=eval_tag,
                    shuffle=False,
                    num_workers=num_workers,
                    data_name=data_name,
                    signal_type=signal_type,
                    num_mesh=num_mesh,
                    seq_len=seq_len,
                    k_shot=k_shot
                )

                pred_loader = getattr(data_loaders, data_set)(
                    batch_size=batch_size,
                    data_dir=data_dir,
                    split=pred_tag,
                    shuffle=False,
                    num_workers=num_workers,
                    data_name=data_name,
                    signal_type=signal_type,
                    num_mesh=num_mesh,
                    seq_len=seq_len,
                    k_shot=k_shot
                )
                eval_loader_tag[eval_tag] = eval_loader
                pred_loader_tag[pred_tag] = pred_loader
            eval_loaders[data_name] = eval_loader_tag
            pred_loaders[data_name] = pred_loader_tag

    if stage == 1:
        return train_loaders, valid_loaders
    elif stage == 2:
        return eval_loaders
    elif stage == 3:
        return eval_loaders, pred_loaders


def get_network_paramcount(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


def train(hparams, checkpt, train_loaders, valid_loaders, exp_dir):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])

    # setup parameters for each patient
    graph_method = hparams.data['graph_method']
    data_dir = os.path.join(osp.dirname(osp.realpath('__file__')), hparams.data['data_dir'])
    batch_size = hparams.batch_size
    ecgi = hparams.ecgi
    for data_name in hparams.data['data_names']:
        model.setup(data_name, data_dir, batch_size, ecgi, graph_method)

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
    if not hparams.lr_scheduler or hparams.lr_scheduler == 0:
        lr_scheduler = None
    else:
        lr_scheduler_info = dict(hparams.lr_scheduler)
        lr_scheduler = getattr(optim.lr_scheduler, lr_scheduler_info['type'])(optimizer, **lr_scheduler_info['args'])
    
    
    # count number of parameters in the mdoe
    num_params = get_network_paramcount(model)
    print('Number of parameters: {}'.format(num_params))

    # train model
    training.train_driver(model, checkpt, epoch_start, optimizer, lr_scheduler, \
        train_loaders, valid_loaders, loss, metrics, hparams, exp_dir)


def evaluate(hparams, eval_loaders, exp_dir, data_tags):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])

    # setup parameters for each patient
    graph_method = hparams.data['graph_method']
    data_dir = os.path.join(osp.dirname(osp.realpath('__file__')), hparams.data['data_dir'])
    batch_size = hparams.batch_size
    ecgi = hparams.ecgi
    for data_name in hparams.data['data_names']:
        model.setup(data_name, data_dir, batch_size, ecgi, graph_method)

    model.to(device)
    checkpt = torch.load(exp_dir + '/' + hparams.best_model, map_location=device)
    model.load_state_dict(checkpt['state_dict'])

    # metrics
    metrics = [getattr(model_metric, met) for met in hparams.metrics]
    
    # evaluate model
    evaluating.evaluate_driver(model, eval_loaders, metrics, hparams, exp_dir, data_tags)


def prediction(hparams, eval_loaders, pred_loaders, exp_dir, data_tags):
    # models
    model_info = dict(hparams.model)
    model = getattr(model_arch, model_info['type'])(**model_info['args'])

    # setup parameters for each patient
    graph_method = hparams.data['graph_method']
    data_dir = os.path.join(osp.dirname(osp.realpath('__file__')), hparams.data['data_dir'])
    batch_size = hparams.batch_size
    ecgi = hparams.ecgi
    for data_name in hparams.data['data_names']:
        model.setup(data_name, data_dir, batch_size, ecgi, graph_method)

    model.to(device)
    checkpt = torch.load(exp_dir + '/' + hparams.best_model, map_location=device)
    model.load_state_dict(checkpt['state_dict'])

    # metrics
    metrics = [getattr(model_metric, met) for met in hparams.metrics]
    
    # evaluate model
    evaluating.prediction_driver(model, eval_loaders, pred_loaders, 
                                  metrics, hparams, exp_dir, data_tags)


def main(hparams, checkpt, stage, data_tags='test'):
    # directory path to save the model/results
    exp_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', hparams.exp_name, hparams.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    if stage == 1:
        copy2(json_path, exp_dir)

        # load data
        train_loaders, valid_loaders = data_loading(hparams, stage)

        # start training
        train(hparams, checkpt, train_loaders, valid_loaders, exp_dir)
    elif stage == 2:
        # load data
        eval_loaders = data_loading(hparams, stage)

        # start testing
        evaluate(hparams, eval_loaders, exp_dir, data_tags)
    elif stage == 3:
        # load data
        eval_loaders, pred_loaders = data_loading(hparams, stage)

        # start personalization
        prediction(hparams, eval_loaders, pred_loaders, exp_dir, data_tags)


def make_graph(hparams):
    data_config = hparams.data
    data_dir = data_config['data_dir']
    data_names = data_config['data_names']
    signal_type = data_config['signal_type']
    num_meshes = data_config['num_meshes']
    graph_method = data_config['graph_method']
    seq_len = data_config['seq_len']

    for data_name, num_mesh in zip(data_names, num_meshes):
        print(data_name)
        root_dir = os.path.join(data_dir, 'signal/{}'.format(data_name))
        structure_name = data_name.split('_')[0]
        g = mesh2graph.GraphPyramid(data_name, structure_name, num_mesh, seq_len, graph_method)
        g.make_graph()


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
    
    tags = args.tag.split(',')

    if args.stage == 1:
        print('Stage 1: begin training ...')
        main(hparams, checkpt, stage=args.stage)
        print('Training completed!')
        print('--------------------------------------')
    elif args.stage == 2:
        print('Stage 2: begin evaluating ...')
        main(hparams, checkpt, stage=args.stage, data_tags=tags)
        print('Evaluating completed!')
        print('--------------------------------------')
    elif args.stage == 3:
        print('Stage 3: begin prediction ...')
        main(hparams, checkpt, stage=args.stage, data_tags=tags)
        print('Prediction completed!')
        print('--------------------------------------')
    elif args.stage == 0:
        print('Stage 0: begin making graphs ...')
        make_graph(hparams)
        print('Making graph completed!')
        print('--------------------------------------')
    else:
        print('Invalid stage option!')
