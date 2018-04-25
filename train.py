# -*- coding:utf-8 -*-

import mxnet as mx



import argparse
parser = argparse.ArgumentParser(description='Process train and val list.')
parser.add_argument('--num_classes', dest='num_classes',type=int, default=7,  \
                    help='the num of the classes (default: 7)')
parser.add_argument('--batch_size', dest='batch_size',type=int, default=128,  \
                    help='batch size (default: 128)')
parser.add_argument('--network', dest='network',type=str, default='mobilenet',  \
                    help='the net structure uesd (default: mobilenet)')
parser.add_argument('--layer_name', dest='layer_name',type=str, default='pool6',  \
                    help='the layer start to train, for finetune, you should change it (default: pool6 for mobile net)')
parser.add_argument('--viz_net', dest='viz_net',type=bool, default=0,  \
                    help='to viz the net structure or not (default: false)')
parser.add_argument('--epoch', dest='epoch',type=int, default=0,  \
                    help='the models for finetune epoch (default: 0)')
parser.add_argument('--finetune', dest='finetune',type=bool, default=0,  \
                    help='finetune frome a pretrained model (default: false)')
parser.add_argument('--scratch', dest='scratch',type=bool, default=0,  \
                    help='train from sctrach (default: false)')
parser.add_argument('--num_epoch', dest='num_epoch',type=int, default=100,  \
                    help='epoch be trained (default: 100)')
parser.add_argument('--data_shape', dest='data_shape',type=int, default=112,  \
                    help='the image shape  (default: 112)')
parser.add_argument('--log_file', dest='log_file',type=str, default='log.log',  \
                    help='the log file (default: log.log)')
args = parser.parse_args()



def get_fine_tune_model(symbol, arg_params, num_classes, layer_name=args.layer_name):

    all_layers = symbol.get_internals()

    net = all_layers[layer_name+'_output']
    net = mx.symbol.LeakyReLU(data=net,act_type='prelu')
    net = mx.symbol.Dropout(data=net, p=0.1)

    embedding=mx.symbol.FullyConnected(data=net,num_hidden=512,no_bias=True)
    embedding = mx.symbol.LeakyReLU(data=embedding, act_type='prelu')
    embedding = mx.symbol.Dropout(data=embedding, p=0.5)
    #embedding = mx.symbol.L2Normalization(data=embedding)
    fc2 = mx.symbol.FullyConnected(data=embedding, num_hidden=num_classes)  ####classify layer hidden=num_class
    fc2 = mx.symbol.flatten(data=fc2)
    net = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    new_args = dict({k:arg_params[k] for k in arg_params if layer_name not in k})
    return (net, new_args)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
fh = logging.FileHandler(args.log_file)
logger.addHandler(fh)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    import  os
    if not os.access('./trained_models',os.F_OK):
        os.mkdir('./trained_models')
    lr_scheduler = mx.lr_scheduler.FactorScheduler(20, 0.8)

    epoch_end_callback = mx.callback.do_checkpoint("./trained_models/your_model", 1)

    devs = [mx.gpu(i) for i in range(num_gpus)]

    mod = mx.mod.Module(symbol=symbol,
                        context=devs,
                        data_names=['data'],
                        label_names=['softmax_label'])

    mod.fit(train, val,
        num_epoch=args.num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 5),
        epoch_end_callback=epoch_end_callback,
        kvstore='device',
        optimizer='sgd',
        optimizer_params={
            'learning_rate':0.01,
            'momentum': 0.9,
            'lr_scheduler': lr_scheduler,
            'wd':0.005
        },
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        eval_metric='acc')

import  Mydataiter
train_iter, val_iter=Mydataiter.get_iterator(args.batch_size,args.data_shape)


if args.finetune:
    print('finetune from pretrained model ...with', args.network)
    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/' + args.network, args.epoch)
    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, args.num_classes)

    if args.viz_net:
        print(new_sym.list_arguments())
        b = mx.viz.plot_network(new_sym)#可视化网络结构
        b.view()

    fit(new_sym, new_args, aux_params, train_iter, val_iter, args.batch_size,num_gpus=1)



if args.scratch:

    print('train from scratch...with',args.network)
    if args.network=='mobilenet':
        from symbols import  mobilenet
        sym=mobilenet.get_symbol(args.num_classes)

    elif args.network=='mobilenet_v2':
        from symbols import mobilenetv2
        sym=mobilenetv2.get_symbol(args.num_classes)

    elif args.network=='resnext':
        from symbols import  resnext

        ###you should set the params to get the symbol
        #sym=resnext.get_symbol(***)


    if args.viz_net:
        print(new_sym.list_arguments())
        b = mx.viz.plot_network(new_sym)#可视化网络结构
        b.view()

    lr_scheduler = mx.lr_scheduler.FactorScheduler(20, 0.8)
    epoch_end_callback = mx.callback.do_checkpoint("./trained_models/your_model", 1)

    devs = mx.gpu()

    mod = mx.mod.Module(symbol=sym,
                        context=devs,
                        data_names=['data'],
                        label_names=['softmax_label'])

    mod.fit(train_iter, val_iter,
            num_epoch=args.num_epoch,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 5),
            epoch_end_callback=epoch_end_callback,
            kvstore='device',
            optimizer='sgd',
            optimizer_params={
                'learning_rate': 0.01,
                'momentum': 0.9,
                'lr_scheduler': lr_scheduler,
                'wd': 0.005
            },
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            eval_metric='acc')


























