import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import torch.utils.data as DATA
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import torch.nn.functional as F
import time

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def entropy_loss(p):
    p = F.softmax(p,dim=1)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def image_classification_test(loader, model, timestep,test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):                                    
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                outputs = outputs.float().cpu()
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy




        
def train(config):
    print('iid',config['IID'],'base',config['base'],'sfd',config['sfd'],'All',config['All'])
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])\

    ## prepare data
    
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    ###############################################################
    class_num = config["network"]["params"]["class_num"]
    iter_source = iter(dset_loaders["source"]) 
    len_train_source = len(dset_loaders["source"])
    source_count = torch.zeros(class_num)
    for i in range(len_train_source):
        _, labels_source = iter_source.next()
        for j in range(labels_source.size()[0]):
            source_count[labels_source[j]] = source_count[labels_source[j]] +1 
    print('source_count',source_count,torch.sum(source_count))
        
    
    ##################################################################
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    ####################################################
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    

    ## set base network
    net_config = config["network"]
    ############################# train from 
    #base_network = net_config["name"](**net_config["params"])
    ############################## import base_network
    base_network = torch.load('./savedmodel/amazon_source_only.pkl')
    ############################## import source_only
    source_only = torch.load('./savedmodel/amazon_source_only.pkl')
    base_network = base_network.cuda()
    source_only = source_only.cuda()


    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        #ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
        if config['method'] == 'DANN': 
            ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        if config['method'] == 'MCC' : 
            ad_net = network.AdversarialNetwork(base_network.output_num()* class_num, 1024)
        if config['method'] == 'CDAN': 
            ad_net = network.AdversarialNetwork(base_network.output_num()* class_num, 1024)
            
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        

    ## train 
    count_step = 0
    count_iter = 0
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    ###################### set train_epoch
    epoch = 20
    
    iter_target = iter(dset_loaders["target"])
    acc_so = image_classification_test(dset_loaders, \
                    base_network,0,test_10crop=prep_config["test_10crop"])
    print('source-only-acc', acc_so)
    ######################### online target data flow
    for timestep in range(len_train_target):
        inputs_target, labels_target = iter_target.next()
        inputs_target, labels_target = inputs_target.cuda(), labels_target.cuda()
        ######################################## Target data in each step
        for i in range(epoch):
            loss_params = config["loss"]                  
            ## train one iter
            base_network.train(True)
            ad_net.train(True)
            optimizer = lr_scheduler(optimizer, i, **schedule_param)
            optimizer.zero_grad()            
            if i % len_train_online_source == 0:
                iter_source = iter(online_source_loader)               
            inputs_source, labels_source = iter_source.next()
            inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
            features_source, outputs_source = base_network(inputs_source)
            ###############################################################            
            features_source_only, _ = source_only(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            #############################################################
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            if config['method'] == 'CDAN+E':           
                entropy = loss.Entropy(softmax_out)
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
            elif config['method']  == 'CDAN':
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
            elif config['method']  == 'DANN':
                transfer_loss = loss.DANN(features, ad_net)
            elif config['method']  == 'MCC':
                outputs_target_temp = outputs_target / 2.5
                target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
                target_entropy_weight = loss.Entropy(target_softmax_out_temp).detach()
                target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
                target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
                cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
                cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
                transfer_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
            else:
                raise ValueError('Method cannot be recognized.')
            loss_rem = torch.mean((features_source-features_source_only)*(features_source-features_source_only))
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            if config['base'] == True:
                total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
            else:
                total_loss =  transfer_loss + loss_rem + classifier_loss     
            total_loss.backward()
            optimizer.step()
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='DANN', choices=['CDAN', 'CDAN+E', 'DANN','MCC','our'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--sfd', type=bool, default=False, help="whether use sfd")
    parser.add_argument('--base', type=bool, default=False, help="whether use base")   
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/webcam.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")     
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 1000
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    config['sfd'] = args.sfd
    config['base'] = args.base
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":36}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
