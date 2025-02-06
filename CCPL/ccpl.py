import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from CCPL.function import calc_mean_std, nor_mean_std, nor_mean, calc_cov
import random
import itertools

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

NovelMLP = True
enableBias = True
mlp = nn.Sequential(nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(64, 16,bias=enableBias),
                    
                    nn.Linear(128, 128,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(128, 128,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(128, 32,bias=enableBias),
                    
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(256, 64,bias=enableBias),
                    
                    nn.Linear(512, 512,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(512, 512,bias=enableBias),
                    nn.ReLU(),
                    nn.Linear(512, 128,bias=enableBias)
                    ) 
                                           

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        return x
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp
        print('hittted')
        
        
        # self.mlp1 = nn.Sequential(nn.Linear(64, 64),
        #                     nn.ReLU(),
        #                     nn.Linear(64, 16))
        # self.mlp2 = nn.Sequential(
        #             nn.Linear(128, 128),
        #             nn.ReLU(),
        #              nn.Linear(128, 32)
        #             ) 
        # self.mlp3 = nn.Sequential(
        #             nn.Linear(256, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 64),) 
        
        # self.mlp3 = nn.Sequential(
        #             nn.Linear(512, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 128))                                       
                                           
        # print(self.mlp2)  
        # self.mlp2.requires_grad = True
        # # self.mlp1.requires_grad = True
        # self.mlp.requires_grad = True

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        # feat_d = torch.ones_like(feat_d)
        layercount = 5
        for i in range(layercount):
            feat_d =self.mlp[layercount*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.01):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            # return f_q.sum()
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)   
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp    

class Net(nn.Module):
    def __init__(self, encoder, training_mode='art'):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        mlp_altered = mlp if training_mode == 'art' else mlp[:9]
        
        self.CCPL = CCPL(mlp_altered)
        self.CCPL.train()
        self.end_layer = 3 if training_mode == 'art' else 3
        self.mode = training_mode

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


    def forward(self, content, gimage):
        num_s = 256
        num_layer = 1

        content_feats = self.encode_with_intermediate(content)

        g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer

        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)
        return loss_ccp


def InitCPPL():
    vgg_read = vgg
    training_mode = 'art'
    vgg_read.load_state_dict(torch.load('./models//vgg_normalized.pth'))
    vgg_read = nn.Sequential(*list(vgg_read.children())[:31]) if training_mode == 'art' else nn.Sequential(*list(vgg_read.children())[:18])
    network = Net(vgg_read, training_mode)
    network.cuda()
    for name in network.state_dict():
        print(name)
    
    return network

def trainCCPL(network,content, gimage,itercount,lr):
    
    itercount = 1000
    optimizer = torch.optim.Adam(itertools.chain(network.CCPL.parameters()), lr=lr)

    # for param_group in optimizer.param_groups:# 
    #     tensors = (param_group['params'])
    #     for ten in tensors:
    #         print(ten.grad)


    network.CCPL.requires_grad_(True)
    for i in range(itercount):
        optimizer.zero_grad()
        loss_ccp = network(content, gimage)

        loss_ccp.backward()
        if i % 100 ==0:
            print(loss_ccp.item())
        optimizer.step()

    network.CCPL.requires_grad_(False)
    # for i in range(itercount):
    #     optimizer.zero_grad()
    #     loss_ccp = network(content, gimage)

    #     loss_ccp.backward()
    #     if i % 100 ==0:
    #         print(loss_ccp.item())
    #     optimizer.step()
    
