import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule
device = torch.device('cuda')

class PCEncoder(nn.Module):
    """
        PointNet2 with single-scale grouping
        Classification network
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, beta_sz=16, use_xyz=True):
        super(PCEncoder, self).__init__()

        self.d_rate = 0.3

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
            )
        )

        self.BN_modules = nn.ModuleList()
        self.BN_modules.append(
            nn.BatchNorm1d(128)
        )
        self.BN_modules.append(
            nn.BatchNorm1d(256)
        )
        self.BN_modules.append(
            nn.BatchNorm1d(1024)
        )
        
        self.FiLM_modules = nn.ModuleList()
        self.FiLM_modules.append(
            HyperNetworkLayer(beta_sz,128,128,'relu')
        )
        self.FiLM_modules.append(
            HyperNetworkLayer(beta_sz,256,256,'relu')
        )
        self.FiLM_modules.append(
            HyperNetworkLayer(beta_sz,1024,1024,'relu')
        )

    def forward(self, pointcloud, betas):
        # note: pass in encoded betas
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz = pointcloud.contiguous()
        features = pointcloud.transpose(1, 2).contiguous()
        
        for i in range(len(self.SA_modules)):

            xyz, features = self.SA_modules[i](xyz, features)

            features = self.FiLM_modules[i](betas,features)
            features = self.BN_modules[i](features)
            features = F.relu(features)
            
        return features.squeeze(-1)

class PCEncoderPose(nn.Module):
    """
        PointNet2 with single-scale grouping
        Classification network
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=2, use_xyz=True):
        super(PCEncoderPose, self).__init__()

        self.d_rate = 0.3

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024]
            )
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(256, bn=False, activation=None)
        )

    def forward(self, pointcloud):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz = pointcloud[...,:3].contiguous()
        features = pointcloud[...,3:].transpose(1,2).contiguous()
        
        for SA_module in self.SA_modules:
            xyz, features = SA_module(xyz, features)
            if self.training:
                features = F.dropout(features,self.d_rate)

        features = self.FC_layer(features.squeeze(-1))

        return features

activations = {
    'relu':F.relu,
    'leaky_relu':F.leaky_relu,
    'linear':lambda x: x
}

class FeedForward(nn.Module):
    def __init__(self, indim, hdim, outdim, acts):
        super(FeedForward,self).__init__()

        n = len(acts)
        sizes = [indim] + [hdim] * (n - 1) + [outdim]

        network = []
        for i in range(n):
            layer = nn.Linear(sizes[i],sizes[i+1])
            gain = nn.init.calculate_gain(acts[i])
            nn.init.xavier_normal_(layer.weight,gain=gain)
            nn.init.zeros_(layer.bias)
            network.append(layer)
            if acts[i] == 'relu':
                network.append(nn.ReLU())
            elif acts[i] == 'leaky_relu':
                network.append(nn.LeakyReLU())
        self.network = nn.Sequential(*network)
        
    def forward(self, x):
        return self.network(x)

class HyperNetwork(nn.Module):
    def __init__(self, cdim, fdim):
        super(HyperNetwork,self).__init__()
        
        self.n = 3
        network = []
        activation = 'leaky_relu'
        self.acts = [activation] * (self.n-1) + ['linear']
        for i in range(self.n):
            layer = HyperNetworkLayer(cdim, fdim, fdim, self.acts[i])
            network.append(layer)
        self.network = nn.ModuleList(network)
            
    def forward(self, c, x):
        
        for i in range(self.n):
            x = self.network[i](c,x)
            x = activations[self.acts[i]](x)
        return x
    
class HyperNetworkLayer(nn.Module):
    def __init__(self, cdim, fdim, odim, nonlinearity):
        super(HyperNetworkLayer,self).__init__()

        self.fdim = fdim
        self.odim = odim

        self.h = FeedForward(cdim,cdim,cdim,['leaky_relu']*2+['linear'])
        self.g = FeedForward(cdim,cdim,cdim,['leaky_relu']*2+['linear'])

        gain = nn.init.calculate_gain(nonlinearity)
        
        self.W_net = nn.Linear(cdim,fdim*odim)
        std_W = gain * (2 * cdim * fdim) ** -.5
        nn.init.normal_(self.W_net.weight,std=std_W)
        nn.init.zeros_(self.W_net.bias)

        std_b = gain * (2 * cdim) ** -.5
        self.b_net = nn.Linear(cdim,odim)
        nn.init.normal_(self.b_net.weight,std=std_b)
        nn.init.zeros_(self.b_net.bias)
       
    def forward(self, c, x):

        c = c.view(len(c),-1)

        h = self.h(c)
        g = self.g(c)
        
        W = self.W_net(h).view(len(x),self.odim,self.fdim)
        b = self.b_net(g).unsqueeze(-1)
        
        two_d = len(x.shape) == 2
        if two_d: x = x.unsqueeze(-1)
        x_out = torch.bmm(W,x) + b
        if two_d: x_out = x_out.squeeze(-1)
        
        return x_out
    
"""
class MLP_FiLM_OLD(nn.Module):
    def __init__(self, cdim, fdim):
        super(MLP_FiLM_OLD, self).__init__()

        self.fs = nn.ModuleList()
        self.ls = nn.ModuleList()
        cdim = 69

        self.fs.append(FiLMNetwork(cdim,fdim))
        self.fs.append(FiLMNetwork(cdim,fdim))
        self.fs.append(FiLMNetwork(cdim,fdim))

        self.ls.append(nn.Linear(fdim,fdim))
        self.ls.append(nn.Linear(fdim,fdim))
        self.ls.append(nn.Linear(fdim,fdim))

    def forward(self, c, x, is_train=False):
        for l,f in list(zip(self.ls,self.fs)):
            h = l(x)
            x = x + f(c,h)
        return x


# Multi-layer FiLM Network
class MLP_FiLM(nn.Module):
    def __init__(self, cdim, fdim):
        super(MLP_FiLM, self).__init__()

        self.film = nn.ModuleList()

        self.film.append(LayerFiLM(cdim,fdim,fdim))
        self.film.append(LayerFiLM(cdim,fdim,fdim))
        self.film.append(LayerFiLM(cdim,fdim,fdim))
        
    def forward(self, c, x, is_train=False):
        for f in self.film:
            x = f(c,x)
        return x

class LayerFiLM(nn.Module):

    def __init__(self, c_sz, f_sz, out_sz):
        super(LayerFiLM, self).__init__()
        gain = nn.init.calculate_gain('tanh')
        self.l = HyperNetworkLayer(c_sz, out_sz, gain=gain)
        #self.l = nn.Linear(f_sz, out_sz)
        #nn.init.xavier_normal_(self.l.weight,gain=nn.init.calculate_gain('tanh'))
        self.f = FiLMNetwork(c_sz, out_sz)

    def forward(self, inputs, features):

        features = self.l(inputs,features).tanh()
        #features = self.l(features).tanh()
        return self.f(inputs,features)
    
class FiLMNetwork(nn.Module):
    
    def __init__(self, in_sz, out_sz):
        super(FiLMNetwork, self).__init__()

        self.f = nn.Linear(in_sz, out_sz)
        nn.init.xavier_normal_(self.f.weight)
        self.h = nn.Linear(in_sz, out_sz)
        nn.init.xavier_normal_(self.h.weight)

    def forward(self, inputs, features):
        reshape = [len(features)] + [1]*(len(features.shape)-2)+[-1]

        gamma = self.f(inputs).view(reshape)
        beta = self.h(inputs).view(reshape)

        return features * gamma + beta
"""
