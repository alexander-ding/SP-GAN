import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import etw_pytorch_utils as pt_utils
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule

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
                mlp=[256, 256, 512, 1024]
            )
        )
        
        self.FiLM_modules = nn.ModuleList()
        self.FiLM_modules.append(
            FiLMNetwork(beta_sz,128)
        )
        self.FiLM_modules.append(
            FiLMNetwork(beta_sz,256)
        )
        self.FiLM_modules.append(
            FiLMNetwork(beta_sz,1024)
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(256, bn=False, activation=None)
        )

    def forward(self, pointcloud, betas):
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
        
        for SA_module,FiLM_module in list(zip(self.SA_modules,self.FiLM_modules)):
            
            xyz, features = SA_module(xyz, features)
            features = FiLM_module(betas,
                                   features.permute(0,2,1)).permute(0,2,1)
            features = F.dropout(features,self.d_rate)

        return self.FC_layer(features.squeeze(-1))

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
            features = F.dropout(features,self.d_rate)

        return self.FC_layer(features.squeeze(-1))

# Multi-layer perceptron helper function
class MLP_FiLM(nn.Module):
    def __init__(self, cdim, fdim):
        super(MLP_FiLM, self).__init__()
        self.l1 = nn.Linear(fdim, fdim)
        self.l2 = nn.Linear(fdim, fdim)
        self.l3 = nn.Linear(fdim, fdim)

        self.f1 = FiLMNetwork(cdim,fdim)
        self.f2 = FiLMNetwork(cdim,fdim)

    def forward(self, c, x):
        x = self.f1(c,self.l1(x)).tanh()
        x = self.f2(c,self.l2(x)).tanh()
        return self.l3(x)
    
class FiLMNetwork(nn.Module):
    
    def __init__(self, in_sz, out_sz):
        super(FiLMNetwork, self).__init__()
        
        self.f = nn.Linear(in_sz, out_sz)
        self.h = nn.Linear(in_sz, out_sz)

    def forward(self, inputs, features):
        gamma = self.f(inputs).unsqueeze(1)
        beta = self.h(inputs).unsqueeze(1)

        return features * gamma + beta
    
if __name__ == '__main__':
    device = torch.device('cuda')
    a = torch.randn(10,10000,3).to(device)
    enc = PCEncoder().to(device)
    print(enc(a).shape)
