import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.update import BasicMultiUpdateBlock


class RAFTStereoRefiner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.channel_width = 128
        
        context_dims = args.hidden_dims
    
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.fnet = BasicEncoder(output_dim = 256, norm_fn = 'instance', downsample = args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

    def forward(self, net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, corr, flow):

        inp_list = []

        inp_list.append(list(torch.split(inp_list_0, self.channel_width, 1)))
        inp_list.append(list(torch.split(inp_list_1, self.channel_width, 1)))
        inp_list.append(list(torch.split(inp_list_2, self.channel_width, 1)))

        net_list_0, net_list_1, net_list_2, up_mask, delta_flow = self.update_block(net_list_0, net_list_1, net_list_2, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

        return net_list_0, net_list_1, net_list_2, up_mask, delta_flow