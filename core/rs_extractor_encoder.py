import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.update import BasicMultiUpdateBlock


class RAFTStereoExtractorEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.fnet = BasicEncoder(output_dim = 256, norm_fn = 'instance', downsample = args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

    def forward(self, image1, image2, flow_init = None, test_mode = False):

        image_set = torch.cat([image1, image2], dim = 1)
        image1, image2 = torch.split(image_set, split_size_or_sections = 3, dim = 1)

        cnet_list = self.cnet(image1, num_layers = self.args.n_gru_layers)
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1, fmap2 = fmap1.float(), fmap2.float()

        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]

        # rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        inp_list_0 = torch.cat(inp_list[0], dim = 1)
        inp_list_1 = torch.cat(inp_list[1], dim = 1)
        inp_list_2 = torch.cat(inp_list[2], dim = 1)

        
        net_list_0, net_list_1, net_list_2 = net_list[:len(net_list)]

        return net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2