import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxsim
import ast

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from onnx import shape_inference
from loguru import logger
logger.remove(0)
logger.add(sys.stderr, level = "TRACE")

from rs_extractor_encoder import RAFTStereoExtractorEncoder
from rs_refiner import RAFTStereoRefiner
from utils.utils import InputPadder
#from core.utils.utils import coords_grid, upflow8
from utils.utils import coords_grid, upflow8
from core.corr import PytorchAlternateCorrBlock1D, CorrBlock1D
from export.export import export_onnx
from core.utils.utils import initialize_flow
from core.utils.utils import upsample_flow, load_image

DEVICE = 'cuda:2'

terminal_script = """
    python3 demo.py --restore_ckpt "checkpoints/raftstereo-middlebury.pth" "output_depth" --input_dims "[1, 3, 416, 800]" > logs/logs.txt
"""

def demo(args):
    test_mode = True
    model_featureExtractor = torch.nn.DataParallel(RAFTStereoExtractorEncoder(args), device_ids=[0])
    model_featureExtractor.load_state_dict(torch.load(args.restore_ckpt))
    model_featureExtractor = model_featureExtractor.module
    model_featureExtractor.to(DEVICE)
    model_featureExtractor.eval()
    
    model_updator = torch.nn.DataParallel(RAFTStereoRefiner(args), device_ids=[0])
    model_updator.load_state_dict(torch.load(args.restore_ckpt))
    model_updator = model_updator.module
    model_updator.to(DEVICE)
    model_updator.eval()

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to output_disparity/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1, args.input_dims, tvm_fp32 = False).to(DEVICE)
            image2 = load_image(imfile2, args.input_dims, tvm_fp32 = False).to(DEVICE)

            padder = InputPadder(image1.shape, divis_by=32)
           
            image1, image2 = padder.pad(image1, image2)

            image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
            image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
            image_set = torch.cat([image1, image2], dim = 2)

            #net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2 = model_featureExtractor(image1, image2, test_mode = True)
            net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2 = model_featureExtractor(image_set)

            # initialize correlation function and flow predictions
            corr_block = PytorchAlternateCorrBlock1D
            corr_fn = corr_block(fmap1, fmap2, radius = args.corr_radius, num_levels = args.corr_levels)
            coords0, coords1 = initialize_flow(net_list_0)
            
            # initialize flow_predictions
            flow_predictions = []
            for itr in range(args.valid_iters):

                coords1 = coords1.detach()

                corr = corr_fn(coords1)
                flow = coords1 - coords0
                
                net_list_0, net_list_1, net_list_2, up_mask, delta_flow = model_updator(net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, corr, flow)

                delta_flow[:, 1] = 0.0
                coords1 = coords1 + delta_flow

                if test_mode and itr < args.valid_iters - 1:
                    continue

                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = upsample_flow(args,coords1 - coords0, up_mask)
                flow_up = flow_up[:, :1]
                flow_predictions.append(flow_up)
            
            flow_up = padder.unpad(flow_up).squeeze()

            file_stem = imfile1.split('/')[-2]
            plt.imsave("output_disparity/demo_output.png", -flow_up.cpu().numpy().squeeze(), cmap = 'jet')

            if args.export:
                export_onnx(args, model_featureExtractor, model_updator, tuple(args.input_dims))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="input_images/left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="input_images/right.png")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=64, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--input_dims', type = str, default = "[0,0,0,0]", help = "input images dimension")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="alt", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--export', action='store_true', help='flag for exporting models to onnx')
    
    args = parser.parse_args()
    args.input_dims = ast.literal_eval(args.input_dims)

    demo(args)
