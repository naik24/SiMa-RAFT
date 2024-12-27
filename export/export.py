#import sys
#sys.path.append('core')

import torch
import onnx
import onnxsim
import argparse
import numpy as np

from core.rs_extractor_encoder import RAFTStereoExtractorEncoder
from core.rs_refiner import RAFTStereoRefiner
from core.utils.utils import InputPadder
from core.utils.utils import coords_grid, upflow8, initialize_flow, upsample_flow, load_image
from core.corr import PytorchAlternateCorrBlock1D, CorrBlock1D
from loguru import logger
from PIL import Image

DEVICE = 'cuda:2'

logger.add('testlogger.log')

@logger.catch()
def export_onnx(
    args: argparse.Namespace,
    feature_extractor_encoder: RAFTStereoExtractorEncoder, 
    refiner: RAFTStereoRefiner, 
    input_shape: tuple) -> None:

    # ==================================== feature extractor conversion ==================================== #

    # defining dummy inputs
    dummy_input_left = load_image(args.left_imgs, args.input_dims, tvm_fp32 = False).to(DEVICE)
    dummy_input_right = load_image(args.right_imgs, args.input_dims, tvm_fp32 = False).to(DEVICE)

    padder = InputPadder(dummy_input_left.shape, divis_by=32)
    dummy_input_left, dummy_input_right = padder.pad(dummy_input_left, dummy_input_right)

    dummy_input_left = (2 * (dummy_input_left / 255.0) - 1.0).contiguous()
    dummy_input_right = (2 * (dummy_input_right / 255.0) - 1.0).contiguous()

    # defining input and output names
    input_names = [
        'left_image',
        'right_image'
    ]

    output_names = [
        'net_list_0',
        'net_list_1',
        'net_list_2',
        'inp_list_0',
        'inp_list_1',
        'inp_list_2',
        'fmap1',
        'fmap2'
    ]

    # sample run through feature_extractor_encoder
    net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2 = feature_extractor_encoder(
                                                                                                image1 = dummy_input_left, 
                                                                                                image2 = dummy_input_right
                                                                                            )

    # exporting to onnx
    torch.onnx.export(
        model = feature_extractor_encoder,  # model being run
        args = (dummy_input_left, dummy_input_right),  # model input (or a tuple for multiple inputs)
        f = "checkpoints/rs_extractor_encoder.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the onnx version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names, # the model's output names
    )

    # simplifying the onnx model
    onnx_model = onnx.load("checkpoints/rs_extractor_encoder.onnx")
    model_opt, check = onnxsim.simplify(onnx_model)
    onnxsim.model_info.print_simplifying_info(onnx_model, model_opt) 
    onnx.save(model_opt, "checkpoints/rs_extractor_encoder_simplified.onnx")

    # ==================================== mid-processsing ==================================== #

    # defining correlation block and functions
    corr_block = PytorchAlternateCorrBlock1D
    corr_fn = corr_block(fmap1, fmap2, radius = args.corr_radius, num_levels = args.corr_levels)
    coords0, coords1 = initialize_flow(net_list_0)

    coords1 = coords1.detach()
    corr = corr_fn(coords1)
    flow = coords1 - coords0

    # ==================================== refiner conversion ==================================== #
    
    # defining input and output names
    input_names = [
        'net_list_0',
        'net_list_1', 
        'net_list_2',
        'inp_list_0',
        'inp_list_1', 
        'inp_list_2',
        'corr',
        'flow'
    ]

    output_names = [
        'net_list_0',
        'net_list_1',
        'net_list_2', 
        'up_mask',
        'delta_flow'
    ]

    # exporting to onnx
    torch.onnx.export(
        model = refiner, # model being run
        args = (net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, corr, flow), # model input (or a tuple for multiple inputs)
        f = "checkpoints/rs_refiner.onnx", # where to save the model
        export_params = True, # store the trained parameters inside the model file
        opset_version = 17, # the onnx version to export the model to
        do_constant_folding = True, # whether to execute constant folding for optimization
        input_names = input_names, # model's input names
        output_names = output_names # model's output names
    )

    # simplifying the onnx model
    onnx_model = onnx.load("checkpoints/rs_refiner.onnx")
    model_opt, check = onnxsim.simplify(onnx_model) # simplifying
    onnxsim.model_info.print_simplifying_info(onnx_model, model_opt) #logging changes after simplifying
    onnx.save(model_opt, "checkpoints/rs_refiner_simplified.onnx") # saving the simplified model