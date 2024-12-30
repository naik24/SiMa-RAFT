import numpy as np
import time
import dataclasses
import torch
import matplotlib.pyplot as plt
import logging
import argparse
import ast
import onnxruntime as ort 

from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.apis.loaded_net import load_model
from afe.ir.tensor_type import ScalarType
from afe.apis.defines import default_quantization, quantization_scheme, int16_quantization, CalibrationMethod
from afe.apis.defines import default_quantization,MinMaxMethod,MovingAverageMinMaxMethod,HistogramMSEMethod,HistogramEntropyMethod,HistogramPercentileMethod
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from core.rs_extractor_encoder import RAFTStereoExtractorEncoder
from core.rs_refiner import RAFTStereoRefiner
from core.utils.utils import load_image, InputPadder
from core.utils.utils import initialize_flow, upflow8, upsample_flow
from core.corr import CorrBlock1D

DEVICE = 'cpu'

logging.basicConfig(
    filename = "logs/LCQE_torch.log",
    filemode = "w",
    level = logging.INFO,
    force = True,
)

# ===================== loading models =====================
@logger.catch()
def load(args):

    # loading images
    left_img = load_image(args.left_imgs, args.input_dims, tvm_fp32 = False)
    right_img = load_image(args.right_imgs, args.input_dims, tvm_fp32 = False)

    # pad the input
    padder = InputPadder(left_img.shape, divis_by = 32)
    image1, image2 = padder.pad(left_img, right_img)
    
    image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
    image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

    image1, image2 = image1.numpy(), image2.numpy()
    image_set = np.concatenate([image1, image2], axis = 2)

    # ===================== loading feature extractor and encoder =====================
    fee_path = args.fee_path

    fee_inputs = {
        'input_pair': image_set,
    }

    fee_input_shapes = {
        'input_pair': image_set.shape,
    }

    fee_input_types = {
        'input_pair': ScalarType.float32,
    }

    fee_importer_params: ImporterParams = onnx_source(
        model_path = fee_path,
        shape_dict = fee_input_shapes,
        dtype_dict = fee_input_types
    )

    loaded_fee = load_model(fee_importer_params, log_level = logging.DEBUG)

    # ===================== inference on feature extractor and encoder =====================

    #fee_inputs['left_image'] = np.transpose(fee_inputs['left_image'], (0, 2, 3, 1))
    #fee_inputs['right_image'] = np.transpose(fee_inputs['right_image'], (0, 2, 3, 1))
    fee_inputs['input_pair'] = np.transpose(fee_inputs['input_pair'], (0, 2, 3, 1))

    net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2 = loaded_fee.execute(fee_inputs, log_level = logging.DEBUG)

    # ===================== correlation processing =====================

    correlation_block = CorrBlock1D
    corr_fn = correlation_block(
        torch.from_numpy(fmap1.transpose((0, 3, 1, 2))), 
        torch.from_numpy(fmap2.transpose((0, 3, 1, 2))), 
        radius = args.corr_radius, 
        num_levels = args.corr_levels)
    coords0, coords1 = initialize_flow(torch.from_numpy(net_list_0.transpose((0, 3, 1, 2))))

    corr = corr_fn(coords1)
    flow = coords1 - coords0

    # ===================== loading refiner =====================
    refiner_path = args.refiner_path

    refiner_input_shapes = {
        'net_list_0.1': tuple(net_list_0.transpose((0, 3, 1, 2)).shape),
        'net_list_1.1': tuple(net_list_1.transpose((0, 3, 1, 2)).shape),
        'net_list_2.1': tuple(net_list_2.transpose((0, 3, 1, 2)).shape),
        'inp_list_0': tuple(inp_list_0.transpose((0, 3, 1, 2)).shape),
        'inp_list_1': tuple(inp_list_1.transpose((0, 3, 1, 2)).shape),
        'inp_list_2': tuple(inp_list_2.transpose((0, 3, 1, 2)).shape),
        'corr': tuple(corr.shape),
        'flow': tuple(flow.shape)
    }

    refiner_inputs = {
        'net_list_0.1': net_list_0,
        'net_list_1.1': net_list_1,
        'net_list_2.1': net_list_2,
        'inp_list_0': inp_list_0,
        'inp_list_1': inp_list_1,
        'inp_list_2': inp_list_2,
        'corr': corr.detach().cpu().numpy().transpose((0, 2, 3, 1)),
        'flow': flow.detach().cpu().numpy().transpose((0,2, 3, 1))
    }

    refiner_input_types = {
        'net_list_0.1': ScalarType.float32,
        'net_list_1.1': ScalarType.float32,
        'net_list_2.1': ScalarType.float32,
        'inp_list_0': ScalarType.float32,
        'inp_list_1': ScalarType.float32,
        'inp_list_2': ScalarType.float32,
        'corr': ScalarType.float32,
        'flow': ScalarType.float32,
    }

    refiner_importer_params: ImporterParams = onnx_source(
        model_path = refiner_path,
        shape_dict = refiner_input_shapes,
        dtype_dict = refiner_input_types
    )

    loaded_refiner = load_model(refiner_importer_params, log_level = logging.INFO)

    return loaded_fee, fee_inputs, loaded_refiner, refiner_inputs
    
def quantize_model(loaded_model, calibrated_data, model_name):

    quant_configs = default_quantization

    calib_data = []
    calib_data.append(calibrated_data)

    quantized_model = loaded_model.quantize(
        calibration_data = calib_data,
        quantization_config = quant_configs,
        model_name = model_name,
        log_level = logging.INFO
    )

    return quantized_model

def compile_model(args, quantized_model):

    quantized_model.compile(
        output_path = args.output_directory,
        log_level = logging.DEBUG,
    )

def execute_tvm_fp32(loaded_net, inputs: dict):
    
    fp32_model_output = loaded_net.execute(inputs)

    return fp32_model_output

@logger.catch()
def quantized_execution(args, fee_quantized, refiner_quantized, fast_mode = False):

    # loading images
    left_img = load_image(args.left_imgs, args.input_dims, tvm_fp32 = False)
    right_img = load_image(args.right_imgs, args.input_dims, tvm_fp32 = False)

    padder = InputPadder(left_img.shape, divis_by = 32)
    image1, image2 = padder.pad(left_img, right_img)
    
    image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
    image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

    image1, image2 = image1.numpy(), image2.numpy()
    #image1, image2 = np.transpose(image1, (0, 2, 3, 1)), np.transpose(image2, (0, 2, 3, 1))
    image_set = np.concatenate([image1, image2], axis = 2)
    image_set = np.transpose(image_set, (0, 2, 3, 1))

    fee_inputs = {
        'input_pair': image_set
    }

    # executing quantized feature extractor encoder
    net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, fmap1, fmap2 = fee_quantized.execute(fee_inputs, fast_mode = fast_mode, log_level = logging.INFO)

    #converting to torch and NCHW format for next modules
    net_list_0 = torch.from_numpy(np.transpose(net_list_0, (0, 3, 1, 2))).to(DEVICE)
    net_list_1 = torch.from_numpy(np.transpose(net_list_1, (0, 3, 1, 2))).to(DEVICE)
    net_list_2 = torch.from_numpy(np.transpose(net_list_2, (0, 3, 1, 2))).to(DEVICE)
    inp_list_0 = torch.from_numpy(np.transpose(inp_list_0, (0, 3, 1, 2))).to(DEVICE)
    inp_list_1 = torch.from_numpy(np.transpose(inp_list_1, (0, 3, 1, 2))).to(DEVICE)
    inp_list_2 = torch.from_numpy(np.transpose(inp_list_2, (0, 3, 1, 2))).to(DEVICE)
    fmap1 = torch.from_numpy(np.transpose(fmap1, (0, 3, 1, 2))).to(DEVICE)
    fmap2 = torch.from_numpy(np.transpose(fmap2, (0, 3, 1, 2))).to(DEVICE)


    #correlation processing (torch)
    correlation_block = CorrBlock1D
    corr_fn = correlation_block(
        fmap1,
        fmap2,
        radius = args.corr_radius, 
        num_levels = args.corr_levels)
    coords0, coords1 = initialize_flow(net_list_0)

    # ================================= fp32 refiner ================================== #
    refiner = torch.nn.DataParallel(RAFTStereoRefiner(args), device_ids = [0])
    refiner.load_state_dict(torch.load(args.restore_ckpt, map_location = 'cpu'))
    refiner = refiner.module
    refiner.to('cpu')
    refiner.eval()

    for itr in tqdm(range(args.valid_iters)):
        time.sleep(0.1)

        coords1 = coords1.detach()
        corr = corr_fn(coords1)
        
        flow = coords1 - coords0

        net_list_0, net_list_1, net_list_2, up_mask, delta_flow = refiner(net_list_0, net_list_1, net_list_2, inp_list_0, inp_list_1, inp_list_2, corr, flow)

        delta_flow[:, 1] = 0.0
        coords1 = coords1 + delta_flow

        if itr < args.valid_iters - 1:
            continue

        if up_mask is None:
            flow_up = upflow8(coords1 - coords0)
        else:
            flow_up = upsample_flow(args, coords1 - coords0, up_mask)
            
        flow_up = flow_up[:, :1]
   
    plt.imsave(f"output_disparity/test_sima_torch_output.png", -flow_up.detach().numpy().squeeze(), cmap = 'jet')

@logger.catch()
def main(args):

    # load the models
    logger.info("<<<< loading models >>>>")
    fee_loaded, fee_inputs, refiner_loaded, refiner_inputs = load(args)
    logger.success("<<<< models loaded successfully >>>>\n")

    # quantize the models
    logger.info("<<<< quantizing models >>>>")
    fee_quantized = quantize_model(fee_loaded, fee_inputs, model_name = 'feature_extractor_encoder')
    refiner_quantized = quantize_model(refiner_loaded, refiner_inputs, model_name = 'refiner')
    logger.success("<<<< models quantized successfully >>>>\n")

    #compile the quantized models
    if args.compile:
        logger.info("<<<< compiling models. this may take a while >>>>")
        compile_model(args, fee_quantized)
        compile_model(args, refiner_quantized)
        logger.success("<<<< models compiled successfully >>>>\n")

    if args.fast_mode:
        logger.info("<<<< executing model in fast mode >>>>")
        quantized_execution(args, fee_quantized, refiner_quantized, fast_mode = True)
        logger.success("<<<< fast mode execution complete >>>>\n")
    else:
        logger.info("<<<< executing model in normal mode >>>>")
        quantized_execution(args, fee_quantized, refiner_quantized, fast_mode = False)
        logger.info("<<<< model execution complete >>>>\n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dims', type = str, default = "[0, 0, 0, 0]", help = 'input dimensions in NCHW format')
    parser.add_argument('--fee_path', type = str, help = "path of feature extractor and encoder onnx file", required = True)
    parser.add_argument('--refiner_path', type = str, help = "path of refiner onnx file", required = True)
    parser.add_argument('--left_imgs', type = str, required = False, default = "input_images/left.png")
    parser.add_argument('--right_imgs', type = str, required = False, default = "input_images/right.png")
    parser.add_argument('--output_directory', type = str, required = False, default = "outputs/sima")
    parser.add_argument('--compile', action = 'store_true', help = "compile the models or not")
    parser.add_argument('--fast_mode', action = 'store_true', help = "execute sima quantized model in fast mode")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default = "checkpoints/raftstereo-middlebury.pth")
    
    # raft-stereo configs
    parser.add_argument('--valid_iters', type = int, default = 32, help = 'number of flow-field updates during forward pass')
    parser.add_argument('--corr_radius', type = int, default = 4, help = "width of the correlation pyramid")
    parser.add_argument('--corr_levels', type = int, default = 4, help = "number of levels in correlation pyramid")
    parser.add_argument('--hidden_dims', nargs = '+', type = int, default = [128]*3, help = "hidden state and context dimensions")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()
    args.input_dims = ast.literal_eval(args.input_dims)

    main(args)