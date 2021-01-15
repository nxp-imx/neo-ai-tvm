import scipy.io as spio
import numpy as np
import os
import sys
from PIL import Image
import numpy as np

from tflite_models import *

MEASURE_PERF = False
SUPPORTED_MODELS = {}  # name to TFModel mapping
image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def read_annotation(annotations_path):
    mat = spio.loadmat(annotations_path)
    img = mat['GTcls']['Segmentation'][0][0]
    return img

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


def add_supported_model(name, where, is_quant=False, formats='tflite'):
    m = TFModel(name, where, is_quant, formats)
    SUPPORTED_MODELS[m.name] = m

    return m


def init_supported_models():
    QUANT = True
    where = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu"
    m = add_supported_model("deeplabv3_257_mv_gpu", where)
    m.input_size = 257
    m.inputs = "sub_7"

    where = "https://github.com/google-coral/edgetpu/raw/master/test_data"
    m = add_supported_model("deeplabv3_mnv2_pascal", where, is_quant=QUANT)
    m.input_size = 513
    m.inputs = "MobilenetV2/MobilenetV2/input"

    return SUPPORTED_MODELS

init_supported_models()
for model_name, m in SUPPORTED_MODELS.items():
    print("\nTesting {0: <50}".format(model_name.upper()))

    is_quant = m.is_quant
    input_size = m.input_size

    shape = (1, input_size, input_size, 3)

    image_data = get_img_data(image_url, shape[1:3], is_quant)
    ref_output, prof_res = get_ref_result(shape, m, image_data, MEASURE_PERF)
    if MEASURE_PERF:
        print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))

    try:
        tvm_output, prof_res = inference_remotely(m, shape, image_data, MEASURE_PERF)
        if MEASURE_PERF:
            print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
                  % (np.mean(prof_res), np.std(prof_res)))
        if is_quant:
            ref_output = ref_output.reshape(shape[1:3])
            tvm_output = tvm_output.reshape(shape[1:3])

            pix_acc = pixel_accuracy(ref_output, tvm_output)
            print("pixel accuracy:", pix_acc)
            m_acc = mean_accuracy(ref_output, tvm_output)
            print("mean accuracy:", m_acc)
            IoU = mean_IU(ref_output, tvm_output)
            print("mean IU:", IoU)
            freq_weighted_IU = frequency_weighted_IU(ref_output, tvm_output)
            print("frequency weighted IU:", freq_weighted_IU)
        else:
            np.testing.assert_allclose(ref_output, tvm_output, rtol=1e-4, atol=1e-4, verbose=True)
    except Exception as err:
        print(model_name, ": FAIL")
    else:
        print(model_name, ": PASS")
