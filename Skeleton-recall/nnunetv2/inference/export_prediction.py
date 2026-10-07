import os
from copy import deepcopy
from typing import Union, List
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def convert_predicted_logits_to_regression_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    print("predicted_logits", predicted_logits.shape)
    
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])

    print("predicted_logits", predicted_logits.shape)
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    #predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    #del predicted_logits
    #segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # predicted_logits may be torch.Tensor but we continue with numpy
    if isinstance(predicted_logits, torch.Tensor):
        predicted_logits = predicted_logits.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    logits_reverted_cropping = np.zeros([3] + list(properties_dict['shape_before_cropping']))
    print("logits_reverted_cropping", logits_reverted_cropping.shape)
    print("bbox", properties_dict['bbox_used_for_cropping'])
    slicer = (slice(None),) + bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    logits_reverted_cropping[slicer] = predicted_logits
    print("slicer", slicer)

    # revert transpose
    print("logits_reverted_cropping", logits_reverted_cropping.shape)
    transpose_axes = [0] + (np.array(plans_manager.transpose_backward) + 1).tolist()  
    # logits_reverted_cropping = logits_reverted_cropping.transpose(plans_manager.transpose_backward)
    logits_reverted_cropping = logits_reverted_cropping.transpose(transpose_axes)
    print("logits_reverted_cropping", logits_reverted_cropping.shape)

    """
    pred_np = logits_reverted_cropping.copy()
    plt.figure()
    plt.subplot(131)
    plt.imshow(pred_np[0, pred_np.shape[1]//2], cmap="gray")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(pred_np[1, pred_np.shape[1]//2], cmap="gray")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(pred_np[2, pred_np.shape[1]//2], cmap="gray")
    plt.colorbar()
    plt.savefig("/home/amartinezmora/plot_pred.png")
    plt.close()
    """
    
    torch.set_num_threads(old_threads)
    return logits_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    

def export_regression_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_regression_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
    )
    del predicted_array_or_file

    regression_final = ret
    del ret

    save_regression_pred(pred = regression_final,
                         properties=properties_dict,
                         outfile=output_file_truncated + dataset_json_dict_or_file['file_ending'])
    #rw = plans_manager.image_reader_writer_class()
    #rw.write_reg(regression_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 #properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)



def resample_and_save_regression(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # obtain regression results
    #regression = predicted_array_or_file[-1]
    #regression[predicted_array_or_file[0] > 0] = 0.0
    
    # regression may be torch.Tensor but we continue with numpy
    #if isinstance(regression, torch.Tensor):
    #    regression = regression.cpu().numpy()

    regression = predicted_array_or_file.copy()
    np.savez_compressed(output_file, seg=regression)
    torch.set_num_threads(old_threads)



def save_regression_pred(pred : np.ndarray, properties : dict, outfile : os.PathLike):
    """
    Save predicted regression

    Params
    ------
    pred : predicted regression
    properties : dict with data on original image spacing, affine matrix, etc.
    outfile : output file
    
    """

    # Save array
    pred = np.moveaxis(pred, 0, -1) # Set channel as last dimension
    npy_file = outfile.replace(".nii.gz", ".npy")
    np.save(file=npy_file, arr=pred)
    """
    pred_image = sitk.GetImageFromArray(pred, isVector=False)

    # Derive output image properties
    spacing = tuple(properties['spacing'])
    affine = properties['nibabel_stuff']['original_affine']
    origin, direction = get_origin_and_direction(affine=affine, 
                                                 spacing=spacing)
    pred_image.SetSpacing(spacing)
    pred_image.SetOrigin(origin)
    pred_image.SetDirection(direction)

    sitk.WriteImage(pred_image, outfile)
    """


def get_origin_and_direction(affine, spacing=None):
    """
    Extracts the origin and direction cosines from a 4x4 affine matrix.

    Parameters:
    - affine (np.ndarray): A 4x4 affine transformation matrix.
    - spacing (list or np.ndarray, optional): Spacing along each axis [z, y, x] or [x, y, z].
        If None, spacing is computed from the affine.

    Returns:
    - origin (tuple): 3-element array with the image origin in world coordinates.
    - directions (tuple): 3x3 matrix of direction cosines (unit vectors for x, y, z axes).
    """
    affine = np.asarray(affine)
    if affine.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4.")

    origin = affine[:3, 3]

    # Extract voxel axes vectors
    axes = affine[:3, :3]

    # If spacing is not given, compute it from the norm of each column vector
    if spacing is None:
        spacing = np.linalg.norm(axes, axis=0)

    # Normalize each column to get direction cosines
    directions = axes / spacing

    return tuple(origin.tolist()), tuple(directions.flatten().tolist())