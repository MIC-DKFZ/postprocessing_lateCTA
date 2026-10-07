import numpy as np
import os,sys
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset, nnUNetDataset_reg2chan
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

import torch.nn.functional as F


import inspect

import matplotlib.pyplot as plt

from typing import Union, Tuple


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}



class nnUNetDataLoader3D_reg2chan(nnUNetDataLoader3D):

    
    def determine_shapes(self):
        # load one case
        data, dist, tta, brain_mask, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, dist.shape[0], *self.patch_size)
        return data_shape, seg_shape

    

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        dist_all = np.zeros(self.seg_shape, dtype=np.float32)
        tta_all = np.zeros(self.seg_shape, dtype=np.float32)
        brain_mask_all = np.zeros(self.seg_shape, dtype=np.int16)
        vessel_mask_all = np.zeros(self.seg_shape, dtype=np.int16)

        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, dist, tta, brain_mask, properties = self._data.load_case(i)
            
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, dist.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            #seg = seg[this_slice]
            dist, tta, brain_mask = dist[this_slice], tta[this_slice], brain_mask[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=data.min())
            #seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)
            dist_all[j] = np.pad(dist, padding, 'constant', constant_values=dist.max())
            tta_all[j] = np.pad(tta, padding, 'constant', constant_values=0)
            brain_mask_all[j] = np.pad(brain_mask, padding, 'constant', constant_values=0)


        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    dist_all = torch.from_numpy(dist_all)
                    tta_all = torch.from_numpy(tta_all)
                    brain_mask_all = torch.from_numpy(brain_mask_all == 1).to(torch.int16)
                    seg_all = torch.from_numpy(np.concatenate([dist_all, tta_all, brain_mask_all], axis=1)).float()

                    images, segs = [], []

                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'seg': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['seg'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    
                    del images, segs

            target_all = seg_all[:,:2]

            vessel_segs, brain_segs = [], []
            for b in range(self.batch_size):
                brain_segs.append(brain_mask_all[b])
                vessel_segs.append((seg_all[b,1] >= 0.05)*brain_segs[-1])
                brain_segs[-1] = brain_segs[-1].unsqueeze(0)
                vessel_segs[-1] = vessel_segs[-1].unsqueeze(0)

            vessel_mask_all = torch.cat(vessel_segs, 0)
            brain_mask_all = torch.cat(brain_segs, 0)

            return {'data': data_all, 'target': target_all, 'brain' : brain_mask_all, 'vessel': vessel_mask_all, 'keys': selected_keys}

        data_all = torch.from_numpy(data_all).float()
        #seg_all = torch.from_numpy(seg_all).to(torch.int16)
        dist_all = torch.from_numpy(dist_all).float()
        tta_all = torch.from_numpy(tta_all).float()
        
        target_all = torch.cat((dist_all, tta_all), dim=1)

        vessel_segs, brain_segs = [], []
        for b in range(self.batch_size):
            brain_segs.append(brain_mask_all[b])
            vessel_segs.append((tta_all[b] >= 0.05)*brain_mask_all[b])
            vessel_segs[-1] = vessel_segs[-1].unsqueeze(0)
            brain_segs[-1] = torch.from_numpy(brain_segs[-1]).unsqueeze(0)

        vessel_mask_all = torch.cat(vessel_segs, 0)
        brain_mask_all = torch.cat(brain_segs, 0)

        return {'data': data_all, 'target': target_all, 'brain' : brain_mask_all, 'vessel': vessel_mask_all, 'keys': selected_keys}



if __name__ == '__main__':
    from nnunetv2.run.run_training import get_trainer_from_args
    folder = '/scratch/amartinezmora/translation_model_data/preprocessed_nnunet/Dataset004_distnorm/nnUNetPlans_3d_fullres'
    plans_file = '/scratch/amartinezmora/translation_model_data/preprocessed_nnunet/Dataset004_distnorm/nnUNetResEncUNetLPlans.json'
    dataset_file = "/scratch/amartinezmora/translation_model_data/preprocessed_nnunet/Dataset004_distnorm/dataset.json"
    dataset_json = load_json(dataset_file) 
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    label_manager = plans_manager.get_label_manager(dataset_json)

    keys = [
            "R2668",
            "R2729",
            "R2949",
            "R2964",
            "R3190",
            "R3274",
            "R4988",
            "R4989",
            "mrclean_late_30053",
            "mrclean_late_30058",
            "mrclean_late_30066",
            "mrclean_late_30080",
            "mrclean_late_30203",
            "mrclean_late_30217",
            "mrclean_late_30221",
            "mrclean_late_30358",
            "mrclean_late_30362",
            "mrclean_late_30368",
            "mrclean_late_30369",
            "mrclean_late_30416",
            "mrclean_med_20041",
            "mrclean_med_20055",
            "mrclean_med_20083",
            "mrclean_med_20143",
            "mrclean_med_20287",
            "mrclean_med_20445",
            "mrclean_med_20495",
            "mrclean_med_20630",
            "mrclean_med_20672",
            "mrclean_noiv_10012",
            "mrclean_noiv_10027",
            "mrclean_noiv_10028",
            "mrclean_noiv_10034",
            "mrclean_noiv_10072",
            "mrclean_noiv_10079",
            "mrclean_noiv_10082",
            "mrclean_noiv_10142",
            "mrclean_noiv_10153",
            "mrclean_noiv_10214",
            "mrclean_noiv_10283",
            "mrclean_noiv_10478"
        ]


    ds = nnUNetDataset_reg2chan(folder, plans, keys)  # this should not load the properties!

    transforms = []
    rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    transforms.append(
            SpatialTransform(
                (192, 80, 160), patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, 
                scaling=(0.85, 1.25), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False, mode_seg='nearest', 
                border_mode_seg='reflection',
                padding_mode_image='reflection'  # , mode_seg='nearest'
            )
        )

    transforms = ComposeTransforms(transforms)
    transforms = None

    dl = nnUNetDataLoader3D_reg2chan(data = ds, 
                                     batch_size = 8,
                                     patch_size = (192, 80, 160), 
                                     final_patch_size = (192, 80, 160), 
                                     oversample_foreground_percent = 0.5,
                                     label_manager = label_manager,
                                     transforms=transforms)
    a = next(dl)

    elements = {'data' : a['data'], 'dist': a['target'][:,0], 'tta': a['target'][:,1],
                'brain' : a['brain'], 'vessel': a['vessel']}

    """   
    for b in range(a["data"].shape[0]):
        plt.figure()
        c = 1
        for k,v in elements.items():
            cmap = "gray" if k == "data" else "jet"
            plt.subplot(1,5,c)
            plt.imshow(v.numpy().squeeze()[b,:,v.shape[-2]//2], cmap=cmap)
            plt.colorbar()
            c += 1

        plt.savefig(f"/home/amartinezmora/plot{b}.png")
    """

    # Checkpoint loading 
    cpt_file = "/home/amartinezmora/scratch/translation_model_results/Dataset004_distnorm/nnUNetRegressionTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_latest.pth"
    state_dict = torch.load(cpt_file, weights_only=False, map_location=torch.device('cpu'))['network_weights']

    configuration_manager = plans_manager.get_configuration("3d_fullres")
    trainer = get_trainer_from_args("004", "3d_fullres", 0, "nnUNetRegressionTrainer",
                                    "nnUNetResEncUNetLPlans", use_compressed=True, device=torch.device('cpu'))

    network = trainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            1,
            2,
            enable_deep_supervision=False
        )
    
    network.load_state_dict(state_dict)
    network.to("cpu")
    network.eval()

    out = network(a['data'])

    out_dist_mask = (out[:,0].cpu().detach().numpy() <= 0).astype(np.uint8)

    #out[:,1] = F.sigmoid(out[:,1]) # TODO: keep or remove this depending on the application

    for b in range(out.shape[0]):
        plt.figure(figsize=(20,10))
        plt.subplot(3,6,1)
        plt.imshow(a['data'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,2)
        plt.imshow(a['target'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy()*a['brain'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,3)
        plt.imshow(a['target'][b,1,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,4)
        plt.imshow(out[b,0,:,a['data'].shape[3]//2].cpu().detach().numpy()*a['brain'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,5)
        plt.imshow((out[b,1,:,a['data'].shape[3]//2].cpu().detach().numpy())*out_dist_mask[b,:,a['data'].shape[3]//2]*a['brain'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,6)
        plt.imshow((out[b,-1,:,a['data'].shape[3]//2].cpu().detach().numpy())*a['brain'][b,0,:,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,7)
        plt.imshow(a['data'][b,0,a['data'].shape[2]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,8)
        plt.imshow(a['target'][b,0,a['data'].shape[2]//2].cpu().detach().numpy()*a['brain'][b,0,a['data'].shape[2]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,9)
        plt.imshow(a['target'][b,1,a['data'].shape[2]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,10)
        plt.imshow(out[b,0,a['data'].shape[2]//2].cpu().detach().numpy()*a['brain'][b,0,a['data'].shape[2]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,11)
        plt.imshow((out[b,1,a['data'].shape[2]//2].cpu().detach().numpy())*out_dist_mask[b,a['data'].shape[2]//2]*a['brain'][b,0,a['data'].shape[2]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,12)
        plt.imshow((out[b,-1,a['data'].shape[2]//2].cpu().detach().numpy())*a['brain'][b,0,a['data'].shape[3]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,13)
        plt.imshow(a['data'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,14)
        plt.imshow(a['target'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy()*a['brain'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,15)
        plt.imshow(a['target'][b,1,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,16)
        plt.imshow(out[b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy()*a['brain'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,17)
        plt.imshow((out[b,1,:,:,a['data'].shape[4]//2].cpu().detach().numpy())*out_dist_mask[b,:,:,a['data'].shape[4]//2]*a['brain'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.subplot(3,6,18)
        plt.imshow((out[b,-1,:,:,a['data'].shape[4]//2].cpu().detach().numpy())*a['brain'][b,0,:,:,a['data'].shape[4]//2].cpu().detach().numpy(), cmap="gray")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"/home/amartinezmora/plot{b}.png")
        plt.close()