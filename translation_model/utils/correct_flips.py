import numpy as np
import os,sys
import matplotlib.pyplot as plt
import SimpleITK as sitk

folder = "/scratch/amartinezmora/translation_model_data/preprocessed_nnunet/Dataset002_nonRestrictedDistance/nnUNetPlans_3d_fullres"
gt_folder = "/scratch/amartinezmora/translation_model_data/preprocessed_nnunet/Dataset002_nonRestrictedDistance/gt_segmentations"
cids = ["mrclean_late_30110", "mrclean_late_30289", "mrclean_late_30309",
       "mrclean_med_20329", "mrclean_noiv_10142", "mrclean_noiv_10214"]

flip = (1,2)
for cid in cids:
    file = os.path.join(folder, f"{cid}.npz")
    data = np.load(file)
    img, dist, time = data['data'], data["dist"], data["time"]

    flip_dist = np.flip(dist, flip)
    flip_time = np.flip(time, flip)

    brain_coords = np.array(np.where(img > img.min()))
             
    brain_centroid = np.median(brain_coords,1).astype(int)
    brain_centroid = np.clip(brain_centroid, a_min=0, a_max=None)

    if brain_centroid[1] > img.shape[1]:
        brain_centroid[1] = img.shape[1]//2
    if brain_centroid[2] > img.shape[2]:
        brain_centroid[2] = img.shape[2]//2 
    if brain_centroid[3] > img.shape[3]:
        brain_centroid[3] = img.shape[3]//2

    outfile_png = os.path.join(folder, f"{cid}.png")
    outfile_npz = os.path.join(folder, f"{cid}_new.npz")

    plt.figure()

    plt.subplot(331)
    plt.imshow(img[0,brain_centroid[1]], cmap="gray")
    plt.colorbar()
    plt.subplot(332)
    plt.imshow(flip_dist[0,brain_centroid[1]])
    plt.colorbar()
    plt.subplot(333)
    plt.imshow(flip_time[0,brain_centroid[1]])
    plt.colorbar()
    plt.subplot(334)
    plt.imshow(img[0,:,brain_centroid[2]], cmap="gray")
    plt.colorbar()
    plt.subplot(335)
    plt.imshow(flip_dist[0,:,brain_centroid[2]])
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(flip_time[0,:,brain_centroid[2]])
    plt.colorbar()
    plt.subplot(337)
    plt.imshow(img[0,:,:,brain_centroid[3]], cmap="gray")
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(flip_dist[0,:,:,brain_centroid[3]])
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(flip_time[0,:,:,brain_centroid[3]])
    plt.colorbar()
    plt.savefig(outfile_png)
    plt.show()
    plt.close()


    # save flipped file
    np.savez_compressed(outfile_npz, data=img, dist=flip_dist, time=flip_time)
    