import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

class CurveExtractor:
    def __init__(self, ctp_array: np.ndarray,
                 aif_roi_center=(37, 100, 128),
                 vof_roi_center=(37, 211, 128), 
                 template_nii=None, 
                 save_masks=False, 
                 outdir = None):

        self.template_nii = template_nii

        self.ctp_array = ctp_array
        self.aif_roi_center = aif_roi_center
        # aif roi should be (20, 50, 50) mm
        # so in voxels it should be
        self.aif_roi_width = (int(20//template_nii.GetSpacing()[2]),
                              int(50//template_nii.GetSpacing()[1]),
                              int(50//template_nii.GetSpacing()[0]))
        self.vof_roi_center = vof_roi_center
        self.vof_roi_width = (int(20//template_nii.GetSpacing()[2]),
                              int(50//template_nii.GetSpacing()[1]),
                              int(50//template_nii.GetSpacing()[0]))
        self.make_rois()

        self.calculate_baseline()
        self.subtract_baseline()
        self.calculate_max_diff()

        self.aif = None
        self.vof = None

        self.save_masks = save_masks
        self.outdir = outdir


    def calculate_baseline(self):
        self.baseline = np.mean(self.ctp_array[:2], axis=0, keepdims=True)
    def subtract_baseline(self):
        self.ctp_array = self.ctp_array - self.baseline
        self.ctp_array = np.clip(self.ctp_array, 0, 600)
    def calculate_max_diff(self):
        self.max_diff = np.max(self.ctp_array, axis=0) - np.min(self.ctp_array, axis=0)

    def make_rois(self):
        self.aif_roi = np.zeros(self.ctp_array.shape)
        self.aif_roi[:, self.aif_roi_center[0] - self.aif_roi_width[0] // 2:self.aif_roi_center[0] + self.aif_roi_width[0] // 2,
                        self.aif_roi_center[1] - self.aif_roi_width[1] // 2:self.aif_roi_center[1] + self.aif_roi_width[1] // 2,
                        self.aif_roi_center[2] - self.aif_roi_width[2] // 2:self.aif_roi_center[2] + self.aif_roi_width[2] // 2] = 1
        self.vof_roi = np.zeros(self.ctp_array.shape)
        self.vof_roi[:, self.vof_roi_center[0] - self.vof_roi_width[0] // 2:self.vof_roi_center[0] + self.vof_roi_width[0] // 2,
                        self.vof_roi_center[1] - self.vof_roi_width[1] // 2:self.vof_roi_center[1] + self.vof_roi_width[1] // 2,
                        self.vof_roi_center[2] - self.vof_roi_width[2] // 2:self.vof_roi_center[2] + self.vof_roi_width[2] // 2] = 1

    def calculate_aif(self):
        ctp_array_aif = self.ctp_array * self.aif_roi
        max_diff_aif = self.max_diff * self.aif_roi[0]
        #p95 = np.percentile(max_diff_aif[max_diff_aif > 0], 98)
        # select the 10 highest voxels max_diff_aif and get indices
        indices_with_highest_values = np.unravel_index(np.argsort(max_diff_aif.ravel())[-100:], max_diff_aif.shape)
        segmentation = np.zeros_like(max_diff_aif)
        segmentation[indices_with_highest_values] = 1
        if self.save_masks:
            seg_sitk = sitk.GetImageFromArray(segmentation)
            seg_sitk.CopyInformation(self.template_nii)
            sitk.WriteImage(seg_sitk, os.path.join(self.outdir, 'aif_mask.nii.gz'))
        self.aif = np.mean(ctp_array_aif[:, segmentation == 1], axis=1)
        return self.aif
    def calculate_vof(self):
        ctp_array_vof = self.ctp_array * self.vof_roi
        max_diff_vof = self.max_diff * self.vof_roi[0]
        p95 = np.percentile(max_diff_vof[max_diff_vof > 0], 98)
        indices_with_highest_values = np.unravel_index(np.argsort(max_diff_vof.ravel())[-100:], max_diff_vof.shape)

        segmentation = np.zeros_like(max_diff_vof)
        segmentation[indices_with_highest_values] = 1
        if self.save_masks:
            seg_sitk = sitk.GetImageFromArray(segmentation)
            seg_sitk.CopyInformation(self.template_nii)
            sitk.WriteImage(seg_sitk, os.path.join(self.outdir, 'vof_mask.nii.gz'))
        self.vof = np.mean(ctp_array_vof[:, segmentation == 1], axis=1)
        return self.vof


def extract_ctp_array(folder: os.PathLike) -> np.ndarray:
    """
    Extract CTP array from all .nii.gz files in input folder

    Params
    ------
    folder : folder with CTP information

    Returns
    -------
    ctp_array : array with the different CTP frames

    """
    assert os.path.exists(folder), f"Folder '{folder}' does not exist"

    # Load and iterate through each frame file
    ctp_array = []
    files = os.listdir(folder)
    cont_img = 0
    for enum_file, file in enumerate(files):
        if ".nii.gz" in file:
            full_file = os.path.join(folder, file)
            cont_img += 1
            if enum_file == 0:
                image = sitk.ReadImage(full_file)
                img = sitk.GetArrayFromImage(image)
            else:
                img = sitk.GetArrayFromImage(sitk.ReadImage(full_file))
            ctp_array.append(img)
    ctp_array = np.array(ctp_array)
    return ctp_array, image


def get_center_of_mass_index(sitk_image):
    # Label the regions in the image
    label_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = label_filter.Execute(sitk_image)

    # Calculate properties of labeled regions
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(labeled_img)

    # Assuming the label of interest is 1 (often, labeled images start labeling from 1)
    center_of_mass = shape_stats.GetCentroid(1)
    # Convert physical coordinates to index coordinates
    center_of_mass_index = sitk_image.TransformPhysicalPointToIndex(center_of_mass)
    # swap x and z
    center_of_mass_index = (center_of_mass_index[2], center_of_mass_index[1], center_of_mass_index[0])
    return center_of_mass_index


if __name__ == "__main__":
    ctp_folder = "/scratch/amartinezmora/raw_data/ctp_registered_to_atlas/mrclean_late_30002"
    atlas_file = "/scratch/amartinezmora/raw_data/atlas/atlas.nii.gz"
    aif_mask_file = "/scratch/amartinezmora/raw_data/atlas/aif.nii.gz"
    vof_mask_file = "/scratch/amartinezmora/raw_data/atlas/vof.nii.gz"

    # Load CTP data
    ctp_array,image = extract_ctp_array(folder = ctp_folder)
    # Load atlas data
    atlas = sitk.ReadImage(atlas_file)
    # Load masks data
    aif_mask = sitk.ReadImage(aif_mask_file)
    vof_mask = sitk.ReadImage(vof_mask_file)

    # Load AIF and VOF indexes
    aif_index = get_center_of_mass_index(sitk_image=aif_mask)
    vof_index = get_center_of_mass_index(sitk_image=vof_mask)

    # Derive curves
    curves = CurveExtractor(ctp_array, template_nii=atlas, 
                            aif_roi_center=aif_index, 
                            vof_roi_center=vof_index)
    aif = curves.calculate_aif()
    vof = curves.calculate_vof()
    
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange(aif.shape[0]), aif)
    plt.title("AIF")
    plt.subplot(122)
    plt.plot(np.arange(vof.shape[0]), vof)
    plt.title("VOF")
    plt.show()


# example:
#     curves = CurveExtractor(ctp_array, aif_roi_center=aif_index, vof_roi_center=vof_index,
#                             template_nii=atlas, save_masks=True)
#
#     aif = curves.calculate_aif()
#     vof = curves.calculate_vof()