import SimpleITK as sitk

def set_parameters(method='euler', DefaultPixelValue=0, metric='AdvancedMattesMutualInformation'):
    p = sitk.ParameterMap()
    # // Example parameter file for rotation registration
    # // C-style comments:
    if metric == 'AdvancedMattesMutualInformation':
        p['Metric'] = ['AdvancedMattesMutualInformation']
    elif metric == 'AdvancedNormalizedCorrelation':
        p['Metric'] = ['AdvancedNormalizedCorrelation']
    else:
        raise NotImplementedError('Metric not implemented, see parameter file.')
    # // The internal pixel type, used for internal computations
    # // Leave to float in general.
    # // NB: this is not the type of the input images! The pixel
    # // type of the input images is automatically read from the
    # // images themselves.
    # // This setting can be changed to "short" to save some memory
    # // in case of very large 3D images.
    p['FixedInternalImagePixelType'] = ['short']
    p['MovingInternalImagePixelType'] = ['short']
    # // The dimensions of the fixed and moving image
    # // NB: This has to be specified by the user. The dimension of
    # // the images is currently NOT read from the images.
    # // Also note that some other settings may have to specified
    # // for each dimension separately.
    # (FixedImageDimension 3)
    # //(MovingImageDimension 3)
    p['FixedImageDimension'] = ['3']

    # // Specify whether you want to take into account the so-called
    # // direction cosines of the images. Recommended: true.
    # // In some cases, the direction cosines of the image are corrupt,
    # // due to image format conversions for example. In that case, you
    # // may want to set this option to "false".
    p['UseDirectionCosines'] = ['true']

    # // **************** Main Components **************************

    # // The following components should usually be left as they are:
    p['Registration'] = ['MultiResolutionRegistration']
    #p['Interpolator'] = ['BSplineInterpolator']  
    p['Interpolator'] = ['LinearInterpolator']   # TODO: CHANGE THIS!! 
    p['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    p['Resampler'] = ['DefaultResampler']

    # // These may be changed to Fixed/MovingSmoothingImagePyramid.
    # // See the manual.
    # p['FixedImagePyramid'] = ['FixedRecursiveImagePyramid'] 
    # p['MovingImagePyramid'] = ['MovingRecursiveImagePyramid']
    p['FixedImagePyramid'] = ['FixedSmoothingImagePyramid'] # TODO: CHANGE THIS!!
    p['MovingImagePyramid'] = ['MovingSmoothingImagePyramid'] # TODO: CHANGE THIS!!
    # //(FixedImagePyramid "FixedSmoothingImagePyramid")
    # //(MovingImagePyramid "MovingSmoothingImagePyramid")
    # // The following components are most important:
    # // The optimizer AdaptiveStochasticGradientDescent (ASGD) works
    # // quite ok in general. The Transform and Metric are important
    # // and need to be chosen careful for each application. See manual.
    # (MT:AdvancedMattesMutualInformation, Transform option "Affine")
    # (Bob for between modalities MRI-NCCT-CTA: AdvancedNormalizedCorrelation)
    p['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    if method == 'euler':
        p['Transform'] = ['EulerTransform']
    elif method == 'affine':
        p['Transform'] = ['AffineTransform']
    else:
        raise NotImplementedError('Not implemented.')

    # // ***************** Transformation **************************

    # // Scales the rotations compared to the translations, to make
    # // sure they are in the same range. In general, it's best to
    # // use automatic scales estimation:
    p['AutomaticScalesEstimation'] = ['true']

    # // Automatically guess an initial translation by aligning the
    # // geometric centers of the fixed and moving.
    p['AutomaticTransformInitialization'] = ['true']
    # // Whether transforms are combined by composition or by addition.
    # // In generally, Compose is the best option in most cases.
    # // It does not influence the results very much.
    p['HowToCombineTransforms'] = ['Compose']
    # // ******************* Similarity measure *********************

    # // Number of grey level bins in each resolution level,
    # // for the mutual information. 16 or 32 usually works fine.
    # // You could also employ a hierarchical strategy:
    # //(NumberOfHistogramBins 16 32 64)
    # p['NumberOfHistogramBins'] = ['64']
    p['NumberOfHistogramBins'] = ['32'] # TODO: change this!!
    p['ASGDParameterEstimationMethod'] = ['DisplacementDistribution'] # TODO: REMOVE THIS!!
    p['MaximumNumberOfIterations'] = ['2000'] # TODO: REMOVE THIS!!         
    # // If you use a mask, this option is important.
    # // If the mask serves as region of interest, set it to false.
    # // If the mask indicates which pixels are valid, then set it to true.
    # // If you do not use a mask, the option doesn't matter.
    p['ErodeMask'] = ['false']
    # // ******************** Multiresolution **********************

    # // The number of resolutions. 1 Is only enough if the expected
    # // deformations are small. 3 or 4 mostly works fine. For large
    # // images and large deformations, 5 or 6 may even be useful. pm: Pyramid in voxels of geometric space (check anisotropy)
    p['NumberOfResolutions'] = ['4']
    # // The downsampling/blurring factors for the image pyramids.
    # // By default, the images are downsampled by a factor of 2
    # // compared to the next resolution.
    # // So, in 2D, with 4 resolutions, the following schedule is used:
    # //(ImagePyramidSchedule 8 8  4 4  2 2  1 1 )
    # // And in 3D:
    # //(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
    # // You can specify any schedule, for example:
    # //(ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
    # // Make sure that the number of elements equals the number
    # // of resolutions times the image dimension.

    # // ******************* Optimizer ****************************

    # // Maximum number of iterations in each resolution level:
    # // 200-500 works usually fine for rigid registration.
    # // For more robustness, you may increase this to 1000-2000.
    p['MaximumNumberOfIterations'] = ['1000']
    # // The step size of the optimizer, in mm. By default the voxel size is used.
    # // which usually works well. In case of unusual high-resolution images
    # // (eg histology) it is necessary to increase this value a bit, to the size
    # // of the "smallest visible structure" in the image:
    p['MaximumStepLength'] = ['1.0']
    # // **************** Image sampling **********************

    # // Number of spatial samples used to compute the mutual
    # // information (and its derivative) in each iteration.
    # // With an AdaptiveStochasticGradientDescent optimizer,
    # // in combination with the two options below, around 2000
    # // samples may already suffice. pm: choose more samples --> more samples
    # //(NumberOfSpatialSamples 4096) #// use 4096 if to many samples map outside of moving image --> did not work
    p['NumberOfSpatialSamples'] = ['2048'] #// default is 2048
    # // Refresh these spatial samples in every iteration, and select
    # // them randomly. See the manual for information on other sampling
    # // strategies.
    p['NewSamplesEveryIteration'] = ['true']
    #  p['ImageSampler'] = ['Random']
    p['ImageSampler'] = ['RandomCoordinate']  
    # // ************* Interpolation and Resampling ****************

    # // Order of B-Spline interpolation used during registration/optimisation.
    # // It may improve accuracy if you set this to 3. Never use 0.
    # // An order of 1 gives linear interpolation. This is in most
    # // applications a good choice.
    p['BSplineInterpolationOrder'] = ['1']
    # // Order of B-Spline interpolation used for applying the final
    # // deformation.
    # // 3 gives good accuracy; recommended in most cases.
    # // 1 gives worse accuracy (linear interpolation)
    # // 0 gives worst accuracy, but is appropriate for binary images
    # // (masks, segmentations); equivalent to nearest neighbor interpolation.
    p['FinalBSplineInterpolationOrder'] = ['3']
    # //Default pixel value for pixels that come from outside the picture:
    p['DefaultPixelValue'] = [str(DefaultPixelValue)]
    # // Choose whether to generate the deformed moving image.
    # // You can save some time by setting this to false, if you are
    # // only interested in the final (nonrigidly) deformed moving image
    # // for example.
    p['WriteResultImage'] = ['false']
    # // The pixel type and format of the resulting deformed moving image
    p['ResultImagePixelType'] = ['short']
    p['ResultImageFormat'] = ["nii.gz"]
    return p