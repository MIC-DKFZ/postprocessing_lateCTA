import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion
import os
import tempfile

def Clipper(scan, minimum=-1024, maximum=1900):
    """
    Clip values of input with numpy.clip, with minimum and maximum as min/max.
    Returns clipped array.

    """
    scan_array = sitk.GetArrayFromImage(scan) # T x D x H x W
    scan_array = np.clip(scan_array, minimum, maximum)
    scan_sitk = np2itk(scan_array, scan)
    return scan_sitk

def RemoveSkull(scan, maximum=150, background=-1024):
    scan_array = sitk.GetArrayFromImage(scan) # T x D x H x W
    scan_array[scan_array>maximum] = background
    scan_sitk = np2itk(scan_array, scan)
    return scan_sitk

def np2itk(arr, original_img):
    if len(arr.shape) == 4:
        t_dim = arr.shape[0]
        frames = []
        for t in range(t_dim):
            frames.append(sitk.GetImageFromArray(arr[t], False))
        img = sitk.JoinSeries(frames)
    elif len(arr.shape) == 3:
        img = sitk.GetImageFromArray(arr, False)

    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    img.CopyInformation(original_img)
    return img

def np_slicewise(mask, funcs, repeats=1):
    """
    Fills holes slice by slice of an 3D np volume
    """
    original = mask
    if isinstance(mask,sitk.SimpleITK.Image):
        mask = sitk.GetArrayFromImage(mask)
    out = np.zeros_like(mask)
    for sliceno in range(mask.shape[0]):
        m = mask[sliceno,:,:]
        for r in range(repeats):
            for func in funcs:
                m = func(m)
        out[sliceno,:,:] = m
    out = np2itk(out, original)
    return out

def np_slicewise_ss(mask, funcs, repeats=1):
    """
    Fills holes slice by slice of an 3D np volume
    """
    original = mask
    if isinstance(mask,sitk.SimpleITK.Image):
        mask = sitk.GetArrayFromImage(mask)
    out = np.zeros_like(mask)
    for sliceno in range(mask.shape[0]):
        m = mask[sliceno,:,:]
        for r in range(repeats):
            for func in funcs:
                m = func(m)
        out[sliceno,:,:] = m
    return out

def LargestConnectedComponent3D(img,min_threshold=0, background=0):
    """
    Retrieves largest connected component mask for 3D sitk
    """
    # compute connected components (in 3D)
    cc = sitk.ConnectedComponent(img>min_threshold)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc,img)
    max_size = 0
    # get largest connected component
    for l in stats.GetLabels():
        if stats.GetPhysicalSize(l)>max_size:
            max_label = l
            max_size = stats.GetPhysicalSize(l)
    # return mask
    return sitk.BinaryThreshold(cc, max_label, max_label+1e-2)


def ApplyMask(mask, scan, foreground_m=1, background=-1024, sitk_type=sitk.sitkInt32):
    """
    Applies mask (m) to 3D volume, sets background of image
    returns and 4D volume with only mask foreground.
    """
    if foreground_m == 0:
        mf = sitk.MaskNegatedImageFilter()
    elif foreground_m == 1:
        mf = sitk.MaskImageFilter()
    if background != None:
        mf.SetOutsideValue(background)
    scan = sitk.Cast(scan, sitk_type)
    mask = sitk.Cast(mask, sitk_type)

    assert np.allclose(scan.GetOrigin(), mask.GetOrigin(), atol=0.01)
    assert np.allclose(scan.GetSpacing(), mask.GetSpacing(), atol=0.01)

    mask.SetOrigin(scan.GetOrigin())
    mask.SetSpacing(scan.GetSpacing())

    result = mf.Execute(scan, mask)

    return result


def register(fixed, moving, seg, parameters, moving_brainmask=None):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetParameterMap(parameters)
    # elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    if moving_brainmask:
        elastixImageFilter.SetMovingMask(moving_brainmask)
    elastixImageFilter.Execute()
    moving_result = elastixImageFilter.GetResultImage()

    # # Get transform parameter
    if seg:
        TranformParameters = elastixImageFilter.GetTransformParameterMap()
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.LogToFileOff()
        transformixImageFilter.SetTransformParameterMap(TranformParameters)
        sitk.PrintParameterMap(transformixImageFilter.GetTransformParameterMap()[0])
        transformixImageFilter.SetTransformParameter('FinalBSplineInterpolationOrder', '0')
        transformixImageFilter.SetTransformParameter('ResultImagePixelType', 'short')
        transformixImageFilter.SetTransformParameter('WriteResultImage', 'false')
        transformixImageFilter.SetTransformParameter('DefaultPixelValue', '0')

        transformixImageFilter.SetMovingImage(seg)
        transformixImageFilter.Execute()
        seg_results = transformixImageFilter.GetResultImage()

        return moving_result, seg_results
    else:
        return moving_result

def register_ctp_frame(fixed, moving, parameters, fixed_brainmask=None):
    fixed_clipped = Clipper(fixed, None, 150)
    moving_clipped = Clipper(moving, None, 150)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetParameterMap(parameters)
    # elastixImageFilter.PrintParameterMap()
    if fixed_brainmask:
        elastixImageFilter.SetFixedMask(fixed_brainmask)
    elastixImageFilter.SetFixedImage(fixed_clipped)
    elastixImageFilter.SetMovingImage(moving_clipped)
    elastixImageFilter.Execute()

    # # Get transform parameter
    TranformParameters = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.SetTransformParameterMap(TranformParameters)
    sitk.PrintParameterMap(transformixImageFilter.GetTransformParameterMap()[0])
    transformixImageFilter.SetTransformParameter('WriteResultImage', 'false')
    transformixImageFilter.SetMovingImage(moving)
    transformixImageFilter.Execute()
    moving_result = transformixImageFilter.GetResultImage()
    return moving_result

def get_transformation_matrix(fixed, moving, parameters, affine=False, clipvalue= [0, 150], moving_brainmask=False):

    if all(elem is None for elem in clipvalue):
        fixed_clipped = fixed
        moving_clipped = moving
    else:
        fixed_clipped = Clipper(fixed, *clipvalue)
        moving_clipped = Clipper(moving, *clipvalue)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()

    if parameters is not None:
        elastixImageFilter.SetParameterMap(parameters)
    else:
        if affine:
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        else:
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    # elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(fixed_clipped)
    elastixImageFilter.SetMovingImage(moving_clipped)
    elastixImageFilter.Execute()
    # # Get transform parameter
    TranformParameters = elastixImageFilter.GetTransformParameterMap()
    return TranformParameters

def get_transformation_matrix_masked(fixed, moving, mask, parameters, affine=False, clipvalue= [0, 150], moving_brainmask=False):

    if all(elem is None for elem in clipvalue):
        fixed_clipped = fixed
        moving_clipped = moving
    else:
        fixed_clipped = Clipper(fixed, *clipvalue)
        moving_clipped = Clipper(moving, *clipvalue)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()

    if parameters is not None:
        elastixImageFilter.SetParameterMap(parameters)
    else:
        if affine:
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        else:
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    # elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(fixed_clipped)
    elastixImageFilter.SetMovingImage(moving_clipped)
    elastixImageFilter.SetMovingMask(mask)
    elastixImageFilter.Execute()
    # # Get transform parameter
    TranformParameters = elastixImageFilter.GetTransformParameterMap()
    return TranformParameters
def apply_transformation(transform_parameters, moving, segmentation, default_pixel = 0):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameters)
    transformixImageFilter.SetTransformParameter('FinalBSplineInterpolationOrder', '1')

    if segmentation:
        transformixImageFilter.SetTransformParameter('FinalBSplineInterpolationOrder', '0')
        transformixImageFilter.SetTransformParameter('ResultImagePixelType', 'short')
        transformixImageFilter.SetTransformParameter('WriteResultImage', 'false')
        transformixImageFilter.SetTransformParameter('DefaultPixelValue', str(default_pixel))

    transformixImageFilter.SetTransformParameter('WriteResultImage', 'false')
    # sitk.PrintParameterMap(transformixImageFilter.GetTransformParameterMap()[0])
    transformixImageFilter.SetMovingImage(moving)
    transformixImageFilter.Execute()
    moving_result = transformixImageFilter.GetResultImage()
    return moving_result

def register_translation(fixed, moving):
    fixed = Clipper(fixed, 0, 300)
    moving = Clipper(moving, 0, 300)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
    # elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.Execute()
    # # Get transform parameter
    TranformParameters = elastixImageFilter.GetTransformParameterMap()
    return TranformParameters
def closing(img):
    elastixImageFilter = sitk.BinaryMorphologicalClosingImageFilter()
    elastixImageFilter.SetKernelRadius([10,10,1])
    elastixImageFilter.SetForegroundValue(1)
    elastixImageFilter.SetKernelType(sitk.sitkBall)
    img = elastixImageFilter.Execute(img)
    return img

def invert_transformation(image: sitk.Image, reference_image: sitk.Image,
                          transformation_matrix, interpolator=sitk.sitkNearestNeighbor):
    # convert back to CTP space
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.SetTransformParameterMap(transformation_matrix)

    center_of_rot = transformixImageFilter.GetTransformParameter(0, 'CenterOfRotationPoint')
    params = transformixImageFilter.GetTransformParameter(0, 'TransformParameters')
    center_of_rot = [float(x) for x in center_of_rot]
    params = [float(x) for x in params]
    transform = sitk.AffineTransform(3)
    transform.SetCenter(center_of_rot)
    transform.SetParameters(params)
    transform.SetInverse()
    default_value = 0
    transformed_image = sitk.Resample(image, reference_image,
                                             transform, interpolator, default_value)
    return transformed_image

def inject_fixed_image_metadata(param_map, fixed_image):
    """
    Inject fixed image metadata on registration parameter map,
    to inform point registration process

    Params
    ------
    param_map : transformation parameter map
    fixed_image : fixed image

    Returns
    -------
    Updated parameter map
    
    """
    # Get image metadata
    spacing = fixed_image.GetSpacing()
    origin = fixed_image.GetOrigin()
    direction = fixed_image.GetDirection()

    # Format direction matrix as flat list
    direction_flat = [str(v) for v in direction]
    spacing_str = [str(s) for s in spacing]
    origin_str = [str(o) for o in origin]

    # Inject metadata into the parameter map
    param_map[0]['Spacing'] = spacing_str
    param_map[0]['Origin'] = origin_str
    param_map[0]['Direction'] = direction_flat

    return param_map

def transform_point(transform_parameters, fixed, moving, point : np.ndarray) -> np.ndarray:
    """
    Transform point with a parameter map

    Params
    ------
    transform_parameters : transformation parameters computed from registration process
    fixed : fixed image used in the registration process
    moving : moving image used in the registration process
    point : input point


    Returns
    -------
    transformed_point : transformed point
    
    """
    # Set up physical point 
    physical_point = moving.TransformIndexToPhysicalPoint(tuple(point.tolist()))

    # Create a Transformix object to transform the point
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transform_parameters)
    transformix.SetMovingImage(moving)
    transformix.LogToConsoleOff()
    transformix.Execute()

    transform_obj = sitk.Transformix().ReadTransformParameterMap(transform_parameters)

    # Transform a single point (landmark) from fixed to moving space
    transformed_physical = transform_obj.TransformPoint(physical_point)

    transformed_index = fixed.TransformPhysicalPointToIndex(transformed_physical)

    print(transformed_physical, transformed_index)

    return transformed_index


def euler_transform_matrix(rx, ry, rz, tx, ty, tz):
    rx, ry, rz = float(rx), float(ry), float(rz)
    tx, ty, tz = float(tx), float(ty), float(tz)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,         1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def transform_point_euler(param_map, fixed, moving, point : np.ndarray) -> np.ndarray:
    """
    Transform point given a parameter map from a registration procedure,
    assuming an Euler method (rotation + translation)

    Params
    ------
    param_map : parameter map from previous registration process
    fixed : fixed image
    moving : moving image
    point : point to register


    Returns
    -------
    transformed_point : transformed point
    
    """
    point = tuple(point.tolist()) # Transform point into tuple 
    rx, ry, rz, tx, ty, tz = param_map[0]['TransformParameters']  

    # 1. Convert voxel index to physical point (in moving space)
    physical_moving = moving.TransformIndexToPhysicalPoint(point)
    print(physical_moving)

    # 2. Apply Euler transform in physical space
    M = euler_transform_matrix(rx, ry, rz, tx, ty, tz)
    moving_physical_hom = np.array([*physical_moving, 1.0])
    print(moving_physical_hom)
    fixed_physical = M @ moving_physical_hom
    print(fixed_physical)

    # 3. Convert to voxel index in fixed image
    fixed_voxel_index = fixed.TransformPhysicalPointToIndex(fixed_physical[:3])

    return np.array(fixed_voxel_index)


def transform_voxel_coordinate(transform_parameters, moving_image, fixed_image, index_coord):
    """
    Transforms a voxel index coordinate using the provided transformation parameters.

    Args:
        transform_parameters: elastix transform parameter map
        moving_image: SimpleITK image (used to convert index -> physical)
        fixed_image : fixed image
        index_coord: list or tuple of 3 integers (i, j, k) in voxel/index space

    Returns:
        transformed_point_phys: list of 3 floats (physical coordinates after transform)
    """
    # Convert index (voxel) to physical coordinate
    point_phys = moving_image.TransformIndexToPhysicalPoint(tuple(index_coord.tolist()))

    # Write the point in required format
    # with tempfile.TemporaryDirectory() as tmpdir:
    input_file = os.path.join(os.getcwd(), "inputpoint.txt")
    output_file = os.path.join(os.getcwd(), "outputpoints.txt")

    with open(input_file, "w") as f:
        f.write("point\n1\n")
        f.write(f"{point_phys[0]} {point_phys[1]} {point_phys[2]}\n")
        f.close()

    # Set up Transformix
    transformix = sitk.TransformixImageFilter()
    transformix.LogToConsoleOff()
    transformix.LogToFileOff()
    transformix.SetTransformParameterMap(transform_parameters)
    transformix.SetFixedPointSetFileName(input_file)
    transformix.SetTransformParameter('WriteResultPointFile', 'true')  # ✅ Required!
    transformix.SetMovingImage(moving_image)  # ✅ Prevents direction error
    transformix.Execute()

    # Read transformed point
    with open(output_file, "r") as f:
        for line in f:
            print(line)
            if line.startswith("Point"):
                for part in line.split(";"):
                    if "OutputPoint" in part:
                        coords = part.split("[")[1].split("]")[0].split()
                        transformed_point_phys = [float(c) for c in coords]
                        transformed_point = fixed_image.TransformPhysicalPointToIndex(transformed_point_phys)
                        print(transformed_point)
                        sys.exit()
                        return np.array(transformed_point)  # This is still in physical space
                    
    return ValueError("Output point could not be read")
