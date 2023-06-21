# Load and import necessary libraries and functions
import os
import cv2 as cv
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pydicom import dcmread

from scipy.ndimage import binary_opening, binary_erosion, binary_dilation

from skimage.morphology import disk
from skimage.feature import canny
from skimage.filters import frangi, threshold_otsu
from skimage.measure import label
from sklearn.linear_model import LinearRegression

from radiomics import featureextractor

        
    
### ----------------------------SEGMENTATION---------------------------- ###

def breast_mask(image) :
    """For a given dicom image file, creates and returns a mask of the breast using Otsu thresholding."""
    
    # Define Otsu filter to separate background - It finds threshold that separates intensity values in two main groups
    otsu_val = threshold_otsu(image)
    
    # Only consider the foreground i.e. the breast below the threshold in the recombined image
    otsu_array = image < otsu_val
    
    # Find contours in breast
    invert_otsu = (np.ones(otsu_array.shape) - otsu_array).astype(np.uint8)
    (contours,_) = cv.findContours(invert_otsu, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)    

    # Find the contour describing the largest area
    # It is assumed the region within this contour is the breast
    max_ctr = contours[0]
    for ctr in contours:
        if cv.contourArea(ctr) > cv.contourArea(max_ctr):
            max_ctr = ctr
            
    # Create an image that is only 1 within the largest contour
    otsu_mask = np.zeros(invert_otsu.shape)
    cv.fillPoly(otsu_mask, [max_ctr], [1])

    return otsu_mask


def open_mask(mask) :
    """For a given mask, creates and returns its binary opening."""
    
    # Create a disk-shaped footprint of radius 10 and use this to get the binary opening of mask
    se = disk(10)
    mask_open = binary_opening(mask, structure=se, iterations=1)
    
    # Return the binary opened np.array mask_open
    return mask_open


def erode_mask(mask) :    
    """For a given mask, creates and returns its binary erosion."""
    
    # Create a disk-shaped footprint of radius 10 and use this to get the binary erosion of mask
    se = disk(10)
    mask_erode = binary_erosion(mask, structure=se).astype(np.uint16)

    # Return the binary eroded np.array mask_open
    return mask_erode


def remove_pect(image, mask, laterality) :
    """For a given dicom image file, mask and laterality, creates and returns a mask of the breast with the pectoralis removed"""
    
    mask_pect = np.copy(mask)

    # Use Canny edge filter to find edges in the image
    canny_mask = canny(image, sigma=50)
    canny_mask = canny_mask*mask
    canny_mask[canny_mask > 0] = 1

    print(np.unique(canny_mask,return_counts=True))

    # Find connected edges and only save the longer ones
    all_labels = label(canny_mask)
    for l in np.unique(all_labels)[1:] :
        if len(np.where(all_labels == l)[0]) < 200 :
            all_labels[all_labels==l] = 0

    canny_mask = canny_mask*all_labels

    line_found = 0
    while line_found == 0 :

        canny_nz = np.nonzero(canny_mask)
        # print(np.unique(canny_mask,return_counts=True))

        # In a right breast, the pectoralis will be defined by an upper right edge
        if laterality == 'R' :
            xmin = np.min(canny_nz[0])
            xind = np.where(canny_nz[0]==xmin)
            ymin = np.min(canny_nz[1][xind])
            labelind = np.where(all_labels==all_labels[xmin,ymin])

            # Function to fit a straight line through the points found for the pectoralis
            reg_pect = LinearRegression().fit(labelind[0].reshape(-1,1), labelind[1])

            # Angle and interception of fitted line should be realistic
            if reg_pect.coef_[0]>1 or reg_pect.coef_[0]<0 or reg_pect.intercept_<min(np.nonzero(mask[0])[0]):
                print('not found', canny_mask[xmin,ymin], reg_pect.coef_[0], reg_pect.intercept_)
                canny_mask[canny_mask==canny_mask[xmin,ymin]] = 0
                
            else :
                line_found = 1
                print('found', reg_pect.coef_[0], reg_pect.intercept_)
                xmax = (image.shape[1]-reg_pect.intercept_)/reg_pect.coef_[0]
                # All pixels in the mask lying on the right are removed
                for xidx in range(min(np.ceil(xmax).astype('int'),image.shape[0])) :
                    mask_pect[xidx,min(image.shape[1],np.floor(reg_pect.coef_[0]*xidx+reg_pect.intercept_).astype('int')):] = 0


        # In a left breast, the pectoralis will be defined by an upper left edge
        elif laterality == 'L' :
            xmin = np.min(canny_nz[0])
            xind = np.where(canny_nz[0]==xmin)
            ymax = np.max(canny_nz[1][xind])
            labelind = np.where(all_labels==all_labels[xmin,ymax])

            # Function to fit a straight line through the points found for the pectoralis
            reg_pect = LinearRegression().fit(labelind[0].reshape(-1,1), labelind[1])

            # Angle and interception of fitted line should be realistic
            if reg_pect.coef_[0]<-1 or reg_pect.coef_[0]>0 or reg_pect.intercept_>max(np.nonzero(mask[0])[0]):
                print('not found', canny_mask[xmin,ymax], reg_pect.coef_[0], reg_pect.intercept_)
                canny_mask[canny_mask==canny_mask[xmin,ymax]] = 0
            
            else :
                line_found = 1
                print('found', reg_pect.coef_[0], reg_pect.intercept_)
                xmax = -reg_pect.intercept_/reg_pect.coef_[0]
                # All pixels in the mask lying on the left are removed
                for xidx in range(min(image.shape[0],np.ceil(xmax).astype('int'))) :
                    mask_pect[xidx,:max(0,np.ceil(reg_pect.coef_[0]*xidx+reg_pect.intercept_).astype('int'))] = 0

    return reg_pect, mask_pect


def remove_fold(image, mask, laterality, r=1, th=0.9) :
    """For a given dicom image file, mask and laterality, creates and returns a mask of the breast with skin folds removed."""
    
    mask_copy = mask.copy()
    if laterality == 'L' :
        mask_nz = np.nonzero(mask_copy)
        mask_copy[:,int(max(mask_nz[1])/2):] = 0
    elif laterality == 'R' :
        mask_nz = np.nonzero(mask_copy)
        mask_copy[:,:int(mask.shape[1]-min(mask_nz[1])/2)] = 0
    
    edges_frangi = frangi_filter(image, sigma=10)
    # print('Frangi',np.unique(edges_frangi,return_counts=True))

    edges_frangi = edges_frangi*mask_copy
    # print('Frangi mask', np.unique(edges_frangi,return_counts=True))

    edges_frangi[edges_frangi<th] = 0
    # print('Threshold',np.unique(edges_frangi,return_counts=True))

    se = disk(r)
    edges_label = binary_opening(edges_frangi, structure=se)
    # print('Opening',np.unique(edges_label,return_counts=True))

    edges_label = label(edges_label)
    # print('Label',np.unique(edges_label,return_counts=True))

    label_counts = np.unique(edges_label, return_counts=True)
    for idx, lc in enumerate(label_counts[0]):
        if label_counts[1][idx]<3000:
            edges_label[edges_label==lc] = 0
    edges_label[edges_label>0] = 1

    return edges_label*edges_frangi


def canny_filter(image, mask, sigma=50) :

    # Use Canny edge filter to find edges in the image
    canny_mask = canny(image, sigma)
    canny_mask = canny_mask*mask
    canny_mask[canny_mask > 0] = 1

    return binary_dilation(canny_mask)


def frangi_filter(image, sigma=50) :

    # Use Frangi filter to find edges in the image
    frangi_mask = frangi(image, scale_range=(0.1,sigma))

    return frangi_mask


def get_mask(image, output_info, save_masks=True):
    """Get mask of interest and save the calculated masks simultaneously
    with output_info = [specific_output_dir_path, patient_name, laterality, view, state],
    and specifying whether the masks should be saved or not."""
    
    # Calculate the mask of the breast based on Otsu thresholding, and its binary opening and erosion
    mask_breast = breast_mask(image)
    mask_open = open_mask(mask_breast)
    mask_erode = erode_mask(mask_breast)
    
    # Save these masks
    if save_masks:
        save_mask(mask_breast, output_info, 'breast')
        save_mask(mask_open, output_info, 'open')
        save_mask(mask_erode, output_info, 'erode') 
    
    # Specify the mask that'll be used to calculated the nofold mask
    mask_to_use_for_nofold = mask_erode
    
    # If we have an MLO image, calculate the mask in which the pect is removed and specify this as the one to use for nofold
    if output_info[3] == 'MLO':
        reg_test, mask_pect = remove_pect(image.astype('float'), mask_open, output_info[2]) # mask_pect based on mask_open
        mask_pect = erode_mask(mask_pect) # binary erosion of mask_pect
        mask_to_use_for_nofold = mask_pect # the mask that'll be used to calculated the nofold mask
        if save_masks:
            save_mask(mask_pect, output_info, 'pect') # save the eroded mask_pect
                
    # Calculate the mask in which skin folds are removed
    mask_fold = remove_fold(image.astype('float'), mask_to_use_for_nofold, output_info[2])
    mask_nofold = mask_to_use_for_nofold.copy()
    mask_nofold[mask_fold>0] = 0

    # Save the mask_nofold
    if save_masks:
        save_mask(mask_nofold, output_info, 'nofold')
    
    # Return the mask of interest that'll be used for the image analysis
    if output_info[3] == 'MLO': # if the file we consider is in MLO view, return mask_nofold
        return mask_nofold
    else:               # else (if our file is in CC view), return mask_erode
        return mask_erode


def save_mask(mask, output_info, tag):
    """Save a given mask,
    with output_info = [specific_output_dir_path, patient_name, laterality, view, state]
    and a tag indicating what type of mask it is."""

    # Create the path to the directory in which the mask will be saved
    path = output_info[0]
    os.makedirs(path, exist_ok=True)
    
    # Specify the file name of the mask
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_mask_' + tag)
        
    # Save the mask under file_name in the right path
    np.save(os.path.join(path, file_name), mask, allow_pickle=True) # read using np.load('my_file.npy',allow_pickle=True)
    
    
    
### ----------------------------CALCULATION---------------------------- ###


def position_grid(image, mask, roi_xsize, roi_ysize, step_xsize=None, step_ysize=None):
    """Create ROI grid.
       Returns a pandas dataframe with the xmin position, ymin position, width and height of the different ROIs in the grid."""
    
    # If no step size is given, the step size equals the size of the ROIs such that there is no overlap
    if step_xsize==None:
        step_xsize = roi_xsize
    if step_ysize==None:
        step_ysize = roi_ysize
        
    # Create a tuple with the indices of the elements in our mask that are non-zero.    
    mask_nz = np.nonzero(mask)
    
    regCount = 0
    roi_list = []
    
    # Create a pandas dataframe to save all the results in
    column_names = ['xmin', 'ymin', 'width', 'height']
    df_grid = pd.DataFrame(columns = column_names)
    
    print('Create dataframe ROI DONE')
    
    # Fit a grid of ROIs on the mask of the breast - using the tuple of indices
    for row in range(min(mask_nz[0]), min(max(mask_nz[0]),image.shape[0]-roi_xsize) + 1, int(step_xsize)) :
        for col in range(min(mask_nz[1]), min(max(mask_nz[1]),image.shape[1]-roi_ysize) + 1, int(step_ysize)) :
            if np.all(mask[row:row+roi_xsize, col:col+roi_ysize]) :
                regCount += 1
                roi_list.append((row, col, roi_xsize, roi_xsize))
                df_grid.loc[regCount] = [row, col, roi_xsize, roi_ysize] # corr. to 'xmin', 'ymin', 'width', 'height' of the ROI

    print('Create grid DONE', len(roi_list))
    
    return df_grid



def compute_radiomics_breast(image, df_grid) :
    """Compute interesting radiomic features for given regions of interest.
       Returns a pandas dataframe with the different rdmics features as column names and their values for each of the different 
       ROIs on the image following df_grid."""

    # Initialize radiomics settings
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    
    # Initialize feature extractor - only enabling the radiomics features of interest
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Range', 'Variance', 'Skewness', 'Kurtosis'],
                                   glcm=['ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Correlation'],
                                   glszm=['ZoneEntropy','SmallAreaHighGrayLevelEmphasis'],
                                   glrlm=['LongRunEmphasis', 'GrayLevelVariance', 'LongRunHighGrayLevelEmphasis', 'RunVariance'],
                                   ngtdm=['Coarseness','Contrast','Strength'],
                                   gldm=['LargeDependenceEmphasis','DependenceNonUniformity', 'DependenceVariance', 'DependenceEntropy', 'HighGrayLevelEmphasis'])
    
    # Create a pandas dataframe to save all the results in
    column_names = ['range', 'variance', 'skewness', 'kurtosis', 'cluster prominence', 'cluster shade', 'cluster tendency', 'correlation',
                    'zone entropy', 'small area high gray level emphasis', 'long run emphasis', 'gray level variance', 
                    'long run high gray level emphasis','run variance', 'coarseness', 'contrast', 'strength', 'large dependence emphasis',
                    'dependence non-uniformity', 'dependence variance', 'dependence entropy', 'high gray level emphasis']
    df_features = pd.DataFrame(columns = column_names)
    
    sitk_img = sitk.GetImageFromArray(image)
    maskROI = np.zeros_like(image)
    
    df_grid = df_grid.reset_index()  # make sure indexes pair with number of rows

    for idx, roi in df_grid.iterrows() :
        # Extract radiomics features in the different ROIs and save in df_features
        maskROI[roi["xmin"]:roi["xmin"]+roi["width"], roi["ymin"]:roi["ymin"]+roi["height"]] = 1
        sitk_mask = sitk.GetImageFromArray(maskROI)        
        featureVector = extractor.execute(sitk_img, sitk_mask)        
        maskROI[roi["xmin"]:roi["xmin"]+roi["width"], roi["ymin"]:roi["ymin"]+roi["height"]] = 0
        
        df_features.loc[idx+1] = [featureVector['original_firstorder_Range'][()], featureVector['original_firstorder_Variance'][()],
                                featureVector['original_firstorder_Skewness'][()], featureVector['original_firstorder_Kurtosis'][()],
                                featureVector['original_glcm_ClusterProminence'][()], featureVector['original_glcm_ClusterShade'][()],
                                featureVector['original_glcm_ClusterTendency'][()], featureVector['original_glcm_Correlation'][()],
                                featureVector['original_glszm_ZoneEntropy'][()], featureVector['original_glszm_SmallAreaHighGrayLevelEmphasis'][()],
                                featureVector['original_glrlm_LongRunEmphasis'][()], featureVector['original_glrlm_GrayLevelVariance'][()],
                                featureVector['original_glrlm_LongRunHighGrayLevelEmphasis'][()],
                                featureVector['original_glrlm_RunVariance'][()], featureVector['original_ngtdm_Coarseness'][()],
                                featureVector['original_ngtdm_Contrast'][()], featureVector['original_ngtdm_Strength'][()],
                                featureVector['original_gldm_LargeDependenceEmphasis'][()], featureVector['original_gldm_DependenceNonUniformity'][()],
                                featureVector['original_gldm_DependenceVariance'][()], featureVector['original_gldm_DependenceEntropy'][()],
                                featureVector['original_gldm_HighGrayLevelEmphasis'][()]]
        
    
    print('Compute radiomics DONE')
    
    # Return dataframe with all extracted features
    return df_features


def sort_radiomics(image, df_grid, df_rmics) :
    """Sort computed radiomics and return them with according ROI.
       Returns a pandas dataframe with the concatenation of df_grid and df_rmics
       and a dictionary with the names of the different rmics features as keys and as corr. values a sorted df_all dataframe 
       based on the specified rmcis feature."""

    # Concatenate the dataframe corr. to the grid points with the one corr. to the calculated radiomics values for said points
    df_all = pd.concat([df_grid, df_rmics], axis=1)
    
    # Sort the ROIs multiple times, each time based on one of the radiomics features
    df_sorted_each_feature={}
    for column in df_rmics:
        asc = False
        # for "skewness","kurtosis","long run emphasis",'run variance',"large dependence emphasis" and "dependence variance" the ROIs should be sorted asccendingly instead of descendingly
        if str(column) in ["skewness","kurtosis","long run emphasis",'run variance',"large dependence emphasis","dependence variance"]:
            asc = True
        df_all_temp = df_all.copy()
        df_sorted_each_feature[str(column)]= df_all_temp.reindex(df_all_temp[str(column)].abs().sort_values(ascending=asc).index)

    print('Sort df DONE')
    
    # Return the dataframe with the concatenated information of the ROI grid and the radiomics
    #    and the dictionary existing of different dataframes similar to df_all sorted for the different radiomics features 
    return df_all, df_sorted_each_feature


def get_highest_scoring_radiomics(df_all, df_sorted, nb):
    """Display images with the nb highest scoring ROIs for each feature.
       Returns a dict with the coords of the nb highest scoring ROIs based on different features, with featureName as key."""
    
    # large range due to clips: to be removed as possible location - using IQR
    Q1 = np.percentile(df_all['range'], 25, interpolation = 'midpoint')       
    Q3 = np.percentile(df_all['range'], 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1
    
    # Create a dictionary to save the results in
    dict_roi_coord = {}
    
    # If nb is larger than the amount of grid points, put it equal to the amount of grid points
    if nb > len(df_all['range']):
        nb = len(df_all['range'])
    
    # Loop over all different radiomics features
    for featureName in df_sorted:
        
        # Iterate over the rows in the dataframe (item of the dict) sorted for a certain feature
        df_iter = df_sorted[featureName].iterrows()
        temp_list_roi_coord = []
        roi_count = 0
        
        # Add the coordinates of the nb highest ROIs (by running over the sorted df) to temp_list_roi_coord
        while roi_count < nb : # run as long as we don't have selected nb ROIs
            
            try: # try to go to next iteration
                idx, row = next(df_iter)
            except StopIteration as e: # catch StopIteration as exception when no next item in df_iter 
                print("StopIteration error handled successfully")
                break
                
            if (Q1-3*IQR) < row['range'] < (Q3+3*IQR) : # only allow ROIs with not too large intensity range (avoiding clips)
                roi_count += 1
                temp_list_roi_coord.append([row['ymin'], row['xmin'], row['height'], row['width']])

        # Add the list of coordinates together with their corresponding feature to dict_roi_coord
        dict_roi_coord[str(featureName)] = temp_list_roi_coord
    
    # Return dictionary with the coordinates for the nb highest ROI based on different features (with featureName as key)
    return dict_roi_coord


def find_optimal_region(highest_roi_coord, nb): 
    """Find the average nb optimal regions of the ROIs based on the different highest scoring ROI for the different features as
       given in highest_roi_coord.
       Returns a list of the nb most prevalent coordinates corr. to these nb optimal regions."""
    
    # If nb is larger than the amount of saved ROI coordinates, put it equal to the amount of saved coords
    if nb > len(highest_roi_coord["range"]):
        nb = len(highest_roi_coord["range"])
    
    # Create a dictionary to save the results in
    dict_optimal_coord = {}
    
    
    # If highest_roi_coord is a dictionary, loop over all the different features and then the different coordinates corr. to 
    # those featuresElse if highest_roi_coord is an np.array (i.e. corr. to only one feature), loop over all coords in the array
    #
    #    For each of the coordinates, make a tuple of the coords and use this as the keyname for dict_optimal_coord
    #    Keep count of the nb of the repeating coordinates by adding +1 to each dict[keyname=coords] by making use of
    #    dict.get(keyname, value=0), where value=0 indicates that the value for dict[keyname] is 0 if there is not yet another 
    #    value corresponding to it
        
    if isinstance(highest_roi_coord, dict):
        for feature_name in highest_roi_coord:
            
            for coord_list in highest_roi_coord[feature_name]:

                yx_coord_tuple = (coord_list[0],coord_list[1])

                dict_optimal_coord[yx_coord_tuple] = dict_optimal_coord.get(yx_coord_tuple, 0) + 1
    
    elif isinstance(highest_roi_coord, np.ndarray):
        for i in range(len(highest_roi_coord)):
            yx_coord_tuple = (highest_roi_coord[i][0],highest_roi_coord[i][1])

            dict_optimal_coord[yx_coord_tuple] = dict_optimal_coord.get(yx_coord_tuple, 0) + 1
            
    # Sort the dict based on the values (=prevalence) corr. to the different keys (=coords) - ascending
    sorted_dict_optimal_coord = {k: v for k, v in sorted(dict_optimal_coord.items(), key = lambda item : item[1])}
    
    # Only keep the nb most prevalent coordinates, i.e. the nb last coordinates in the sorted dict
    optimal_coord = list(sorted_dict_optimal_coord.keys())[-nb:]
    
    # Return a list of the nb most prevalent coordinates among all elements of highest_roi_coord
    optimal_coord.reverse()
    return optimal_coord


def get_first_optimal_region(img, optimal_roi_coord, roi_xsize=200, roi_ysize=200):
    """Get the highest scoring optimal region and return a np.array corresponding to the values of img in the optimal region,
       together with the centres of the box wrt the image : img[x_centre][y_centre]"""
    
    # Specify optimal y- and x-coord
    point_y = optimal_roi_coord[0][0]; y_centre = point_y + int(roi_ysize/2)
    point_x = optimal_roi_coord[0][1]; x_centre = point_x + int(roi_xsize/2)

    # Initiate box
    box = []
    ind = 0

    # Fill box with the row values [point_y:point_y+roi_ysize] of img starting from point_x to point_x+roi_xsize
    while ind < roi_xsize:
        box.append((img[point_x+ind][point_y:point_y+roi_ysize]))
        ind += 1

    #np.save(os.path.join(path, file_name + '_first_optimal_ROI_box' + tag), np.array(box), allow_pickle=True)

    return np.array(box), x_centre, y_centre


def frangi_filter_applied(box_image, threshold):
    """Apply the frangi filter from skimage.filters to box_image and keep only the values below a certain quantile threshold.
       Returns a boolean np.array of the frangi filtered image with applied threshold -- can act like a mask."""
    
    # Apply frangi filter
    frangi_box = frangi(box_image, sigmas=range(1,10,2), alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0)
    
    # Apply threshold
    frangi_box[frangi_box > np.quantile(frangi_box, threshold)] = 0
    
    # Make boolean array of our frangi image (w/ threshold applied): all values==0 -> False, values!=0 -> True
    frangi_boolean_array = np.array(frangi_box, dtype=bool)
    
    # Return the boolean array
    return frangi_boolean_array


def opening_frangi(frangi_boolean_array):
    """Returns boolean np.array onto which binary_opening from scipy.ndimage is applied."""
    
    # Return binary opened boolean array
    return binary_opening(frangi_boolean_array)


def intensity_mask(box_image, threshold):
    """Create a mask based on the values (pixel/intensity) of box_image and a certain given threshold.
       Returns a binary mask based on this threshold as a np.array."""
    
    # Copy the given image box_image
    box_thresh_mask = np.copy(box_image)
    
    # Make a binary mask of the copy by putting everything below a certain quantile threshold to 0 and everything else to 1
    box_thresh_mask[box_thresh_mask < np.quantile(box_thresh_mask, threshold)] = 0
    box_thresh_mask[box_thresh_mask >= np.quantile(box_thresh_mask, threshold)] = 1
    
    # Return this binary mask
    return box_thresh_mask


def combined_frangi_intensity_mask(frangi_mask, intensity_mask):
    """Combine the intensity mask and the binary opening of the frangi mask.
       Returns a binary mask as a np.array."""
    
    # Multiply the intensity mask and the integer version of the binary opening of the frangi mask
    multpl_mask = (intensity_mask)*binary_opening(frangi_mask).astype(int)
    
    # Return the multiplied mask
    return multpl_mask


def size_circularity_conditions(mask, threshold_circularity=0.5, threshold_size=12):
    """Apply conditions to the possible calcifications, represented in mask, based on their size and circularity.
       Returns the mask to which the conditions are applied as a np.array."""
    
    # Use label from skimage.measure to create a labeled mask in which all pixels with 1-connectivity have the same value, 
    #    where mask serves as the image to label and labeled_mask is the - values that were 0 in mask remain 0 
    labeled_mask = label(mask, connectivity=1)

    # Run over all unique values (corr. to "connected" calcifications) in labeled_mask, excluding 0
    for i in np.delete(np.unique(labeled_mask),0):
        
        # The surface of a calcification equals the amount of pixels with the corresponding value i
        surface = np.count_nonzero(labeled_mask == i)
        
        # Run over all columns by using the transpose of labeled_mask
        # Keep record of the indices of the columns in which our calcification w/ value i lies
        columns = []
        for count_c, column in enumerate(np.transpose(labeled_mask)):
            if i in column:
                columns.append(count_c)
        
        # Run over all rows of labeled_mask and keep record of the indices of the rows in which our calcification w/ value i lies
        rows = []
        for count_r, row in enumerate(labeled_mask):
            if i in row: 
                rows.append(count_r)
        
        # Use the indices of the rows and columns to get the y- and x-size
        # Here, the length & width correspond to those of the smallest rectangle enclosing the calcification (oriented as grid)
        y_size = max(rows) - min(rows) + 1 # = height
        x_size = max(columns) - min(columns) + 1 # = width
        
        # Calculate the perimeter, using the largest one of height and width as the diameter
        perimeter = np.pi*max(y_size,x_size)
        
        # Condition 1 - size : the calcification cannot be longer than the threshold (in number of pixels) in width or height
        if (x_size > threshold_size) or (y_size > threshold_size):
            labeled_mask[labeled_mask == i] = 0 # removing the calc from the mask by putting all values==i to 0
        
        # Condition 2 - circularity : the circularity of our calc should cannot be lower than the threshold -- for a circle, circularity = 1
        if (4*np.pi*surface/(perimeter**2)) < threshold_circularity: # or (4*np.pi*surface/(perimeter**2)) > 1.5:
            labeled_mask[labeled_mask == i] = 0 # removing the calc from the mask by putting all values==i to 0
            
    # Return a np.array which acts as a mask where all each calcification has a unique non-zero value - acc. to the conditions
    return labeled_mask


def compute_contrast_box(image, df_grid) :
    """Compute interesting radiomic features for given regions of interest."""

    # Initialize radiomics settings
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Range'], ngtdm=['Contrast'])
    
    # Compute range of intensity features
    column_names = ['range', 'contrast']
    df_features = pd.DataFrame(columns = column_names)
    
    sitk_img = sitk.GetImageFromArray(image)
    maskROI = np.zeros_like(image)
    
    df_grid = df_grid.reset_index()  # make sure indexes pair with number of rows

    for idx, roi in df_grid.iterrows() :
        # Extract radiomics features
        maskROI[roi["xmin"]:roi["xmin"]+roi["width"], roi["ymin"]:roi["ymin"]+roi["height"]] = 1
        sitk_mask = sitk.GetImageFromArray(maskROI)        
        featureVector = extractor.execute(sitk_img, sitk_mask)        
        maskROI[roi["xmin"]:roi["xmin"]+roi["width"], roi["ymin"]:roi["ymin"]+roi["height"]] = 0
        
        df_features.loc[idx+1] = [featureVector['original_firstorder_Range'][()] ,featureVector['original_ngtdm_Contrast'][()]]
        
    # Return dataframe with all extracted features
    return df_features


def get_highest_scoring_contrast_box_considering_mask(mask, df_all, df_sorted, nb=25, strict=True):
    """Find the highest scoring ROIs according to their contrast, containing +nb calcifications. If strict=True, the number of
       calcs will be exactly nb. Else, it might be a bit more if the last ROI added contains more than one new unique calc.
       The given mask should be of the form of a labeled mask of calcifications.
       Returns a list of the coords corr. to these ROIs, and a list with the unique label values of the corr. calcs."""
    
    # Large range due to clips: to be removed as possible location -- using IQR
    Q1 = np.percentile(df_all['range'], 25, interpolation = 'midpoint')       
    Q3 = np.percentile(df_all['range'], 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1

    # Create two lists to save the results in
    array_roi_coord = [] # will contain the coordinates of the ROI of interest
    uniquely_added_calcs = [] # will contain the unique label values for the different calcs present in the ROIs
    
    # Iterate over the different rows in df_sorted[contrast] - dataframe sorted acc. to contrast values
    # Each row represents a box
    df_iter = df_sorted['contrast'].iterrows()
    
    while len(uniquely_added_calcs) < nb: # run as long as we don't have selected nb unique calcs
        
        try: # try to go to next iteration
            idx, row = next(df_iter)
        except StopIteration as e: # catch StopIteration as exception when no next item in df_iter 
            print("StopIteration error handled successfully")
            break
            
        add = False # Initially assume that the row considered, doesn't contain a calc and thus shouldn't be added
        
        if (Q1-3*IQR) < row['range'] < (Q3+3*IQR) : # large range condition
            
            # Loop over all elements in mask that correspond to the ROI of the row of df_sorted['contrast'] considered
            for i in np.unique(mask[row['xmin']:row['xmin']+row['height'], 
                                    row['ymin']:row['ymin']+row['width']]) :
                    
                # If element i!=0 , it corresponds to a calcification (acc. to mask) and the ROI considered should be added
                if i!=0:
                    add = True 
                    
                    if i not in uniquely_added_calcs: # this specific calc is not yet considered - a new calc
                        # If strict=False, append the unique label value i for the calc to uniquely_added_calcs
                        if not strict:
                            uniquely_added_calcs.append(i)
                        # If strict=True, only append i to uniquely_added_calcs if len(uniquely_added_calcs) < nb
                        elif strict and len(uniquely_added_calcs) < nb : # not yet nb calcs
                            uniquely_added_calcs.append(i)
                            
        
        if add: # if a calc is present in the ROI, append its coords to array_roi_coord
            array_roi_coord.append([row['ymin'], row['xmin'], row['height'], row['width']])
    
    # Return the two lists
    return array_roi_coord, uniquely_added_calcs


def get_highest_scoring_contrast_box_vicinity(mask, df_all, df_sorted, nb=25):
    """Find the highest scoring ROIs according to their contrast and in each other vicinity, containing +nb calcifications.
           Starting from the first highest scoring ROI, a search region will be build up around it and expanded based on the
           newly added highest scoring ROI within this region. As such, the resulting ROI and their calcs will lie more closely
           together instead of spread out over the whole box. This is done to better simulate the cluster behaviour.
       If strict=True, the number of calcs will be exactly nb. Else, it might be a bit more if the last ROI added contains more
       than one new unique calc. The given mask should be of the form of a labeled mask of calcifications.
       Returns an np.array of the coords corr. to these ROIs, and an np.array with the unique label values of the corr. calcs."""
    
    # large range due to clips: to be removed as possible location -- using IQR
    Q1 = np.percentile(df_all['range'], 25, interpolation = 'midpoint')       
    Q3 = np.percentile(df_all['range'], 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1

    # Make copy of df_sorted['contrast'] -> pandas dataframe df_sorted_search
    df_sorted_search = df_sorted['contrast'].copy()
    # Iterate over the different rows in df_sorted_search - dataframe sorted acc. to contrast values
    # Each row represents a box
    df_iter = df_sorted_search.iterrows()
    
    # Get first ROI, with highest contrast value and containing at least one calc, using full grid
    array_roi_coord, uniquely_added_calcs = get_highest_scoring_contrast_box_considering_mask(mask, df_all, df_sorted, 1, False)
    
    # Get mask representing the search region around the first ROI found above (array_roi_coord[0]). As initial search mask
    # input, use an np.array of zeros with the same shape as our mask. After applying the functino, the ROI and its vicinity
    # have value 1, everything else has value 0. Within this region, we will look for the next highest contrast ROI with calcs.
    search_mask = get_mask_search_region(np.zeros(np.shape(mask)), array_roi_coord[0])
    search_extended_mask = search_mask.copy()

    while len(uniquely_added_calcs) < nb: # run as long as we don't have selected nb ROIs
        plt.imshow(search_mask)
        
        try: # try to go to next iteration
            idx, row = next(df_iter)
            print('Try region', idx, 'xmin', row['xmin'], 'ymin', row['ymin'])
        except StopIteration as e: # catch StopIteration as exception when no next item in df_iter 
            print("StopIteration error handled successfully")
            break
        
        inSearchMask = True # Initially assume that the row considered, can be added as it's in the search_mask
        containsCalcInMask = False # Initially assume that the row considered, doesn't contain a calc in the search_maskInMask
        temp_new_unique_calcs = [] # Make temporary list keeping account of new unique calcs in ROI
        
        if (Q1-3*IQR) < row['range'] < (Q3+3*IQR) : # large range condition
            
            
            # If the ROI is not completely in the search region, it is not further considered
            if np.amin(search_mask[row['xmin']:row['xmin']+row['height'], row['ymin']:row['ymin']+row['width']]) < 1 :
                inSearchMask = False
            # Loop over all elements in mask that correspond to the ROI of the row of df_sorted_search considered
            else :
                for i in np.unique(mask[row['xmin']:row['xmin']+row['height'], 
                                        row['ymin']:row['ymin']+row['width']]) :

                    # If element i!=0 and all the elements in search_mask corr. to the ROI that is being considered are 1 (i.e.
                    # in the vicinity based search region), it corresponds to a calc which can be added
                    if i!=0 :
                        containsCalcInMask = True
                        
                        # If i not in uniquely_added_calcs this specific calc is not yet considered (new calc) and i should be
                        # appended to uniquely_added_calcs only if len(uniquely_added_calcs) < nb (don't have nb calcs yet)
                        if i not in uniquely_added_calcs and len(uniquely_added_calcs) < nb :
                            temp_new_unique_calcs.append(i)

        print('In mask', inSearchMask, 'Calc found', containsCalcInMask, temp_new_unique_calcs)           
        if inSearchMask and containsCalcInMask : 
            # Add the new unique calcs to the list of uniquely added calcs
            uniquely_added_calcs.extend(temp_new_unique_calcs)
            
            # The coords of the ROI in the search region in which at least one calc is present, is appended to array_roi_coord
            new_roi_coord = [row['ymin'], row['xmin'], row['height'], row['width']]
            array_roi_coord.append(new_roi_coord)
            
            # Update the current search region search_mask with the newly added ROI new_roi_coord
            search_mask = get_mask_search_region(search_mask, new_roi_coord)
            
            # Remove the row from which we added the ROI from our sorted dataframe df_sorted_search 
            # and reinitialise the iteration using this adjusted df_sorted_search such that the ROI won't be considered again
            df_sorted_search_transpose = df_sorted_search.T
            df_sorted_search_transpose.pop(idx)
            df_sorted_search = df_sorted_search_transpose.T
            
            df_iter = df_sorted_search.iterrows()

        # Keep an alternative in case the search mask will at some points be zero
        # before the number of calcs has been reached
        elif inSearchMask :
            extend_roi_coord = [row['ymin'], row['xmin'], row['height'], row['width']]
            
            # Update the current search region search_mask with the newly added ROI new_roi_coord
            search_extended_mask = get_mask_search_region(search_extended_mask, extend_roi_coord)
            
            # Remove the row from which we added the ROI from our sorted dataframe df_sorted_search 
            # and reinitialise the iteration using this adjusted df_sorted_search such that the ROI won't be considered again
            df_sorted_search_transpose = df_sorted_search.T
            df_sorted_search_transpose.pop(idx)
            df_sorted_search = df_sorted_search_transpose.T
            
            df_iter = df_sorted_search.iterrows()

        # In case no calc was found in the search region, but the number of calcs was not yet reached,
        # the search region is extended in all directions
        if np.count_nonzero(search_mask) == 0 :
            print('Zero search region')
            search_mask = search_extended_mask
    
    # Return the coordinates of our ROI and the unique label values of the calcs as two np.arrays
    return np.array(array_roi_coord), np.array(uniquely_added_calcs)


def get_mask_search_region(search_mask, roi):
    """Expand search_mask by changing all values in roi and those one roi-size around roi (in all directions) to 1, 
       leaving the rest unchanged. Here, roi = [row['ymin'], row['xmin'], row['height'], row['width']].
       Returns updated search_mask as the same dtype as the given search_mask - normally np.array."""
    
    # Get the min and max row and column for the expansion of the current roi in all directions by one roi-size
    
    row_min = roi[1] - roi[2] # vertically one roi up to top current roi ~ row['xmin'] - row['height']
    if row_min < 0: # outside box -> put row_min to 0
        row_min = 0
        
    row_max = roi[1] + 2*roi[2] # vertically one roi down to bottom current roi ~ row['xmin'] + 2*row['height']
    if row_max > len(search_mask): # outside box -> put row_max to len(search_mask)
        row_max = len(search_mask)
        
    column_min = roi[0] - roi[3] # horizontally one roi left to left side current roi ~ row['ymin'] - row['width']
    if column_min < 0: # outside box -> put column_min to 0
        column_min = 0
    
    column_max = roi[0] + 2*roi[3] # horizontally one roi right to right side current roi ~ row['ymin'] + 2*row['width']
    if column_max > len(search_mask[0]): # outside box -> put column_max to len(search_mask[0])
        column_max = len(search_mask[0])
        
    # Put all elements of search_mask in this expanded region to 1
    search_mask[row_min:row_max, column_min:column_max] = 1
    
    # Return an adjusted search region as (np.)array search_mask
    return search_mask


def combine_contrast_mask(mask, uniquely_added_calcs):
    """Combine mask (with the calcs) with uniquely_added_calcs such that only the calcs with correspondance in 
       uniquely_added_calcs remain.
       Returns a mask as an np.array of all the calcs whose labeled value are present in uniquely_added_calcs."""
    
    # Make a copy of mask
    resulted_mask = np.copy(mask)
    
    # Put all values in resulted_mask that do not occur in uniquely_added_calcs to 0
    for i in np.delete(np.unique(mask),0):
        if i not in uniquely_added_calcs:
            resulted_mask[resulted_mask == i] = 0
    
    # Return an np.array in which only the values present in uniquely_added_calcs aren't 0
    return resulted_mask


def correct_raytracing(buffer, thickness) :
    """Correct intensity values of buffer as if it was produced by ray tracing."""

    # Coefs in the middle of the image
    # distance_coef = -0.08450854
    # thickness_coef = 0.01242449
    # intercept = 0.8885715457389214

    # Coefs at the "average" position
    distance_coef = -0.08833005
    thickness_coef = 0.02494807 # measured for thickness in cm
    intercept = 0.813907506186984

    # # Coefs at the "average" position after gauss with sima=0.5
    # distance_coef = -0.105885
    # thickness_coef = 0.02125417 # measured for thickness in cm
    # intercept = 0.9299081957897973

    buffer_raytr = buffer.astype(float)*distance_coef + float(thickness)/10*thickness_coef + intercept
    buffer_raytr[buffer == 0] = 1

    return buffer_raytr
