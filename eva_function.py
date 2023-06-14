#!/usr/bin/env python
# coding: utf-8


## Load packages
import os
import random
import numpy as np

from matplotlib.patches import Rectangle

from pydicom import dcmread
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from skimage.measure import label
import matplotlib.pyplot as plt

# from .utils_eva import *
from rood_blauw import *
from utils_plotting import plot_masks, plot_grid, plot_highest_scoring_roi, plot_optimal_region, full_extent, plot_third_optimal_region, plot_frangi_intensity_combined_mask, plot_filtered_mask, plot_highest_scoring_contrast_box, plot_final_mask, plot_first_optimal_region

def get_malignant_calc_cluster(image_path, output_dir_path, masks_saved=False, nb_calcs_random=True):
    """ Compute malignant calc cluster, with edt and raytracing transformation already applied
        Input: image_path - path to the dicom image file
               output_dir_path - path to the specific directory in which output and possibly masks should be saved
        Output: buffer - 2D numpy array corr. to the calc cluster in a 200x200 pixel grid point
                X, Y - pixel position of the center of the grid point of the cluster with respect to the full image. """

    roi_xsize = 200; roi_ysize = 200
    nb_calcs_range = (15,25,nb_calcs_random)
    threshold_circularity = 0.2
    threshold_size = 12
    gauss_sigma = 1
    
    buffer_raytr, X, Y = get_calc_cluster(image_path, output_dir_path,                   
                     roi_xsize=roi_xsize, roi_ysize=roi_ysize, masks_saved=masks_saved, 
                     nb_calcs_range=nb_calcs_range, threshold_frangi=0.4, threshold_intensity=0.5, 
                     threshold_circularity=threshold_circularity, threshold_size=threshold_size, gauss_sigma=gauss_sigma)

    return buffer_raytr, X, Y
    
    
def get_benign_calc_cluster(image_path, output_dir_path, masks_saved=False, nb_calcs_random=True):
    """ Compute bening calc cluster, with edt and raytracing transformation already applied
        Input: image_path - path to the dicom image file
               output_dir_path - path to the specific directory in which output and possibly masks should be saved
        Output: buffer - 2D numpy array corr. to the calc cluster in a 200x200 pixel grid point
                X, Y - pixel position of the center of the grid point of the cluster with respect to the full image. """

    roi_xsize = 200 #60
    roi_ysize = 200 #60
    nb_calcs_range = (7,17,nb_calcs_random)
    threshold_circularity = 0.7
    threshold_size = 6
    gauss_sigma = 1
    
    buffer_raytr, X, Y = get_calc_cluster(image_path, output_dir_path,                   
                     roi_xsize=roi_xsize, roi_ysize=roi_ysize, masks_saved=masks_saved, 
                     nb_calcs_range=nb_calcs_range, threshold_frangi=0.4, threshold_intensity=0.5, 
                     threshold_circularity=threshold_circularity, threshold_size=threshold_size, gauss_sigma=gauss_sigma)

    return buffer_raytr, X, Y
    


## Compute the mask, grid, features, optimal region, local features within this region (frangi, intensity and contrast)
##    and get calc cluster
def get_calc_cluster(image_path, output_dir_path, 
                     roi_xsize=200, roi_ysize=200, masks_saved=False, 
                     nb_calcs_range=(15,25,True), 
                     threshold_frangi=0.4, threshold_intensity=0.5, threshold_circularity=0.5, threshold_size=12,
                     gauss_sigma=0.5):
    """ Compute calc cluster, with edt and raytracing transformation already applied
        Input: image_path - path to the dicom image file
               output_dir_path - path to the specific directory in which output and possibly masks should be saved
               nb_calcs_range = (min[int],max[int],random[bool]) - if random=True, a random nb of calcs will be chosen between min and max (inclus)
        Output: buffer - 2D numpy array corr. to the calc cluster in a 200x200 pixel grid point
                X, Y - pixel position of the center of the grid point of the cluster with respect to the full image. """

    dcm_img = dcmread(image_path) # read the dicom input data file
    image = dcm_img.pixel_array # array representing (the pixels of) the image corresponding to our input data file

    
    # Get patient name, laterality, view and state info
    patient_name = str(dcm_img.PatientName)
    laterality = str(dcm_img.ImageLaterality)
    view = str(dcm_img.ViewPosition)
    state = str(dcm_img.ImageType[3])
    thickness = str(dcm_img.BodyPartThickness)
    
    # Put this information together in output_info
    output_info = [output_dir_path, patient_name, laterality, view, state]
    
    
    # Get the mask of the breast that will be used for further calculations: mask_erode for CC, mask_nofold for MLO
    # If specified that masks have already been saved, load the one of interest
    if masks_saved:
        if view == 'MLO': # in MLO view, we used the mask_nofold   
            mask = np.load(os.path.join(output_dir_path,patient_name+'_'+laterality+'_'+view+'_'+state.upper()+'_mask_nofold.npy'), allow_pickle=True)
        else:             # else (in CC view), we use mask_erode
            mask = np.load(os.path.join(output_dir_path,patient_name+'_'+laterality+'_'+view+'_'+state.upper()+'_mask_erode.npy'), allow_pickle=True)
    # Else, compute the mask of interest - and specify whether the computed masks should be saved
    else:
        mask = get_mask(image, output_info, save_masks=False)

        
    # Create a grid of certain size onto our image    
    df_grid = position_grid(image, mask, roi_xsize, roi_ysize) 
    
    # Compute certain (22) radiomics features of the image for each grid point
    df_features = compute_radiomics_breast(image, df_grid)
    # Sort the grid points based on the computed radiomics features (for each feature separately)
    df_all, df_sorted = sort_radiomics(image, df_grid, df_features)
    # For each radiomics feature, select the 10 grid points with the "highest scores"
    nb_optimal_roi = 5
    highest_roi_coord = get_highest_scoring_radiomics(df_all, df_sorted, nb_optimal_roi*2) 

    
    # Find nb highest ROIs averaged over the different features
    optimal_roi_coord = find_optimal_region(highest_roi_coord, nb_optimal_roi)
    # Get the first highest scoring ROI from the averaged 5 optimal ROIs,
    # with X, Y the centres of the box wrt the image : image[X][Y]
    box, X, Y = get_first_optimal_region(image, optimal_roi_coord, roi_xsize, roi_ysize)
    
    
    # Apply frangi filter with threshold on box
    frangi_boolean_array = frangi_filter_applied(box, threshold_frangi)
    # Use binary opening to make mask of the frangi filtered box
    frangi_mask = opening_frangi(frangi_boolean_array) 
    
    # Mask of the box based on an intensity threshold
    intensity_threshold_mask = intensity_mask(box, threshold_intensity) 

    # Get the combined mask of the frangi and the intensity based masks by multiplying them
    combined_mask = combined_frangi_intensity_mask(frangi_mask, intensity_threshold_mask) 
    
    # Apply conditions based on size and circularity on the combined mask to filter it even further
    filtered_mask = size_circularity_conditions(combined_mask, threshold_circularity, threshold_size)
    
    # Create a grid of a certain size onto our box with grid points overlapping each other by half
    roi_box_xsize=10; roi_box_ysize=10
    df_roi_box = position_grid(box, box, roi_box_xsize, roi_box_ysize, step_xsize=roi_box_xsize/2, step_ysize=roi_box_ysize/2) 
    
    # Compute the radiomics feature "contrast" for the gridpoints on the box
    df_features_box = compute_contrast_box(box, df_roi_box) 
    # Sort the gridpoint of the box based on their radiomics feature (here: contrast)
    df_all_box, df_sorted_box = sort_radiomics(box, df_roi_box, df_features_box) 

    
    # As standard take nb calcs to be 25
    nb_calcs = nb_calcs_range[1]
    if nb_calcs_range[2]: # if specified that the nb of calcs should be chosen randomly
        nb_calcs = random.randrange(nb_calcs_range[0],nb_calcs_range[1]+1) # use uniform distr to take random nb between 15 and 25 as the nb of calcs (stop = 26 as 26 not included)
        print(str(nb_calcs)+' unique calcs')
        
    # Get the highest scoring ROI, containing in total nb_calcs calcifications, based on the contrast in said ROI, the presence of calcifications and the vicinity of the ROI
    highest_contrast_array_box, uniquely_added_calcs = get_highest_scoring_contrast_box_vicinity(filtered_mask, df_all_box, df_sorted_box, nb_calcs)
    plot_highest_scoring_contrast_box(box, filtered_mask, highest_contrast_array_box, df_sorted_box, output_info, True) # plot    

    fig, ax = plt.subplots(1,2)

    # Display the image with the nb highest scoring ROIs
    ax[0].imshow(box, cmap='bone')
            
    for row in highest_contrast_array_box:
        rect = Rectangle((row[0], row[1]), row[2], row[3], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    ax[0].set_title("Contrast on image box")
    ax[0].set_xlabel("x-pixels")
    ax[0].set_ylabel("y-pixels")

    ax[1].imshow(box*filtered_mask.astype(bool), cmap='bone')
            
    for row in highest_contrast_array_box:
        rect = Rectangle((row[0], row[1]), row[2], row[3], linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    ax[1].set_title("Contrast on filtered mask")
    ax[1].set_xlabel("x-pixels")
    ax[1].set_ylabel("y-pixels")

    fig2, ax2 = plt.subplots(1,3)

    ax2[0].imshow(box*(1-np.array((combined_mask.astype(bool)))), cmap='bone')

    ax2[0].set_title("Selected mask on image")
    ax2[0].set_xlabel("x-pixels")
    ax2[0].set_ylabel("y-pixels")

    distance_transformed = distance_transform_edt(combined_mask.astype(bool))
    ax2[1].imshow(distance_transformed, cmap='bone')

    ax2[1].set_title("Dist. transformed")
    ax2[1].set_xlabel("x-pixels")
    ax2[1].set_ylabel("y-pixels")

    ax2[2].imshow(gaussian_filter(distance_transformed, 0.5), cmap='bone')

    ax2[2].set_title("Gaussian")
    ax2[2].set_xlabel("x-pixels")
    ax2[2].set_ylabel("y-pixels")


    plt.tight_layout()

    # Combine both the filtered mask from the frangi and intensity masks together with the calcs selected based on the contrast
    buffer = combine_contrast_mask(filtered_mask, uniquely_added_calcs)

    buffer_edt = distance_transform_edt(buffer.astype(bool))

    buffer_raytr = correct_raytracing(buffer_edt, thickness)

    buffer_gauss = gaussian_filter(buffer_raytr, sigma=gauss_sigma)    

    m = (np.amax(buffer_raytr)-np.amin(buffer_raytr)) / (np.amax(buffer_gauss)-np.amin(buffer_gauss))
    q = ((np.amin(buffer_raytr)*np.amax(buffer_gauss)-np.amax(buffer_raytr)*np.amin(buffer_gauss))) / (np.amax(buffer_gauss)-np.amin(buffer_gauss))
    buffer_gauss = buffer_gauss * m + q

    
    # ## Save different outputs resulting from the above functions
    # os.makedirs(output_dir_path, exist_ok=True)
    # generic_file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_')
    
    # # save np.array with the unique label of each calc
    # np.save(os.path.join(output_dir_path, generic_file_name + str(len(uniquely_added_calcs)) + '_uniquely_added_calcs'), uniquely_added_calcs, allow_pickle=True)
    
    # # save np.array of the final combined mask / buffer with the unique label values representing the different calcs
    # np.save(os.path.join(output_dir_path, generic_file_name + 'labeled_final_mask_' + str(len(uniquely_added_calcs)) + '_calcs'), buffer, allow_pickle=True)
    # buffer.astype('float32').tofile(output_dir_path + "/" + generic_file_name + 'labeled_final_mask_' + str(len(uniquely_added_calcs)) + '_calcs' + '.raw')
    
    # # save np.array of the final combined mask / buffer after edt distance transformation
    # np.save(os.path.join(output_dir_path, generic_file_name + 'edt_final_mask_' + str(len(uniquely_added_calcs)) + '_calcs'), buffer_edt, allow_pickle=True)
    # buffer_edt.astype('float32').tofile(output_dir_path + "/" + generic_file_name + 'edt_final_mask_' + str(len(uniquely_added_calcs)) + '_calcs' + '.raw')
    
    # Return output
    return buffer_gauss, X, Y # can also return buffer_edt if wanted