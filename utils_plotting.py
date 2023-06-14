# Load and import necessary libraries and functions
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and titles."""
    
    # For text objects, we need to draw the figure first, otherwise the extents are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def plot_masks(image, specific_output_dir, output_info, show_plot=True):
    """Plot the breast image and the different masks,
    with output_info = [outputdir, patient_name, laterality, view, state, str(roi_xsize)+'x'+str(roi_ysize), used_mask_tag]"""
    
    mask_open = np.load(os.path.join(specific_output_dir,output_info[1]+'_'+output_info[2]+'_'+output_info[3]+'_'+output_info[4].upper()+'_mask_open.npy'), allow_pickle=True)
    mask_erode = np.load(os.path.join(specific_output_dir,output_info[1]+'_'+output_info[2]+'_'+output_info[3]+'_'+output_info[4].upper()+'_mask_erode.npy'), allow_pickle=True)
    mask_nofold = np.load(os.path.join(specific_output_dir,output_info[1]+'_'+output_info[2]+'_'+output_info[3]+'_'+output_info[4].upper()+'_mask_nofold.npy'), allow_pickle=True)
    
    if output_info[3] == 'CC':
        fig,ax = plt.subplots(1,4)

        # CC
        ax[0].imshow(image)

        ax[1].imshow(mask_open) ; ax[1].set_title("Open")
        ax[2].imshow(mask_erode) ; ax[2].set_title("Closed")
        ax[3].imshow(mask_nofold) ; ax[3].set_title("Nofold")

    elif output_info[3] == 'MLO':
        
        mask_pect = np.load(os.path.join(specific_output_dir,output_info[1]+'_'+output_info[2]+'_'+output_info[3]+'_'+output_info[4].upper()+'_mask_pect.npy'), allow_pickle=True)
        
        fig,ax = plt.subplots(1,5)
        
        # MLO
        ax[0].imshow(image)

        ax[1].imshow(mask_open) ; ax[1].set_title("Open")
        ax[2].imshow(mask_erode) ; ax[2].set_title("Closed")
        ax[3].imshow(mask_pect) ; ax[3].set_title("Pect")
        ax[4].imshow(mask_nofold) ; ax[4].set_title("Nofold")
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper())
    
    plt.savefig(path + '/' + file_name + '_plot_masks.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()
        

def plot_grid(image, df_grid, output_info, show_plot=True):
    """Plot grid onto image,
    with output_info = [output_dir, patient_name, laterality, view, state, str(roi_xsize)+'x'+str(roi_ysize), used_mask_tag]."""

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Display the image with the grid
    ax.imshow(image)    
    for idx, row in df_grid.iterrows() :
        rect = Rectangle((row['ymin'],row['xmin']), row['height'], row['width'], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(str(output_info[5])+" grid")
    ax.set_xlabel("x-pixels")
    ax.set_ylabel("y-pixels")
        
    
    if show_plot:
        plt.show()
    
    # Save the image with the grid
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_grid.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()
        
        
def plot_valued_roi(image, df_all, df_sorted, output_info, show_plot=True):
    """Display images with the ROI grid colored for each feature's value, leaving out valued ROIs with possible clips
    with output_info = [output_dir, patient_name, laterality, view, state, str(roi_xsize)+'x'+str(roi_ysize), used_mask_tag]."""

    # large range due to clips: to be removed as possible location --> IQR
    Q1 = np.percentile(df_all['range'], 25, interpolation = 'midpoint')       
    Q3 = np.percentile(df_all['range'], 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    # Create figure and axes
    fig, ax = plt.subplots(11,2,sharex=True,sharey=True,figsize=(10,50))

    # Display the image with the ROI grid with its color related to a certain feature value

    for image_count, featureName in enumerate(df_sorted):
        # for our output plot consisting of 2 columns of subplots, the position ax[x_pos, y_pos] of a specific subplot is denoted as:
        x_pos = image_count // 2
        y_pos = image_count % 2

        ax[x_pos, y_pos].imshow(image, cmap='bone')
        
        norm = matplotlib.colors.Normalize(vmin=min(df_sorted[str(featureName)]["range"]), vmax=max(df_sorted[str(featureName)]["range"]))
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.viridis.reversed())
        cmap.set_array([])

        for idx, row in df_sorted[str(featureName)].iterrows() : 
            
            if ((Q1-3*IQR) < row['range'] < (Q3+3*IQR)) : # leaving out valued ROIs/grid points with possible clips
                
                rect = Rectangle((row['ymin'],row['xmin']), row['height'], row['width'], linewidth=1, color=cmap.to_rgba(row['range']), alpha=0.2)
                ax[x_pos, y_pos].add_patch(rect)
            
                #plt.text(row['ymin'], row['xmin'], str(roi_count), color="white")

                #mat_coord_y = -(row['xmin'] - image.shape[0] / 2) * ps_im
                #mat_coord_x = (row['ymin'] - image.shape[1] / 2) * ps_im

            #print('ROI %2d: (%5d,%5d), skewness=%1.5f, Matlab coordinates: (%5d,%5d)' %(roi_count, row['xmin']+roi_xsize/2, row['ymin']+roi_ysize/2, row['skewness'], mat_coord_x, mat_coord_y))
        ax[x_pos,y_pos].set_title(str(featureName))
        cbar = plt.colorbar(cmap, ax = ax[x_pos, y_pos], label=str(featureName))
        
        temp_feature_name = ([i.capitalize() for i in str(featureName).split()])
        
        feature_name = "".join((temp_feature_name))
        
        file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize_' + feature_name + output_info[6])
        #extent = ax[x_pos,y_pos].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        extent = full_extent(ax[x_pos,y_pos]).transformed(fig.dpi_scale_trans.inverted())
        
        plt.savefig(path + '/' + file_name + '_plot_valued_ROI.pdf', bbox_inches=extent)
    
    if show_plot:
        plt.show()
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize_' + 'full' + output_info[6])
    plt.savefig(path + '/' + file_name + '_plot_valued_ROI.pdf', bbox_inches="tight")
    
    if not show_plot:
        plt.close()
        

def plot_highest_scoring_roi(image, dict_roi_coord, df_sorted, output_info, show_plot=True):
    """Display images with the nb highest scoring ROIs for each feature
    with output_info = [output_dir, patient_name, laterality, view, state, str(roi_xsize)+'x'+str(roi_ysize), used_mask_tag]."""
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    fig, ax = plt.subplots(11,2,figsize=(10,50))

    # Display the image with the nb highest scoring ROIs
    
    for image_count, featureName in enumerate(df_sorted):
        # for our output consisting of 2 columns of subplots, the position ax[x_pos, y_pos] of a specific subplot is denoted as:
        x_pos = image_count // 2
        y_pos = image_count % 2

        ax[x_pos, y_pos].imshow(image, cmap='bone')
        
        for row in dict_roi_coord[str(featureName)]:
            #(row['ymin'], row['xmin']), row['height'], row['width']
            rect = Rectangle((row[0], row[1]), row[2], row[3], linewidth=1, edgecolor='r', facecolor='none')
            ax[x_pos, y_pos].add_patch(rect)
                
        ax[x_pos,y_pos].set_title(str(featureName))
        ax[x_pos,y_pos].set_xlabel("x-pixels")
        ax[x_pos,y_pos].set_ylabel("y-pixels")
        
        temp_feature_name = ([i.capitalize() for i in str(featureName).split()])
        
        feature_name = "".join((temp_feature_name))
        
        file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize_' + feature_name+ output_info[6])
        
        extent = full_extent(ax[x_pos,y_pos]).transformed(fig.dpi_scale_trans.inverted())
        
        plt.savefig(path + '/' + file_name + '_plot_highest_ROI.pdf', bbox_inches=extent)
    
    if show_plot:
        plt.show()
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + '_full' + output_info[6])
    plt.savefig(path + '/' + file_name + '_plot_highest_ROI.pdf', bbox_inches="tight")
    
    if not show_plot:
        plt.close()


def plot_optimal_region(img, output_info, optimal_roi_coord, tag, roi_xsize=200, roi_ysize=200, show_plot=True):
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Display the image with the ROI grid
    ax.imshow(img)    
    for coord_tuple in optimal_roi_coord :
        rect = Rectangle((coord_tuple[0],coord_tuple[1]), roi_ysize, roi_xsize, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_xlabel("x-pixels")
    ax.set_ylabel("y-pixels")
              
    if show_plot:
        plt.show()
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_optimal_ROI' + tag + '.pdf', bbox_inches="tight")
    
    if not show_plot:
        plt.close()
        
        
def plot_first_optimal_region(img, output_info, optimal_roi_coord, tag, roi_xsize=200, roi_ysize=200, show_plot=True):
    
    point_y = optimal_roi_coord[0][0]
    point_x = optimal_roi_coord[0][1]
    
    fig, ax = plt.subplots(figsize=(10,5))

    # Create figure and axes
    ax.imshow(img)    
    rect = Rectangle((point_y,point_x), roi_ysize, roi_xsize, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_xlabel("x-pixels")
    ax.set_ylabel("y-pixels")
           
    if show_plot:
        plt.show()
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_first_optimal_ROI' + tag + '.pdf', bbox_inches="tight")
    
    if not show_plot:
        plt.close()

    box = []
    ind = 0

    while ind < roi_xsize:
        box.append((img[point_x+ind][point_y:point_y+roi_ysize]))
        ind += 1

    box = (np.array(box))

    fig, ax = plt.subplots(figsize=(10,5))

    # Display the image with the ROI grid
    ax.imshow(box)
    
    if show_plot:
        plt.show()

    plt.savefig(path + '/' + file_name + '_plot_first_optimal_ROI_box' + tag + '.pdf', bbox_inches ="tight")
    np.save(os.path.join(path, file_name + '_first_optimal_ROI_box' + tag), box, allow_pickle=True)

    if not show_plot:
        plt.close()
        
        
def plot_third_optimal_region(img, output_info, optimal_roi_coord, tag, roi_xsize=200, roi_ysize=200, show_plot=True):
    
    point_y = optimal_roi_coord[2][0]
    point_x = optimal_roi_coord[2][1]
    
    fig, ax = plt.subplots(figsize=(10,5))

    # Create figure and axes
    ax.imshow(img)    
    rect = Rectangle((point_y,point_x), roi_ysize, roi_xsize, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_xlabel("x-pixels")
    ax.set_ylabel("y-pixels")
           
    if show_plot:
        plt.show()
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_third_optimal_ROI' + tag + '.pdf', bbox_inches="tight")
    
    if not show_plot:
        plt.close()

    box = []
    ind = 0

    while ind < roi_xsize:
        box.append((img[point_x+ind][point_y:point_y+roi_ysize]))
        ind += 1

    box = (np.array(box))

    fig, ax = plt.subplots(figsize=(10,5))

    # Display the image with the ROI grid
    ax.imshow(box)
    
    if show_plot:
        plt.show()

    plt.savefig(path + '/' + file_name + '_plot_third_optimal_ROI_box' + tag + '.pdf', bbox_inches ="tight")
    np.save(os.path.join(path, file_name + '_third_optimal_ROI_box' + tag), box, allow_pickle=True)

    if not show_plot:
        plt.close()
        
        
def plot_frangi_intensity_combined_mask(box, frangi_mask, intensity_mask, combined_mask, output_info, tag, show_plot=True):
    fig, ax = plt.subplots(1,3)

    ax[0].imshow(frangi_mask.astype(int)*box); ax[0].set_title('Binary opening Frangi')
    ax[1].imshow(intensity_mask*box); ax[1].set_title('Intensity')

    multpl_mask = (intensity_mask)*frangi_mask.astype(int)

    ax[2].imshow(multpl_mask*box); ax[2].set_title('Combined')
    
    if show_plot:
        plt.show()
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_combined_masks' + tag + '.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()
    

def plot_filtered_mask(filtered_mask, box, output_info, tag, show_plot=True):
    fig, ax = plt.subplots()

    ax.imshow(filtered_mask.astype(bool)*box)

    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_filtered_mask' + tag + '.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()
        
        
def plot_final_mask(final_mask, box, output_info, tag, show_plot=True):
    fig, ax = plt.subplots()

    ax.imshow(final_mask.astype(bool)*box)

    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize' + output_info[6])
    
    plt.savefig(path + '/' + file_name + '_plot_combined_filtered_contrast_mask' + tag + '.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()
    
    
def plot_highest_scoring_contrast_box(image, mask, array_roi_coord, df_sorted, output_info, show_plot=True):
    """Display images with the nb highest scoring ROIs for each feature
    with output_info = [output_dir, patient_name, laterality, view, state, str(roi_xsize)+'x'+str(roi_ysize), used_mask_tag]."""
    
    path = os.path.join(output_info[0], output_info[1], output_info[2]+'_'+output_info[3])
    os.makedirs(path, exist_ok=True)
    
    fig, ax = plt.subplots(1,2)

    # Display the image with the nb highest scoring ROIs
    ax[0].imshow(image, cmap='bone')

    for row in array_roi_coord:
        rect = Rectangle((row[0], row[1]), row[2], row[3], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    ax[0].set_title("Contrast on image box")
    ax[0].set_xlabel("x-pixels")
    ax[0].set_ylabel("y-pixels")

    ax[1].imshow(image*mask.astype(bool), cmap='bone')

    for row in array_roi_coord:
        rect = Rectangle((row[0], row[1]), row[2], row[3], linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    ax[1].set_title("Contrast on filtered mask")
    ax[1].set_xlabel("x-pixels")
    ax[1].set_ylabel("y-pixels")
    
    
    if show_plot:
        plt.show()
    
    file_name = str(output_info[1] + '_' + output_info[2] + '_' + output_info[3] + '_' + output_info[4].upper() + '_' + output_info[5] + 'gridSize_' + "contrast" + output_info[6])

    plt.savefig(path + '/' + file_name + '_plot_highest_ROI.pdf', bbox_inches ="tight")
    
    if not show_plot:
        plt.close()