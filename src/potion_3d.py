import numpy as np
import os
import scipy.ndimage
#import scipy.io
import time
import scipy.sparse
from scipy.interpolate import interp1d
import h5py

sigma = 1                   #standard deviation of gaussian for the blurring
resolution = 64             #resolution of the heatmap
heatmap_dir = 'heatmaps'    #heatmap directory
coordinate_dir = 'coords'   #coordinate ditectory
colour_channels = 4         #colour channels used for colorization
num_joints = 16

array_ext = ['npy']         #extension file for a numpy array file

def is_pickle(file_name):                               #function to check wheither file is a numpy array file
    ext = file_name[file_name.rfind('.') + 1:].lower()  #extract the file extension
    return ext in array_ext                             #return true if extension is in the list of allowed extensions

def get_filename(file_name):                    #function to extract filename without extension
    name = file_name[0: + file_name.rfind('.')] #gets filename by stripping off anything after a period
    return name                                 #returns the name

def quantize(coordinates, res, min, max):       #quantizes coordinates to a 7 bit number
    m = interp1d([min,max],[0,(res-1)])         #interpolates 1d function
    coordinates_q = m(coordinates)              #quantizes coordinates based on the previously interpolated function
    coordinates_q = coordinates_q.astype(int)   #converts the coordinates to integers
    return coordinates_q                        #returns the quantized coordinates

def coordinates_2_heatmap(coordinates,res):                                 #inserts coordinates per joint in a 3D map
    joints = coordinates.shape[0]                                           #the number of joints
    heatmap = np.zeros((joints,res,res,res))                                #instantiate the heatmap
    for i in range(0,joints):                                               #go through each joint
        heatmap[i,coordinates[i,0],coordinates[i,1],coordinates[i,2]] = 1   #use the quantized coordinates as indices to enter into the heatmap
    return heatmap                                                          #return the heatmap

def dilate_heatmap(heatmap, sig):                                                   #gaussian blurring on the heatmap of each joint
    heatmap_d = np.empty(heatmap.shape)                                             #instantiate an empty heatmap
    for i in range(len(heatmap)):                                                   #go through the input heatmap
        heatmap_d[i,:,:,:] = scipy.ndimage.gaussian_filter(heatmap[i,:,:,:],sig)    #gaussian blur the heatmap
    return heatmap_d                                                                #return blurred heatmap

def heatmap_colorization_aggregation(file_name,coords):                                     #time dependent heatmap colorization
    heatmap_sum = np.empty((resolution,resolution,resolution,num_joints*colour_channels))   #instantiate empty heatmap for the temporal aggregation
    heatmap_col = np.empty((resolution,resolution,resolution,num_joints*colour_channels))   #instantiate empty heatmap for the colorization of individual frames
    T = coords.shape[0] - 1 #added -1                                                       #count the number of frames in the video
    partitions = colour_channels - 1.                                                       #partition frames into partitions for the colorization
    for i in range(T+1):                                                                    #go through all frames
        heatmap = coords[i]                                                                 #place the heatmap into this temproary variable
        for joint in range(num_joints):                                                     #go through the joints
            if i < T/partitions:                                                            #if the frame is from the first partition
                heatmap_col[:,:,:,colour_channels*joint+0] = (1. - i*partitions/T)  * heatmap[joint,:,:,:]                  #the first colour channel is a linear decreasing curve
                heatmap_col[:,:,:,colour_channels*joint+1] = (i*partitions/T) * heatmap[joint,:,:,:]                        #the second channel increases
                heatmap_col[:,:,:,colour_channels*joint+2] = np.zeros(heatmap_col.shape[0:3])                               #the third channel is set to zero
                heatmap_col[:,:,:,colour_channels*joint+3] = np.zeros(heatmap_col.shape[0:3])                               #the fourth channel is also set to zero
            elif i < 2*T/partitions:                                                        #if the frames are from the second partiton
                heatmap_col[:,:,:,colour_channels*joint+0] = np.zeros(heatmap_col.shape[0:3])                               #first channel is zero
                heatmap_col[:,:,:,colour_channels*joint+1] = (1. - (i-T/partitions)*partitions/T) * heatmap[joint,:,:,:]    #second decreases
                heatmap_col[:,:,:,colour_channels*joint+2] = ((i-T/partitions)*partitions/T) * heatmap[joint,:,:,:]         #third increases
                heatmap_col[:,:,:,colour_channels*joint+3] = np.zeros(heatmap_col.shape[0:3])                               #fourth is zero
            else:                                                                           #if the frames are from the last partition
                heatmap_col[:,:,:,colour_channels*joint+0] = np.zeros(heatmap_col.shape[0:3])                               #first channel is zero
                heatmap_col[:,:,:,colour_channels*joint+1] = np.zeros(heatmap_col.shape[0:3])                               #second channel is zero
                heatmap_col[:,:,:,colour_channels*joint+2] = (1. - (i-2*T/partitions)*partitions/T) * heatmap[joint,:,:,:]  #third decreases
                heatmap_col[:,:,:,colour_channels*joint+3] = ((i-2*T/partitions)*partitions/T) * heatmap[joint,:,:,:]       #fourth increases
            heatmap_sum[:,:,:,colour_channels*joint+0] += heatmap_col[:,:,:,colour_channels*joint+0]    #temporally aggregate the colorized channels
            heatmap_sum[:,:,:,colour_channels*joint+1] += heatmap_col[:,:,:,colour_channels*joint+1]    #temporally aggregate the colorized channels
            heatmap_sum[:,:,:,colour_channels*joint+2] += heatmap_col[:,:,:,colour_channels*joint+2]    #temporally aggregate the colorized channels
            heatmap_sum[:,:,:,colour_channels*joint+3] += heatmap_col[:,:,:,colour_channels*joint+3]    #temporally aggregate the colorized channels
    print("dividing")                                                               #begin normalization by dividing by the per channel max
    for j in range(num_joints):                                                     #for each joint
        if heatmap_sum[:,:,:,j*3:j*3+3].max() != 0:                                 #if the max is not zero
            heatmap_sum[:,:,:,j*3:j*3+3] /= heatmap_sum[:,:,:,j*3:j*3+3].max()      #divide the temporally aggregated and colorized heatmap by its max
    return heatmap_sum                                                              #return the normalized heatmap

def main():
    ls = os.listdir(coordinate_dir)                                             #list all files in the coordinate directory
    print("Creating 3D heatmaps")
    #hm_col_full = []                                                            #instantiate empty list of heatmaps
    for file_name in sorted(ls):                                                #iterate through all npy files in the directory
        print(file_name)
        if is_pickle(file_name):                                                #read only .npy files
            image_name = os.path.join(coordinate_dir, file_name)                #join image name
            coordinates = np.load(image_name)                                   #load coordiantes from file
            ma = coordinates.max() +.1                                          #get maximum value of the coordinates plus a threshold for the quantization
            mi = coordinates.min() -.1                                          #get minimum value of the coordinates plus a threshold for the quantization
            coordinates=coordinates.reshape((coordinates.shape[0], 16,3))       #reshape the coordinates to make it easier to read
            hm_d_full =[]# np.empty((0,resolution,resolution,resolution))       #instantiate an empty array to store the colorized heatmaps
            for i in range(coordinates.shape[0]):
                co_q = quantize(coordinates[i], resolution, mi, ma)             #requantize coordiantes, so they fit in the heatmap
                hm = coordinates_2_heatmap(co_q, resolution)                    #create heatmaps from coordinates
                hm_d = dilate_heatmap(hm, sigma)                                #blur image with a gaussian kernel of sigma
                hm_d_full.append(hm_d)                                          #append the dilated heatmap to the array of heatmaps
            hm_d_full=np.asarray(hm_d_full)                                     #convert the list of dilated heatmaps into a numpy array
            heatmap_col = heatmap_colorization_aggregation(file_name,hm_d_full) #colorize and temporally aggregate the heatmaps from the list
            np.savez(os.path.join(heatmap_dir,get_filename(file_name),heatmap_col)
            #hm_col_full.append(heatmap_col)                                     #append the colorized heatmap to an array to be later saved to a file
    #hm_col_full=np.asarray(hm_col_full)                                         #convert the array of colorized heatmaps into a numpy array, in order to be saved to a file
    #num_samples_all = 2326                                                      #define the number of samples in order to do the train-test split
    #val_split = int(0.25 * num_samples_all)                                     #use a quarter of the data for validation
    #val_ind = np.random.choice(num_samples_all, val_split)                      #take some random samples for the validation split
    #all_ind = np.arange(0,num_samples_all)                                      #create list of all indices
    #train_ind = np.setdiff1d(all_ind, val_ind)                                  #take all indices not in the validation split and put it in the training set
    #train_data= hm_col_full[train_ind]                                          #take the train data according to the indices
    #val_data= hm_col_full[val_ind]                                              #take the validation data according to the indices

'''
    with h5py.File("/cluster/scratch/majustin/train_data.h5") as f:                       #write the set of all heatmaps into an h5 file
        dst = f.create_dataset("train_data", shape=(train_data.shape[0],64,64,64,64),   #create a dataset datatype from the numpy array
                        dtype=float)
        for frame in range(train_data.shape[0]):                                        #for each video
            dst[frame] = train_data[frame]                                              #add the colorized heatmap into the dataset
        dst = f.create_dataset("train_data", shape=(train_data.shape[0],64,64,64,64),   #create a dataset datatype from the numpy array
                        dtype=float)
        for frame in range(train_data.shape[0]):                                        #for each video
            dst[frame] = train_data[frame]                                              #add the colorized heatmap into the dataset
'''

if __name__ == '__main__':
    main()
