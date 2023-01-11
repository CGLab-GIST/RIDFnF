#  Copyright (c) 2023 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import time
import re
from functools import partial
from sys import exit
import multiprocessing
from multiprocessing import Pool
import parmap
from tensorflow.python.ops import candidate_sampling_ops
import exr
from utilities import printFunc, samplePatchesStrided
import numpy as np
from tensorflow.keras import datasets, layers, models, initializers
import tensorflow as tf
import os
import random
from functools import partial
import glob
import utils_image as util
from config import config


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def load_image_name_fnf_awgn(directory, endMatch, train, config):
    filenames = []
    if train:
        # find ambient images
        if config["numCandidates"] == 2: # (y, z) * 3
            print('Image Load Config : Single buffer')
            pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*' + config["datasetFixword"]) + '*')]
            for path in pathes:
                if path.endswith(endMatch):
                    filenames.append(path)

        elif config["numCandidates"] > 2:    
            print('Image Load Config : Multi buffer (Multi-correlation structure)')
            pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*' + config["datasetFixword"]) + '*')]
            for path in pathes:
                if path.endswith(endMatch):
                    filenames.append(path)
    else :   
        # find ambient images
        if config["numCandidates"] == 2: # (y, z) * 3
            print('Image Load Config : Single buffer')
            pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*' + config["datasetFixword"]) + '*')]
            for path in pathes:
                if path.endswith(endMatch):
                    filenames.append(path)

        elif config["numCandidates"] > 2:    
            print('Image Load Config : Multi buffer (Multi-correlation structure)')
            pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*' + config["datasetFixword"]) + '*')]
            for path in pathes:
                if path.endswith(endMatch):
                    filenames.append(path)            

    filenames = natural_sort(filenames) # Natural sort

    # Shuffle the training images
    if train:
        random.seed(time.time())
        random.shuffle(filenames)
    
    if len(filenames) == 0:
        printFunc(directory, 'is empty!')
        exit()

    return filenames


def load_reference_fnf_awgn_v3(color_filenames, train, config):
    if train:
        folder_name = config["train_target"][0]
        ref_dataset_dir = os.path.join(config['trainDatasetDirectory'], folder_name)
    elif not train:
        folder_name = config["test_target"][0]
        ref_dataset_dir = os.path.join(config['testDatasetDirectory'], folder_name)

    ref_candidates = [fn for fn in glob.glob(os.path.join(ref_dataset_dir +  '/*' + "ambient") + '*')]
    
    ref_filenames = []
    for color_filename in color_filenames:
        cname = os.path.basename(color_filename).split('.')[0].split('_')

        color_scenename = cname[0] + '_' + cname[1]
        found = False
        for ref_filename in ref_candidates:
            rname = os.path.basename(ref_filename).split('.')[0].split('_')
            ref_scenename = rname[0] + '_' + rname[1]
            if color_scenename == ref_scenename:
                found = True
                if ref_filename.endswith('ambient.png'):
                    ref_filenames.append(ref_filename)
                    break

        if not found:
            printFunc('Cannot find reference for', color_filename)
            exit()

    save_images_to_npy_fnf_awgn_v2(config, ref_filenames)
    if config["numCandidates"] == 2: # (y, z) * 3`
        ref_filenames = list(map(partial(util.change_dir, end ='.png'), ref_filenames))
    elif config["numCandidates"] > 2: # (y, z1, z2, z3, z4) * 3`
        ref_filenames = list(map(partial(util.change_dir, end ='.png'), ref_filenames))

    if len(ref_filenames) == 0:
        printFunc(ref_dataset_dir, 'is empty!')
        exit()

    return ref_filenames

def check_paris_fnf(color_filenames, ref_filenames):
    for (c, r) in zip(color_filenames, ref_filenames):

        cname = os.path.basename(c).split('_')
        rname = os.path.basename(r).split('_')

        c = cname[0] + '_' +  cname[1]
        r = rname[0] + '_' +  rname[1]
        if c != r:
            print('Wrong pair!', c, r)
            exit(-1)


def load_tf(path):
    def load_numpy(color_path, ref_path):
        sceneName = color_path.numpy().decode('utf-8').split('/')[-1].split('.')[0]
        return sceneName, np.load(color_path.numpy()), np.load(ref_path.numpy())
    return tf.py_function(load_numpy, inp=[path[0], path[1]], Tout=[tf.string, tf.float32, tf.float32])

def patch_generator(config, paths):
    for idx, (img_path, ref_path) in enumerate(paths.tolist()):
        img, ref = np.load(img_path), np.load(ref_path)
        patch_indices = samplePatchesStrided(img.shape[:2], config['patchSize'], config['patchStride'])
        for pos in patch_indices:
            yield img[pos[1]:pos[1] + config['patchSize'], pos[0]:pos[0] + config['patchSize']], ref[pos[1]:pos[1] + config['patchSize'], pos[0]:pos[0] + config['patchSize']]


# data load for train and test (make npy file for training and test)
def load_fnf_awgn(config, train, preprocessor=None):
    color_filenames = []
    color_filenames_tmp = []
    if train:
        for folder_name in config['train_input']:
            directory = os.path.join(config['trainDatasetDirectory'], folder_name)
            # Collect filenames
            if config["numCandidates"] == 2: # (z, y) * 3
                print('Config : Single buffer')
                color_filenames.extend(load_image_name_fnf_awgn(directory, ".png", train, config))
            elif config["numCandidates"] > 2: # (z1, z2, z3, z4, y) * 3
                print('Config : Multi buffer (Multi-correlation structure)')
                color_filenames.extend(load_image_name_fnf_awgn(directory, ".png", train, config))            
            for idx in range(len(color_filenames)):
                for noise_level in config['noise_levels']:

                    file_basename = os.path.basename(color_filenames[idx]).split('.')[0]
                    file_dir = os.path.split(color_filenames[idx]) 
                    new_file_name = file_basename + '_noise_' + noise_level + '.exr'
                    noise_filenames = os.path.join(file_dir[0],new_file_name)
                    color_filenames_tmp.append(noise_filenames)

            color_filenames = color_filenames_tmp
    else:
        for folder_name in config['test_input']:
            directory = os.path.join(config['testDatasetDirectory'], folder_name)
            # Collect filenames
            if config["numCandidates"] == 2: # (z, y) * 3
                print('Config : Single buffer')
                color_filenames.extend(load_image_name_fnf_awgn(directory, ".png", train, config))
            elif config["numCandidates"] > 2:# (z1, z2, z3, z4, y) * 3
                print('Config : Multi buffer (Multi-correlation structure)')
                color_filenames.extend(load_image_name_fnf_awgn(directory, ".png", train, config))
                
            for idx in range(len(color_filenames)):
                for noise_level in config['noise_levels']:

                    file_basename = os.path.basename(color_filenames[idx]).split('.')[0]
                    file_dir = os.path.split(color_filenames[idx]) 
                    new_file_name = file_basename + '_noise_' + noise_level + '.exr'
                    noise_filenames = os.path.join(file_dir[0],new_file_name)
                    color_filenames_tmp.append(noise_filenames)

            color_filenames = color_filenames_tmp

    save_images_to_npy_fnf_awgn_v2(config, color_filenames, config["channelNames"])
    

    if config["numCandidates"] == 2: # (y, z) * 3
        color_filenames = list(map(partial(util.change_dir, end ='.exr'), color_filenames))
    elif config["numCandidates"] > 2: # (y, z1, z2, z3, z4) * 3
        color_filenames = list(map(partial(util.change_dir, end ='.exr'), color_filenames))
    

    # Load reference files
    ref_filenames = load_reference_fnf_awgn_v3(color_filenames, train, config)
    # Check parity of collected names
    check_paris_fnf(color_filenames, ref_filenames)
    # Combine two lists
    filenames = np.stack([color_filenames, ref_filenames], axis=1)
    numData = len(filenames)
    

    if train:
        printFunc('[%s] num train data: %d' % (config['trainDatasetDirectory'], numData))
    else:
        printFunc('[%s] num test data: %d' % (config['testDatasetDirectory'], numData))

    if train:        
        dataset = tf.data.Dataset.from_generator(partial(patch_generator, config, filenames), 
            output_signature=(
                tf.TensorSpec(shape=(None, None, config['numChannels']), dtype=tf.float32), # img
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),                     # ref
            ))
        dataset = dataset.shuffle(buffer_size=numData, reshuffle_each_iteration=True)
        dataset = dataset.batch(config['global_batch_size'])
    else:
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(load_tf) 
        dataset = dataset.batch(1)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

def convert_images_to_npy_for_single_buffer_fnf_awgn_v2(filename, channels):
    file_basename = os.path.basename(filename).split('.')[0]
    file_name_split = file_basename.split('_')
    file_dir = os.path.split(filename) 
    file_type = os.path.splitext(filename) 

    if file_type[-1] == '.exr':
        # for correlated image
        new_dir = file_dir[0] + '_npy'
        new_filename = os.path.join(new_dir,os.path.basename(filename))
        new_filename = new_filename.replace('exr', 'npy')

        if os.path.isfile(new_filename):
            return  
        for i in range(len(channels)):
            c = channels[i]
            print('{0} corr method :{1}'.format(i,c))
            filename_corr = os.path.join(file_dir[0] , file_name_split[0] + '_' + file_name_split[1] + '_' + c + '.png')
            data = util.imread_uint(filename_corr , 3)

            accm = data / 255.0

        filename_noise = filename
       
        if os.path.isfile(filename_noise):         
            data_noise = exr.read(filename_noise)
            data_noise = np.array(data_noise)
            data_noise = data_noise / 255.0
            data_noise = data_noise[:, :, 0:3]
            accm = np.dstack([accm, data_noise]) # [z y]
        else:
            print('There is no noise file : {}'.format(filename_noise))
            filename_ambient = os.path.join(file_dir[0] , file_name_split[0] + '_' + file_name_split[1] + '_' +file_name_split[2]  + '.png')        
            ambient = util.imread_uint(filename_ambient, 3)
            img_ambient = ambient / 255.0

            noise_level = float(file_name_split[-1]) / 255.0
            noise = np.random.normal(size = img_ambient.shape) * noise_level

            data_noise = img_ambient + noise
            accm = np.dstack([accm, data_noise]) # [z y]

    # for reference images
    elif file_type[-1] == '.png':
        # for correlated image
        new_dir = file_dir[0] + '_npy'
        new_filename = os.path.join(new_dir,os.path.basename(filename))
        new_filename = new_filename.replace('png', 'npy')

        if os.path.isfile(new_filename):
            return  

        data = util.imread_uint(filename , 3)
        accm = data / 255.0

        # for noise image
        if file_name_split[0].isdigit():
            filename_noise = os.path.join(file_dir[0] , file_name_split[0] + '_noise_' + file_name_split[-1] + '.exr')
        elif file_name_split[0].isalpha():
            filename_noise = os.path.join(file_dir[0] , file_name_split[0] +'_' + file_name_split[1] + '_noise_' + file_name_split[-1] + '.exr')

        if os.path.isfile(filename_noise):         
            data_noise = exr.read(filename_noise)
            data_noise = np.array(data_noise)
            data_noise = data_noise / 255.0
            data_noise = data_noise[:, :, 0:3]
            accm = np.dstack([accm, data_noise]) # [z y]
        else:
            print('There is no no png : {}'.format(filename_noise))

    else:
        print('This file is not file : {}'.format(file_dir[1]))
        return
     
    printFunc('convert', filename, 'to', new_filename)

    if np.isnan(accm).any():
        printFunc('There is NaN in', new_filename, 'Set it to zero for training.')
        accm = np.nan_to_num(accm, copy=False)
    if np.isposinf(accm).any() or np.isneginf(accm).any():
        printFunc("There is INF in", new_filename, 'Set it to zero for training.')
        accm[accm == np.inf] = 0
        accm[accm == -np.inf] = 0

    np.save(new_filename, accm)

def convert_images_to_npy_for_multi_buffer_fnf_awgn(filename, channels):
    file_basename = os.path.basename(filename).split('.')[0]
    file_name_split = file_basename.split('_')
    file_dir = os.path.split(filename) 
    file_type = os.path.splitext(filename) 

    if file_type[-1] == '.exr':
        # for correlated image
        new_dir = file_dir[0] + '_npy'
        new_filename = os.path.join(new_dir,os.path.basename(filename))
        new_filename = new_filename.replace('exr', 'npy')

        if os.path.isfile(new_filename):
            return  
        for i in range(len(channels)):
            c = channels[i]
            str_c = str(c) 
            print('{0} corr method :{1}'.format(i,c))
            filename_corr = os.path.join(file_dir[0] , file_name_split[0] + '_' + file_name_split[1] + '_' + c + '.png')
            data = util.imread_uint(filename_corr , 3)
            data = data / 255.0
            if i == 0:
                accm = data
            else:
                accm = np.dstack([accm, data])

        filename_noise = filename

        if os.path.isfile(filename_noise):
            data_noise = exr.read(filename_noise)
            data_noise = np.array(data_noise)
            data_noise = data_noise[:, :, 0:3]
            data_noise = data_noise / 255.0
            accm = np.dstack([accm, data_noise]) # [z1, z2, z3, z4, y]
        else:
            print('There is no noise file : {}'.format(filename_noise))

    elif file_type[-1] == '.png':
        # for correlated image
        new_dir = file_dir[0] + '_npy'
        new_filename = os.path.join(new_dir,os.path.basename(filename))
        new_filename = new_filename.replace('png', 'npy')
        

        if os.path.isfile(new_filename):
            return  

        data = util.imread_uint(filename , 3)
        accm = data / 255.0

        # for noise image
        if file_name_split[0].isdigit():
            filename_noise = os.path.join(file_dir[0] , file_name_split[0] + '_noise_' + file_name_split[-1] + '.exr')
        elif file_name_split[0].isalpha():
            filename_noise = os.path.join(file_dir[0] , file_name_split[0] +'_' + file_name_split[1] + '_noise_' + file_name_split[-1] + '.exr')

        if os.path.isfile(filename_noise):
            data_noise = exr.read(filename_noise)
            data_noise = np.array(data_noise)
            data_noise = data_noise / 255.0
            data_noise = data_noise[:, :, 0:3]
            accm = np.dstack([accm, data_noise])
        else:
            print('There is no noise file : {}'.format(filename_noise))

    else:
        print('This file is not exr : {}'.format(file_dir[1]))
        return
     
    printFunc('convert', filename, 'to', new_filename)

    if np.isnan(accm).any():
        printFunc('There is NaN in', new_filename, 'Set it to zero for training.')
        accm = np.nan_to_num(accm, copy=False)
    if np.isposinf(accm).any() or np.isneginf(accm).any():
        printFunc("There is INF in", new_filename, 'Set it to zero for training.')
        accm[accm == np.inf] = 0
        accm[accm == -np.inf] = 0

    np.save(new_filename, accm)


def save_images_to_npy_fnf_awgn_v2(config, filenames, channels=['default']):
    filenames = list(dict.fromkeys(filenames))
    channels = list(dict.fromkeys(channels))
    print('cpu count:', multiprocessing.cpu_count())

    # make new folder for npy files
    file_dir = os.path.split(filenames[0]) 
    new_dir = file_dir[0] + '_npy'
    util.mkdir(new_dir)

    if config["numCandidates"] == 2: # (y, z) * 3
        print('Convert to npy Config : Single buffer')
        parmap.map(partial(convert_images_to_npy_for_single_buffer_fnf_awgn_v2, channels=channels), filenames, pm_pbar=True, pm_processes=multiprocessing.cpu_count())


    elif config["numCandidates"] > 2:    
        print('Convert to npy Config : Multi buffer (Multi-correlation structure)')
        parmap.map(partial(convert_images_to_npy_for_multi_buffer_fnf_awgn, channels=channels), filenames, pm_pbar=True, pm_processes=multiprocessing.cpu_count())