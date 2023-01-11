# %% Initializaion
import os
import time
import numpy as np

# Disable Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0, 1, 2, 3s
import tensorflow as tf

import glob
from layers import MainNet
from utilities import printFunc
from config import config, gpus
from pytz import timezone
import utils_image as util

def parseInput_single(config, input):
    # input: (B, H, W, numChannels)    
    # Candidate Colors (3 * m)
    candidates = input[:, :, :, config["CANDIDATE_POS"]:config["CANDIDATE_POS"] +
                       config['numCandidates'] * 3]  # 3 for RGB

    return candidates

def buildModel():
    # Update the global variable
    global model

    ####################
    ###### Input #######
    ####################
    # Input shape can be (patchSize, patchSize) or (height, width) at inference
    input = tf.keras.Input(shape=(None, None, config['numChannels']), name="Input")

    ####################
    ###### Model #######
    ####################
    output = MainNet(config, input)
    model = tf.keras.Model(inputs=[input], outputs=[output], name="my_model")

    ####################
    ###### Model #######
    ####################
    sum = 0
    for v in model.trainable_variables:
        print(v.name, v.shape)
        sum += tf.math.reduce_prod(v.shape)
    print('\tNumber of weights:', f"{sum.numpy():,}")
    model.summary()
    return model


if __name__ == "__main__":

    model = buildModel()

    initial_learning_rate = 0.0005 
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate) # 1e-4

    # set task folder and chechpoint folder
    task = config['task']
    task_dir = os.path.join(config['DataDirectory'],task)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(task_dir, '__checkpoints__')         
    checkpoint_path = checkpoint_dir + "/cp-%s" % config['title']
    # Load weights
    checkpoint.restore(checkpoint_path + "-" + str(config["loadEpoch"]))
    print("Restore network parameters for Test : {}".format(checkpoint_path + "-" + config["loadEpoch"]))

    # directory for real noise test
    dir_flash = '/data/__real_test__/real_images/flash'
    dir_noise = '/data/__real_test__/real_images/ambient'
    dir_output = '/data/__real_test__/real_images/output'
    util.mkdir(dir_output)

    filenames = []
    print('Flash Image Load')
    directory = dir_noise
    pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*' + 'ambient') + '*')]
    for path in pathes:
        if path.endswith('.png'):
            filenames.append(path)
            
    for path in filenames:
        file_dir = os.path.split(path) 
        sname =  os.path.basename(path).split('.')[0].split('_')
        sceneName =  sname[0] + '_' + sname[1]
        # new_filename = os.path.join(dir_save,sceneName)
        print('{}'.format(path))

        filename_noise = path
        data_noise = util.imread_uint(filename_noise, 3)
        data_noise = data_noise / 255.0

        height, width, c = data_noise.shape


        imgName = sceneName + '_ambient.png'
        img_filename = os.path.join(dir_noise,imgName)
        ref = util.imread_uint(img_filename, 3)
        ref = ref / 255.0

        flashName = sceneName + '_flash.png'
        flash_filename = os.path.join(dir_flash,flashName)
        
        flash = util.imread_uint(flash_filename, 3)
        flash = flash / 255.0

        input = np.dstack([flash, data_noise])

        print(data_noise.shape)
        print(flash.shape)
        input = tf.convert_to_tensor(input, dtype=np.float64)
        input = tf.expand_dims(input, axis = 0)

        (img) = model(input, training=False)

        np_out = tf.squeeze(img)
        np_out = np_out.numpy()


        np_ref = ref
        np_out = np_out *255
        np_ref = np_ref *255


        np_out = np.clip(np_out, 0, 255)
        np_ref = np.clip(np_ref, 0, 255)

        filename = "%s_%s.png" % (
                    sceneName, 'denoised')

        util.imwrite(os.path.join(dir_output,filename.replace('.png', '_out.png')), np_out)

    print('DONE')