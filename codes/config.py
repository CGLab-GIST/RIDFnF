import tensorflow as tf
from loss import L2Loss

# tf.config.experimental_run_functions_eagerly(False)
# tf.config.run_functions_eagerly(False)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs") 
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


######################
## Global Variables ##
######################
config = {}
config["timezone"] = None
config["title"] = "new"

# Mode selection [TRAINING, TEST]
# config["mode"] = "TRAINING"
config["mode"] = "TEST"
config["loadEpoch"] = "50" # for testing with pre-traind parameter (training epoches : 50)

# System memory related params
# Number of shuffled patches. Should be bigger than # of patches of an image
config["patchBatch"] = 32 # Number of patches per batch (and per replica) 
print('Batch size for single-GPU:', config["patchBatch"])
config['global_batch_size'] = config["patchBatch"] * len(gpus)

# Training
config['epochs'] = 50 # the max epoch for training

# Network
config["loss"] = L2Loss
config["convSize"] = 5 # conv filter size (5 x 5) 
config["convActivation"] = 'relu'
config["numFilters"] = 80 # the number of feature map (CNN)
config["kernelSize"] = 15 # size of combination kernel (19 x 19)
config["kernelArea"] = config["kernelSize"] * config["kernelSize"]
config["numCandidates"] = 2 # fixed (flash/no-flash pair)
config["numOutput"] = (config["numCandidates"] - 1) * config["kernelArea"] # the number of weights
config["patchSize"] = 64
config["patchStride"] = int(config["patchSize"] * 1.5)  # Data augmentation
config["kernelSizeConvolK"] = 7 
config["numOutputConvolK"] = (config["kernelSizeConvolK"] * config["kernelSizeConvolK"])

# Retraining 
config["retrain"] = False
config["restoreEpoch"] = 1

##############
## Dataset  ##
##############
config["datasetFixword"] = 'ambient' 
config["DataDirectory"] = "../data"


# directory of training dataset
config["trainDatasetDirectory"] = "../data/__train_scenes__"
config["train_input"] = ['train_images'] 
config["train_target"] = ['train_images']


# directory of test dataset
config["testDatasetDirectory"] = "../data/__test_scenes__"
config["test_input"] = ['test_images'] 
config["test_target"] = ['test_images']

# noise levels of training/test
config['noise_levels'] = ['25','50','75']

config['train_corr_methods'] = ['flash'] # fixword for flash images
config['test_corr_methods'] = ['flash'] # fixword for flash images

# folder name for task (traing/test)
# config['task'] ='image_denoising_training' # folder name to contain training results
config['task'] ='image_denoising_test_pretrained' # folder name containing pretrained parameters

config["channelNames"] = [
                                #
]

if config["mode"] == "TRAINING":
    for i in range(len(config['train_corr_methods'])):
        config["channelNames"].append(config['train_corr_methods'][i] )


elif config["mode"] == "TEST":
    for i in range(len(config['test_corr_methods'])):
        config["channelNames"].append(config['test_corr_methods'][i] )


# mode selection
if config["numCandidates"] == 2:
    print('Config : Single buffer')
    config["numChannels"] = 3 *  config["numCandidates"] # (z, y) * 3 = 6
    config["numCorr"] = len(config["channelNames"])

elif config["numCandidates"] > 2:    
    print('Config : Multi buffer (Multi-correlation structure)')
    config["numChannels"] = 3 *  config["numCandidates"] # (z1, z2, z3, z4, y) * 3
    config["numCorr"] = len(config["channelNames"])


print('numChannels:', config["numChannels"])
print('numCorr:', config["numCorr"])

config["CANDIDATE_POS"] = 0





