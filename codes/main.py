# %% Initializaion
import os
import datetime
import time
import psutil
from logger import Logger
import numpy as np
import utils_image as util
import psutil

os.environ['MPLCONFIGDIR'] = '/tmp'  # 0, 1, 2, 3s

# Disable Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0, 1, 2, 3s
import tensorflow as tf

import loader
from layers import MainNet
from utilities import printFunc
from config import config, gpus
from pytz import timezone

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


class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, file):
        super(LossCallback, self).__init__()
        self.file = file

    def on_epoch_end(self, epoch, logs={}):
        val = ' '.join(map(str, logs.values()))
        self.file.write(val + '\n')
        self.file.flush()


# for test, use npy file
def evaluateDataset_fnf_awgn_v2(dataset, name, epoch=0, isTrainDataset=False, numOutputBatch=1, batchStride=1, loss=None, save=True, Multi=False, Test = False):
    
    if Multi == False:
        print('Test Config : Single buffer.')
    elif Multi == True:
        print('Test Config : Multi buffer. Please Check the indexing')
        
    cnt = 0
    test_dir = os.path.join(config['DataDirectory'],config['task'])
    util.mkdir(test_dir)
    util.mkdir(os.path.join(test_dir,'output'))

    Loss_average = 0.0
    PSNR_average = 0.0

    Loss_average_corr = 0.0
    PSNR_average_corr = 0.0

    logfile_ours = open(os.path.join(test_dir,'loss_ours_%s_%d.txt') % (name,epoch), 'a+')

    # Iterate over batchs or images
    start = time.time()

    for (batchIdx, element) in enumerate(dataset):
        if batchIdx % batchStride != 0:
            continue

        if isTrainDataset:
            input, ref = element
        else:
            sceneName, input, ref = element
            sceneName = sceneName.numpy()[0].decode('utf-8')

        sname =  os.path.basename(sceneName).split('.')[0].split('_')
        sceneName = sname[0] + '_' + sname[1] + '_' + sname[4] 
        corr = parseInput_single(config, input)
        # with tf.device("/cpu:0"):
        (out_color) = model(input, training=False)

        # Iterate over each image
        for sampleIdx, img in enumerate(out_color):
            if save:
                if Multi == False:
                    corr_img = corr[:, :, :, 0:3] # z (0:3); y (3:6)
                elif Multi == True:
                    corr_img = corr[:, :, :, 3:6] # z (0:6); y (6:9)  
                

                ours_loss = tf.reduce_mean(config['loss'](ref, img)).numpy()
                corr_loss = tf.reduce_mean(config['loss'](ref, corr_img)).numpy()

                np_corr = tf.squeeze(corr_img)
                np_ref = tf.squeeze(ref)
                # np_out = tf.squeeze(img)
                np_out = img.numpy()

                np_corr = np_corr.numpy()
                # np_out = np_out.numpy()
                np_ref = np_ref.numpy()
                
                np_corr = np_corr * 255
                np_out = np_out *255
                np_ref = np_ref *255

                np_corr = np.clip(np_corr, 0, 255)
                np_out = np.clip(np_out, 0, 255)
                np_ref = np.clip(np_ref, 0, 255)

                corr_psnr = util.calculate_psnr(np_corr, np_ref, border=0)
                our_psnr = util.calculate_psnr(np_out, np_ref, border=0)

                corr_ssim = util.calculate_ssim(np_corr, np_ref, border=0)      
                our_ssim = util.calculate_ssim(np_out, np_ref, border=0)

                if Test:
                    filename = "output/%s_%s_epoch%04d_%s%.4f.png" % (
                        sceneName, name, epoch, 'PSNR', our_psnr)
                    filename_corr = "output/%s_%s_epoch%04d_%s%.4f.png" % (
                        sceneName, name, epoch, 'PSNR', corr_psnr)
                else:
                    filename = "output/%s_epoch%04d_%s%.4f.png" % (
                        sceneName, epoch, 'PSNR', our_psnr)
                    filename_corr = "output/%s_epoch%04d_%s%.4f.png" % (
                        sceneName, epoch,  'PSNR', corr_psnr)

                # Write output color
                util.imwrite(os.path.join(test_dir,filename.replace('.png', '_out.png')), np_out)
               
                # Print error
                logfile_ours.write('{0: >35}'.format(sceneName) + '-' + 'Loss :  ' + "{:.6f}".format(ours_loss) + ' PSNR :  ' +   "{:.6f}".format(our_psnr)  + ' SSIM :  ' +  "{:.6f}".format(our_ssim) + '\n')
                logfile_ours.flush()

                Loss_average += ours_loss
                PSNR_average += our_psnr

                Loss_average_corr += corr_loss
                PSNR_average_corr += corr_psnr

            cnt += 1
            if numOutputBatch >= 0 and cnt >= numOutputBatch:
                break
    printFunc("\tTook", time.time() - start, "s for evaluateDataset()")


    Loss_average /= cnt
    PSNR_average /= cnt
    Loss_average_corr /= cnt
    PSNR_average_corr /= cnt

    logfile_ours.write('Loss_average_ :  ' + "{:.6f}".format(Loss_average) + ' PSNR_average_ :  ' +   "{:.6f}".format(PSNR_average)  + '\n')
    logfile_ours.flush()
    logfile_ours.close()

    return Loss_average, PSNR_average


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()

    multi = False
    if config["numCandidates"] == 2: # (y, z) * 3
        print('Config : Single buffer')
        multi = False
    elif config["numCandidates"] > 2: # (y, z1, z2, z3, z4) * 3.
        print('Config : Multi buffer (Multi-correlation structure)')
        multi = True

    with strategy.scope():
        
        # build our model
        model = buildModel()
        
        # set learning rate
        initial_learning_rate = 0.0005
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        
        # set task folder and chechpoint folder
        task_dir = os.path.join(config['DataDirectory'],config['task'])
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_dir = os.path.join(task_dir, '__checkpoints__')         
        util.mkdir(checkpoint_dir)
        checkpoint_path = checkpoint_dir + "/cp-%s" % config['title']

        if config['mode'] == "TEST":
            dataset_test = loader.load_fnf_awgn(config, False, None)

            # Load weights
            checkpoint.restore(checkpoint_path + "-" + config["loadEpoch"])
            print("Restore network parameters for Test : {}".format(checkpoint_path + "-" + config["loadEpoch"]))
            evaluateDataset_fnf_awgn_v2(dataset_test, 'final', epoch=int(config["loadEpoch"]), numOutputBatch=-1, batchStride=1, Multi = multi, Test = True)

        elif config['mode'] == "TRAINING":
            dataset = loader.load_fnf_awgn(config, True, None)
            dataset_test = loader.load_fnf_awgn(config, False, None)

            # Just return dataset to make it as distributed
            # The patchBatch will be allocated for each replica
            dataset = strategy.experimental_distribute_dataset(dataset)


            if config['retrain'] == False:
                evaluateDataset_fnf_awgn_v2(dataset_test, 'initial_result', epoch=0, numOutputBatch=-1, batchStride=1, Multi = multi)
                
            initial_epoch = 1

            # Retain option
            if config['retrain'] == True:
                checkpoint.restore(checkpoint_path + "-" + str(config["restoreEpoch"]))
                print("Restored network parameters : {}".format(checkpoint_path + "-" + str(config["restoreEpoch"])))
                initial_epoch = config["restoreEpoch"] + 1
            
            start = time.time()

            def compute_loss(labels, predictions):
                per_example_loss = config['loss'](labels, predictions)
                # reduce_mean except the first dimension (batch dim)
                per_example_loss = tf.reduce_mean(per_example_loss, axis=[1, 2])
                return tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=len(gpus) * config['patchBatch'])

            def train_step(model, inputs):
                images, refs = inputs

                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = compute_loss(refs, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                return loss

            # `run` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train_step(model, dataset_inputs):
                per_replica_losses = strategy.run(train_step, args=(model, dataset_inputs))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            
            

            process = psutil.Process(os.getpid())
            logger = Logger(os.path.join(task_dir,'log.txt'), 'a+', config['timezone'])
            logfile_training = open(os.path.join(task_dir,'log_train_loss.txt'), 'a+')

            
            globalStart = time.time()
            for epoch in range(initial_epoch, config['epochs'] + 1):
                startTime = time.time()              

                total_loss = 0.0
                batch = 0

                # adjust lr with respect to epoch
                if epoch < 15:
                    optimizer.learning_rate.assign(0.0005) 
                elif epoch > 15:
                    optimizer.learning_rate.assign(0.0001) 


                start = time.time()
                for x in dataset:
                    total_loss += distributed_train_step(model, x)
                    batch += 1
                    if batch == 1 or batch % 10 == 0:
                        mem = int(process.memory_info().rss / 1024 / 1024)
                        print('[ Denoising ] [ Epoch %03d / %03d ] [ Batch %04d / %04d ] Train loss: %.6f, Memory: %d MB, lr: %.5f' %
                              (epoch, config['epochs'], batch, 0, total_loss.numpy() / batch, mem, optimizer.lr.numpy()))
                        # for debug
                        used_mem = psutil.virtual_memory().used
                        print("used memory: {} Mb".format(used_mem / 1024 / 1024))
                train_loss = total_loss / batch

                # log training
                endTime = time.time() - startTime
                if config['timezone'] == None:
                    strDate = datetime.datetime.now().astimezone(None).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    strDate = datetime.datetime.now().astimezone(timezone(config['timezone'])).strftime('%Y-%m-%d %H:%M:%S')
                strPrintLoss = "epoch : %d (%f sec, %s) | training loss : %f\n" % (epoch, endTime, strDate, train_loss)
                print(strPrintLoss)
                logfile_training.write(strPrintLoss)
                logfile_training.flush()

                # Save checkpoint
                if epoch % 1 == 0:
                    checkpoint.save(checkpoint_path)

                duration = time.time() - start
                logger.add_loss(train_loss.numpy(), epoch, title='Multi', time=duration)

            # test on final epoch
            evaluateDataset_fnf_awgn_v2(dataset_test, 'final_result', epoch=0,numOutputBatch=-1, batchStride=1, save=True, Multi = multi, Test = True)        

            print('Training took', '%.2fs' % (time.time() - globalStart),
                  'for %d epochs.' % (config['epochs']))