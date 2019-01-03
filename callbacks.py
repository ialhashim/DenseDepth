import keras
from keras import backend as K
from utils import DepthNorm

def get_nyu_callbacks(model, basemodel, train_generator, test_generator, runPath):
    callbacks = []

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2)
    callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: show intermediate results and save current model
    '''
    is_debug = False
    is_invdepth = True

    minDepth = 10.0
    maxDepth = 1000.0

    resolution_Depth = 240

    def MyCheckPointCallback(epoch):
        if epoch % 3 == 0 if not is_debug else epoch % 3 == 0:
            # Sample validation set
            for i in range(0,len(test_generator),int(len(test_generator)/5)):
                img_x,img_y = test_generator[i]
                img_x,img_y = img_x[0,:,:,:], img_y[0,:,:,:]

                gt_rgb, gt_depth = img_x, img_y
                prediction =  utils.predict(model, img_x, numChannels=1).reshape(resolution_Depth, int(resolution_Depth*4/3),3)

                if is_invdepth:
                    gt_depth = DepthNorm(gt_depth)
                    prediction = DepthNorm(prediction)   

                prediction = np.clip(prediction, minDepth, maxDepth)/maxDepth*255
                utils.save_png(np.clip(prediction,0,255), runPath+'/samples/i' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.OURS.SceneDepth.png')
                
                if epoch == 0:
                    utils.save_png(gt_rgb*255, runPath+'/samples/i' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.GT.FinalImage.png')
                    gt_depth = np.clip(gt_depth, minDepth, maxDepth)/maxDepth*255            
                    utils.save_png(np.clip(gt_depth,0,255), runPath+'/samples/i' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.GT.SceneDepth.png')
                
            # Sample training set
            gt_count = 4
            gt_range = gt_count * 30
            for i in range(0, gt_range, int(gt_range / gt_count)):
                img_x,img_y = train_generator[i]
                img_x,img_y = img_x[0,:,:,:], img_y[0,:,:,:]

                gt_rgb, gt_depth = img_x, img_y
                prediction =  utils.predict(model, img_x, numChannels=1).reshape(resolution_Depth,int(resolution_Depth*4/3),3)

                if is_invdepth:
                    gt_depth = DepthNorm(gt_depth)
                    prediction = DepthNorm(prediction)  

                prediction = np.clip(prediction, minDepth, maxDepth)/maxDepth*255
                utils.save_png(np.clip(prediction,0,255), runPath+'/samples/gt' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.OURS.SceneDepth.png')
                
                if epoch == 0:
                    utils.save_png(gt_rgb*255, runPath+'/samples/gt' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.GT.FinalImage.png')
                    gt_depth = np.clip(gt_depth, minDepth, maxDepth)/maxDepth*255
                    utils.save_png(np.clip(gt_depth,0,255), runPath+'/samples/gt' + f'{i:04}' +'.e'+ f'{epoch:05}' + '.GT.SceneDepth.png')

        # Save intermediate model
        if epoch % 5 == 0 and not is_debug:
            model_filename = runPath+'/model_epoch_'+str(epoch)+'.h5'
            basemodel.save(model_filename)
            print('Saved model to file:\t', model_filename)

    callbacks.append( keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: MyCheckPointCallback(epoch)) )
    '''

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir):
            super().__init__(log_dir=log_dir)
        def on_epoch_end(self, epoch, logs=None):
            logs.update({'zlr': K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)
    callbacks.append( LRTensorBoard(log_dir=runPath) )

    return callbacks