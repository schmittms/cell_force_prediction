import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import pickle 
import utils.utils_data_processing as dp
import utils.utils_callbacks as utils_callbacks



def train(n_epochs, models, train_loader, validation_loader):

    for e in range(n_epochs):

        loss_values_train = {}
        loss_values_val = {}

        for m in models:
            m.reset_running_train_loss()
            m.reset_running_val_loss()

        for sample in train_loader: # This loop is critical: we want to load each sample only once, then pass to models (rather than vice versa)
            for m in models:
                m.training_step(sample, epoch=e)

        for sample in validation_loader: # This loop is critical: we want to load each sample only once, then pass to models (rather than vice versa)
            for m in models:
                m.validation_step(sample, epoch=e)

        for m in models:
            m.scheduler.step()
            m.log_end_of_epoch(epoch=e)

    
    torch.save({'model': models[i].state_dict(),
                'epoch': epoch,
                'loss': loss_mins[i],
                'test_split': dataset.test_cells}, 
               args.out_dir+'/model_%s.pt' % model_names[i])

        

def predict(out_dir, state_dict_dir, models, model_names, device, predict_dataset, scalar_callbacks=[], image_callbacks=[], ipython=False, max_items=1e10):
    d = datetime.now()
    dstr = d.strftime("_%y%m%d_%H%M")
    out_dirs = []
    out_dirs_inputs = []
    out_dirs_outputs = []
    if image_callbacks: out_dirs_grads = []

    if not ipython:
        for modelname in model_names:
            #print(modelname)
            out_dir_m = os.path.join(out_dir, 'predictions_' + modelname)# + dstr) 
            out_dir_m_inputs = os.path.join(out_dir, 'inputs_' + modelname)# + dstr) 
            out_dir_m_outputs = os.path.join(out_dir, 'outputs_' + modelname)# + dstr) 
            if image_callbacks: out_dir_m_grad = os.path.join(out_dir, 'gradients_' + modelname)# + dstr) 
            if not os.path.exists(out_dir_m):
                print('Making directory\t', out_dir_m)
                os.makedirs(out_dir_m)
            if not os.path.exists(out_dir_m_inputs):
                print('Making directory\t', out_dir_m_inputs)
                os.makedirs(out_dir_m_inputs)
            if not os.path.exists(out_dir_m_outputs):
                print('Making directory\t', out_dir_m_outputs)
                os.makedirs(out_dir_m_outputs)
            if image_callbacks and not os.path.exists(out_dir_m_grad):
                print('Making directory\t', out_dir_m_grad)
                os.makedirs(out_dir_m_grad)
                
            out_dirs.append(out_dir_m)
            out_dirs_inputs.append(out_dir_m_inputs)
            out_dirs_outputs.append(out_dir_m_outputs)
            if image_callbacks: out_dirs_grads.append(out_dir_m_grad)
        
    file_mapping = pd.DataFrame(columns=['file', 'source_dir',
                         *['prediction_dir_model%u'%i for i in range(len(models))]])        
    #print(len(file_mapping))
    print("..................LOADING MODELS.................")
    t0 = time()
    for i in range(len(models)):
        state_dict = os.path.join(state_dict_dir,  'model_'+model_names[i]+'.pt')
        models[i].load_state_dict(torch.load(state_dict)['model'])
        models[i].eval()
        print('.........Loaded one model: %0.2f................'%(time()-t0))
        models[i].to(device)
        print('.........Pushed one model to device: %0.2f................'%(time()-t0))
        
    return_vals = []
    hist_values = [[],] * len(models)

    t0 = time()
    
    for n, sample in enumerate(predict_dataset.loader):
        if n>max_items: break

        inputs_multiple, outputs, sample_dir, sample_name = sample
        outputs = outputs.to(device)
        file_mapping.loc[n] = [sample_name, sample_dir,*(['none'] * len(models))] 

        cellmask = np.load(os.path.join(sample_dir[0], sample_name[0]))
        imshape = inputs_multiple[0].shape[-1]
        cellmask = dp.CellCrop(imshape)(cellmask)[4]
        cellmask = scipy.ndimage.binary_dilation(cellmask, structure=disk(radius=50))

        model_outputs = []

        for l, inputs in enumerate(inputs_multiple):
            inputs = inputs.to(device)
            inputs.requires_grad = True

            for m in range(len(models)//len(inputs_multiple)):
                i = m + l*len(models)//len(inputs_multiple)
                if not ipython: file_mapping.loc[n]['prediction_dir_model%u'%i] = out_dirs[i]
                models[i].optimizer.zero_grad()
                prediction = models[i](inputs)
                
                if l==0 and m==0 and n==0: print("........ONE SAMPLE LOADED AND PREDICTED IN %0.2f"%(time()-t0))
                # Do this first because it includes gradients, don't want them getting mixed up during other stuff
                if image_callbacks:
                    for cb in image_callbacks:
                        #if not os.path.exists(cbdir): os.mkdir(cbdir)
                        savepath = os.path.join(out_dirs_grads[i], sample_name[0])  
                        cb(inputs, outputs, prediction, cellmask, save_to=savepath)

                
                if scalar_callbacks:    
                    cb_values = []
                    cb_names = []

                    prediction = prediction.cpu().detach().numpy()
                    O = outputs.cpu().detach().numpy()
                    for cb in scalar_callbacks:

                        if isinstance(cb, callbacks.GradientStats):
                            cb_out, cb_name = cb(inputs.cpu().detach().numpy(), cellmask, grad_loc=savepath)
                        else:
                            cb_out, cb_name = cb(prediction[0], O[0], cellmask)

                        if isinstance(cb, callbacks.HistStats) or isinstance(cb, callbacks.HistStatsMagOnly):
                            if n==0: # cb will output [[list of hists], [[bin1, bin2] for entire list of hists]]
                                if i==0:
                                    hist_values = [[],] * len(models) # nest: [modelidx, list
                                hist_values[i] = cb_out[0]
                                hist_bins = cb_out[1]
                                hist_names = cb_name
                            else:
                                hist_values[i] = [Hold + Hnew for Hold, Hnew in zip(hist_values[i], cb_out[0])]
                                #assert 1==0
                        else:
                            cb_values.append(cb_out)
                            cb_names.append(cb_name)

                    cb_values = np.asarray([x for q in cb_values for x in q]) #flatten
                    cb_names = np.asarray([x for q in cb_names for x in q]) #flatten

                    # Make dataframe if necessary
                    if n==0 and i==0:
                        cb_all_dfs = [pd.DataFrame(columns=[c for c in cb_names]) for m in range(len(models))]

                    if i==0:
                        for df in cb_all_dfs:
                            df.loc[n] = 0 # initialize to 0

                    for val, c in zip(cb_values, cb_names):
                        cb_all_dfs[i].loc[n].loc[c] = val.cpu().numpy() if torch.is_tensor(val) else val

    # We also want to make a csv file to map the predictions to the originals                     
                #print(sample_name)
                    pred = prediction #.cpu().numpy()

                    if ipython: 
                        model_outputs.append(pred)
                    else: 
                        np.save(os.path.join(out_dirs[i], sample_name[0]), pred)
                        np.save(os.path.join(out_dirs_inputs[i], sample_name[0]), inputs.cpu().detach().numpy())
                        np.save(os.path.join(out_dirs_outputs[i], sample_name[0]), outputs.cpu().detach().numpy())
                        #np.save(os.path.join(out_dirs[i], 'inputs_' + sample_name[0]), inputs_multiple)

 
        if ipython: return_vals.append([np.asarray([x.cpu().numpy() for x in inputs_multiple]), outputs.cpu().numpy(), np.asarray(model_outputs)])

        if not n%10: print('Time for %u frames: %0.2f'%((n+1), time()-t0))
    #       if n==2: break    
    if not ipython:
        file_mapping.to_csv(os.path.join(out_dir, 'file_mapping.csv'))
    if scalar_callbacks:
        for m in range(len(model_names)):
            cb_all_dfs[m].to_csv(os.path.join(out_dir, '%s_callback_values.csv'%model_names[m]))
            if hist_values[m]: #then hist values was assigned to
                with open(os.path.join(out_dir, '%s_hist_dict.p'%model_names[m]), 'wb') as handle:
                    histdict ={histname: [hist, histbins] for hist, histbins, histname in zip(hist_values[m], hist_bins, hist_names)}
                    print(len(histdict))
                    print(histdict.keys())
                    print(handle)
                    print(histdict[list(histdict.keys())[0]][0].shape)
                    pickle.dump(histdict, handle)
    print('Time for one cell:\t%.2f'%(time()-t0))         
    if ipython:
        return return_vals
    else: 
        return

def eval_predictions(out_dir,  model_names, scalar_callbacks=[], image_callbacks=[], label=''):
    d = datetime.now()
    dstr = d.strftime("_%y%m%d_%H%M")
    out_dirs = []
    out_dirs_inputs = []
    out_dirs_outputs = []

    # Get out directories the same way they were generated -- via model names
    for modelname in model_names:
        out_dir_m = os.path.join(out_dir, 'predictions_' + modelname)# + dstr) 
        out_dir_m_inputs = os.path.join(out_dir, 'inputs_' + modelname)# + dstr) 
        out_dir_m_outputs = os.path.join(out_dir, 'outputs_' + modelname)# + dstr) 
            
        out_dirs.append(out_dir_m)
        out_dirs_inputs.append(out_dir_m_inputs)
        out_dirs_outputs.append(out_dir_m_outputs)
        
    print("..................LOADING MODELS.................")
    t0 = time()
        
    return_vals = []
    hist_values = [[],] * len(model_names)

    t0 = time()
    
    print("OUT DIR", out_dir)
    for n, filename in enumerate(os.listdir(out_dirs[0])):
        outputs = np.load(os.path.join( out_dirs_outputs[0], filename))
        model_outputs = []

        for modelidx in range(len(model_names)):
            inputs = np.load( os.path.join(out_dirs_inputs[modelidx], filename))
            predictions = np.load( os.path.join( out_dirs[modelidx], filename))

            
            inputs = np.squeeze(inputs) # before this, had shape (1, 1, H, W)
            outputs = np.squeeze(outputs) # had shape (1, 2, H, W)
            predictions = np.squeeze(predictions) # HAD SHAPE (1,2, H, W)
            # CAUTION: NOT THE SAME CELL MASK!
            
            cellmask = copy.copy(inputs)
            if len(cellmask.shape)==3: cellmask = cellmask[0]
            #cellmask -= np.mean(cellmask)*0.5
            
            cellmask /= np.max(inputs)
            cellmask = 1.*(cellmask > 0.1)
            
            cellmask = scipy.ndimage.binary_dilation(cellmask, structure=disk(radius=20))

            
            if n==0: 
                print("........ONE SAMPLE LOADED AND PREDICTED IN %0.2f"%(time()-t0))
                imshow_test(inputs, cellmask, outputs, predictions, out_dir, modelidx)
            
            if scalar_callbacks:    
                cb_values = []
                cb_names = []

                for cb in scalar_callbacks:

                    if isinstance(cb, callbacks.GradientStats):
                        cb_out, cb_name = cb(inputs, cellmask, grad_loc=savepath)
                    else:
                        cb_out, cb_name = cb( np.squeeze(predictions), np.squeeze(outputs), cellmask)

                    if isinstance(cb, callbacks.HistStats) or isinstance(cb, callbacks.HistStatsMagOnly):
                        if n==0: # cb will output [[list of hists], [[bin1, bin2] for entire list of hists]]
                            if modelidx==0:
                                hist_values = [[],] * len(model_names) # nest: [modelidx, list
                            hist_values[modelidx] = cb_out[0]
                            hist_bins = cb_out[1]
                            hist_names = cb_name
                        else:
                            hist_values[modelidx] = [Hold + Hnew for Hold, Hnew in zip(hist_values[modelidx], cb_out[0])]
                    else:
                        cb_values.append(cb_out)
                        cb_names.append(cb_name)
                
                cb_values = np.asarray([x for q in cb_values for x in q]) #flatten
                cb_names = np.asarray([x for q in cb_names for x in q]) #flatten

                # Make dataframe if necessary
                if n==0 and modelidx==0:
                    cb_all_dfs = [pd.DataFrame(columns=[c for c in cb_names]) for m in range(len(model_names))]

                if modelidx==0:
                    for df in cb_all_dfs:
                        df.loc[n] = 0 # initialize to 0

                for val, c in zip(cb_values, cb_names):
                    cb_all_dfs[modelidx].loc[n].loc[c] = val.cpu().numpy() if torch.is_tensor(val) else val

        if not n%10: print('Time for %u frames: %0.2f'%((n+1), time()-t0))

    if scalar_callbacks:
        for m in range(len(model_names)):
            cb_all_dfs[m].to_csv(os.path.join(out_dir, '%s_callback_values_%s.csv'%(model_names[m], label)))
            if hist_values[m]: #then hist values was assigned to
                with open(os.path.join(out_dir, '%s_hist_dict_%s.p'%(model_names[m], label)), 'wb') as handle:
                    histdict = {histname: [hist, histbins] for hist, histbins, histname in zip(hist_values[m], hist_bins, hist_names)}
                    print(histdict[list(histdict.keys())[0]][0].shape)
                    pickle.dump(histdict, handle)
    print('Time for one cell:\t%.2f'%(time()-t0))         
    return


