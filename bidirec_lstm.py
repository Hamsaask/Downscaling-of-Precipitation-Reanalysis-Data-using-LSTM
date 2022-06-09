import xarray as xr
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import stats
import h5netcdf
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras import initializers
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import concatenate
from tensorflow.keras.layers import Bidirectional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
gc.disable()


def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(df) is list else df.shape[1]
        df = pd.DataFrame(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
                agg.dropna(inplace=True)
        return agg

ncep='/home/ramesh/bias/Final_Programs/JJAS.nc'
imd='/home/ramesh/bias/Final_Programs/JJAS_IMD.nc'
mask1 = '/nmhs1/reanalysis/agromet/reanalysis/inter/mask_fin.nc'
#ncep_imd = '/nmhs1/reanalysis/agromet/reanalysis/inter/rf_dly_ncep_imd1.nc'
ncep_ds = xr.open_dataset(ncep)
imd_ds=xr.open_dataset(imd)

cent=0

#data_ds = data_ds.to_array()
mask = xr.open_dataset(mask1)
mask = mask['MASK']
imd_ds= imd_ds['RF']
ncep_ds=ncep_ds['prate']

rmse_train = mask
mae_train = mask
rmse_test = mask
mae_test = mask
rmse_comb = mask
mae_comb = mask

lon= ncep_ds['XFINE']
#lon=lon[1:59]
latt=ncep_ds['YFINE']
#latt=latt[1:59]
times=ncep_ds['time'].values
np.savetxt('times_final.csv',times)
#times=pd.Series(times,index=None)


#print("MASK DATA",mask)
#print('ncep',ncep_ds)

#print(data_ds)

#print('at x , y :\n',data_ds[1,125,2,3])
#print('shape',data_ds.shape)

#k = mask.shape[2]
#l = mask.shape[3]
#a = ncep_ds.shape[1]
#print('a',a,'l',l,'k',k)
#count =0 
lati=ncep_ds.shape[1]
loni=ncep_ds.shape[2]
tt=ncep_ds.shape[0]
#print('lati',lati,'loni',loni,'time',tt)
#np.savetxt('time.csv',ncep_ds['time'])
p = pd.DataFrame([])
for j in range(loni):
    for i in range(lati):
        #print('running',i,j)
        if mask[0,i,j] == 1:
            
            ncep_i = ncep_ds[:,i,j]
            times=ncep_i['time'].values
            ncep_i=ncep_i.values
            #ncep_i=pd.Series(ncep_i,index=None)
            #print('ncep',ncep_i)
            imd_i =imd_ds[:,i,j].values
            #imd_i=pd.Series(imd_i,index=None)
            #print('imd',imd_i)
            lon_i=lon[j].values
            lat_i=latt[i].values
            #print('TIME',times)
            
            df=pd.DataFrame(data={'Variable':'RF','TFINE':times,'XFINE':lon_i ,'YFINE':lat_i,'rf':imd_i})
            #print('Dataframe',df)
            df1=pd.DataFrame(data={'Variable':'RFNCEP','TFINE':times,'XFINE':lon_i,'YFINE':lat_i,'rf':ncep_i})
            #print('Dataframe_NCEP',df1)


            data= pd.concat([df,df1])
            #print('Dataframe_Concat:',data)

            #rf = data_ds.sel(variable='RF',XFINE1=i,YFINE1=j)
            #data = rf.to_dataframe("rf")
            #print(rf)

            ns = int(data.shape[0]/2)
            dfc1 = data[:ns ]
            dfc2 = data[ns:]
            #df = pd.merge(dfc1,dfc2, on = ["TFINE"])
            
            df = pd.merge(dfc2,dfc1, on = ["TFINE"])
            df.reset_index(drop=True,inplace=True)
            newdf = df.drop(['Variable_x','Variable_y','XFINE_x','XFINE_y','YFINE_x','YFINE_y'],axis=1)
            #newdf =newdf.rename(columns={'rf_x': 'ncep','rf_y':'imd'})
            #print('Prepared_data for LSTM : \n', newdf, '\n Data_shape\n', newdf.shape)
            newdf.reset_index(drop=True,inplace=True)
            newdf.set_index('TFINE')
            
            feature = newdf[['rf_x','rf_y']]
            #print('Feature',feature)
            feature = feature.astype(float)
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(feature)
            #print('MinMaxScaler Transformation: \n',scaled)
            
            # Series to supervised learning
            reframed = series_to_supervised(scaled, 1, 1)
            # drop columns we don't want to predict
            cd = scaled.shape[1]+1
            reframed.drop(reframed.iloc[:, cd:], axis=1, inplace=True)
            
            # Train_test split
            values = reframed.values
            tn = int(values.shape[0]*0.8)
            train = values[:tn]
            test = values[tn:]
            
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

            # LSTM model Hyperparametrs
            nodes = [50]
            dr_ly = [0.20]
            #optim = ['Adam']
            opt = tensorflow.keras.optimizers.Adam(learning_rate=0.1)
            optim=[opt]
            activ = ['tanh']
            epoch = [50]
            b_s = [64]
            input =(train_X.shape[1], train_X.shape[2])
            train_in = train_X
            train_out = train_y
            test_in = test_X
            test_out = test_y
        
            initializer = tensorflow.keras.initializers.Orthogonal()
            #initializer = tensorflow.keras.initializers.GlorotNormal()
            bias_initializer = tensorflow.keras.initializers.HeNormal()
            #bias_initializer = 'Zeros'
            for unit in nodes:
                for dropout in dr_ly:
                    for activation in activ:
                        for optimizer in optim:
                            for epochs in epoch:
                                for batch in b_s:
                                    model = Sequential()
                                    model.add(Bidirectional(LSTM(unit, input_shape=input)))
                                    model.add(Dropout(dropout))
                                    model.add(Dense(1, kernel_initializer=initializer,use_bias=False, bias_initializer=bias_initializer))
                                    model.add(Activation(activation))
                                    model.compile(loss='mae', optimizer=optimizer, metrics = ['accuracy'])
                                    history = model.fit(train_in, train_out, epochs=epochs, batch_size=batch, validation_data=(test_in, test_out), verbose=2, shuffle=False)

                                    trhat = model.predict(train_X)
                                    
                                    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
                                    inv_trhat = concatenate((trhat, train_X[:, 1:]), axis=1)
                                    #print(inv_trhat)
                                    inv_trhat = scaler.inverse_transform(inv_trhat)
                                    inv_trhat = inv_trhat[:,0]
                                    
                                    train_y = train_y.reshape((len(train_y), 1))
                                    inv_tny = concatenate((train_y, train_X[:, 1:]), axis=1)
                                    inv_tny = scaler.inverse_transform(inv_tny)
                                    inv_tny2 = inv_tny[:,0]
                                    inv_tny1=inv_tny[:,1]
                                    
                                    prediction = pd.DataFrame(dict(actual = inv_tny1, predicted = inv_trhat)).reset_index()
                                    #print("Train_Prediction :\n", prediction)
                                    #prediction.to_csv("Train_data_AAAAA.csv", index=False)
                                    Train_data= prediction
                                    Train_data[Train_data['predicted'] < 0] = 0
                                    tr_rmse = sqrt(mean_squared_error(Train_data['actual'], Train_data['predicted']))
                                    tr_mae = mean_absolute_error(Train_data['actual'], Train_data['predicted'])
                                    #tr_mape =  np.mean(np.abs((Train_data['actual'] - Train_data['predicted']) / Train_data['actual'])) * 100
                                    print('TRAIN_Data Errors','Tr_RMSE: %.3f' % tr_rmse,'Tr_MAE: %.3f' % tr_mae)#,'Tr_MAPE: %.3f' % tr_mape,)
                                    #print('Latlon',lat_i,lon_i)

                                    yhat = model.predict(test_X)
                                    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
                                    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
                                    inv_yhat = scaler.inverse_transform(inv_yhat)
                                    inv_yhat = inv_yhat[:,0]
                                    test_y = test_y.reshape((len(test_y), 1))
                                    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
                                    inv_y = scaler.inverse_transform(inv_y)
                                    inv_y2 = inv_y[:,0]
                                    inv_y1=inv_y[:,1]
                                    forecast = pd.DataFrame(dict(actual = inv_y1, predicted = inv_yhat)).reset_index()
                                    #print("\nTest_Prediction:\n", forecast)
                                    forecast.to_csv('forecast.csv')
                                    Test_data = forecast
                                    Test_data[Test_data['predicted']<0]=0
                                    ts_rmse = sqrt(mean_squared_error(Test_data['actual'], Test_data['predicted']))
                                    ts_mae = mean_absolute_error(Test_data['actual'], Test_data['predicted'])
                                    #ts_mape =  np.mean(np.abs((Test_data['actual'] - Test_data['predicted']) / Test_data['actual'])) * 100
                                    print('TEST_Data Errors','Ts_RMSE: %.3f' % ts_rmse,'Ts_MAE: %.3f' % ts_mae)#,'Ts_MAPE: %.3f' % ts_mape)

                                    n = Train_data.shape[0]
                                    d = np.abs(  np.diff( Train_data['actual']) ).sum()/(n-1)
                                    errors = np.abs(Test_data['actual'] - Test_data['predicted'] )
                                    MASE = errors.mean()/d
                                    combined_data = Train_data.append(Test_data)
                                    print('combined_data',combined_data)
                                    

                                    rmse = sqrt(mean_squared_error(combined_data['actual'], combined_data['predicted']))
                                    mae = mean_absolute_error(combined_data['actual'], combined_data['predicted'])
                                    #mape = np.mean(np.abs((combined_data['actual'] - combined_data['predicted']) / combined_data['actual'])) * 100
                                    print('combined_data Errors','RMSE: %.3f' % rmse,'MAE: %.3f' % mae,'MASE: %.4f' % MASE)
                                    
                                    
                                    file='ncep_pred'+str(i)+str(j)
                                    
                                    cent=cent+1
                                    #print(tt)
                                    import pickle

                                    ds = xr.Dataset(
                                            data_vars=dict(ncep_act=(["time"], ncep_i[:-1]),
                                                   ncep_pred=(["time"],combined_data['predicted'].values),
                                                   imd_act=(['time'],combined_data['actual'].values),
                                                 ),
            
                                    coords=dict(
                                        time=times[:-1],
                                        center=cent,
                                             ),
                                    attrs=dict(
                                        description="NCEP and IMD actual and predicted values .",
                                        center=[lat_i,lon_i],
                                            ),
                                                 )
                                    
                                   
                                    ds.to_netcdf('NCEP_PRED/'+file+'.nc')



                                    '''
                                    outfile=open('NCEP_PRED/'+file+'.pkl','wb')
                                    pickle.dump(ds,outfile)
                                    outfile.close()

                                    
                                    ncep_pred[:,i,j]=combined_data['predicted'].values
                                    imd_act[:,i,j]=combined_data['actual'].values
                                    ncep_act[:,i,j]=ncep_i.values
                                    '''
                                    gc.collect()
                                    


                                    
            rmse_test[0,i,j]= ts_rmse
            mae_test[0,i,j]= ts_mae
            rmse_train[0,i,j]=tr_rmse
            mae_train[0,i,j]=ts_rmse
            rmse_comb[0,i,j]=rmse
            mae_comb[0,i,j]=mae
            #print('\nrmse and mae @ i, j\n',newarray[0:0:i,j])
            #input("Enter to continue")
            
        else:
            rmse_train[0,i,j] = -999999
            mae_train[0,i,j] =-999999
            rmse_test[0,i,j] = -999999
            mae_test[0,i,j] =-999999
            rmse_comb[0,i,j] = -999999
            mae_comb[0,i,j] =-999999
            #print(newarray[0,0,i,j])
        #input("Enter to continue")
            
        





rmse_test.to_netcdf("rmse_test1.nc")
mae_test.to_netcdf("mae_test1.nc")

rmse_train.to_netcdf("rmse_train1.nc")
mae_train.to_netcdf("mae_train1.nc")

rmse_comb.to_netcdf("rmse_comb1.nc")
mae_comb.to_netcdf("mae_comb1.nc")




#ncep_pred.to_netcdf('PREDICTED_NCEP_TRIAL.nc')






















