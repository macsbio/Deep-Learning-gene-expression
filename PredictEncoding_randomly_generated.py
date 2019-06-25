import numpy as np
import pickle

from SmallParser import SmallDatasetParser, get_doubling_xy, get_x, read_data
from Processing import correct_slopes, decorrect_slopes, normalize_features, normalize_with_norms, \
    denormalize_with_norms, normalize_total, split_train_test
import AutoEncoders
import GeneLists
import CompoundLists


import matplotlib.pyplot as plt

from keras.layers import Input, Dense, concatenate, Conv2D, SpatialDropout2D, MaxPooling2D, Flatten, Dropout, Reshape
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
import scipy.io as sio
from keras.metrics import mse, mae, mape
from keras import backend as K
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.neighbors import KNeighborsRegressor



def cnn_model(x_train, x_valid, y_train, y_valid, norms_X, norms_Y):
    """A Convolutional Neural Network consisting of three coupled convolutional and maxpooling layers and one Dense layer, with data preprocessing like in the Autoencoders."""
    
    #determine the shape of the source and target domain data.
    trg_timepoints = 3
    if x_train.shape[1]!=y_train.shape[1]: 
        trg_timepoints = 4
    padded = False
    genes = x_train.shape[1] // 3
    MIN_GENES = 50

    #if the number of genes in the gene set are less than 50 we intoduce zero padding to insure input data has a dimension of at least 50x3
    if genes < MIN_GENES:
        padded = True
        # Assume y_train has the same amount of genes...
        # shape of train: (num_samples, genes x timepoints)
        x_pad = np.zeros((x_train.shape[0], (MIN_GENES - genes) * 3))
        x_v_pad = np.zeros((x_valid.shape[0], (MIN_GENES - genes) * 3))
        y_pad = np.zeros((y_train.shape[0], (MIN_GENES - genes) * trg_timepoints))
        y_v_pad = np.zeros((y_valid.shape[0], (MIN_GENES - genes) * trg_timepoints))
        x_train = np.append(x_train, x_pad, axis=1)
        x_valid = np.append(x_valid, x_v_pad, axis=1)
        y_train = np.append(y_train, y_pad, axis=1)
        y_valid = np.append(y_valid, y_v_pad, axis=1)

    res_x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] // 3, 3, 1))
    res_x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1] // 3, 3, 1))
    print(y_train.shape)
    print(y_train.shape)
    res_y_train = y_train.reshape((y_train.shape[0], y_train.shape[1] // trg_timepoints, trg_timepoints, 1))
    res_y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1] // trg_timepoints, trg_timepoints, 1))
    print(res_y_valid.shape)
    print(res_y_train.shape)

    input_layer = Input(shape=(x_train.shape[1] // 3, 3, 1))
    # define convolutional+maxpooling  layer 1
    x = Conv2D(16, kernel_size=(10, 1), activation='relu', data_format='channels_last')(input_layer)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 1), data_format='channels_last')(x)
    # define convolutional+maxpooling  layer 2
    x = Conv2D(8, kernel_size=(10, 1), activation='relu', data_format='channels_last')(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 1), data_format='channels_last')(x)
    #define convolutional+maxpooling  layer 3
    x = Conv2D(4, kernel_size=(2, 1), activation='relu', data_format='channels_last')(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 1), data_format='channels_last')(x)
    #flatten output into 1 dimensional vector.
    x = Flatten()(x)

    x = Dense(20, activation='relu', activity_regularizer=regularizers.l1(10e-6))(x)
    #up-sampling
    x = Dense(30, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(y_train.shape[1], activation='sigmoid')(x)

    x = Reshape((y_train.shape[1] // trg_timepoints, trg_timepoints, 1))(x)

    model = Model(input_layer, x)
    #define coset function mean absolute error (mae) and optimiser Adam.
    adam = Adam()
    model.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])
    #model fit
    model.fit(res_x_train, res_y_train, epochs=10000, batch_size=128, shuffle=True,
              validation_data=(res_x_valid, res_y_valid),
              verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=40, verbose=0, mode='auto')])

    recon_train = model.predict(res_x_train)
    recon_valid = model.predict(res_x_valid)

    recon_train_res = recon_train.reshape((res_y_train.shape[0], -1))
    recon_valid_res = recon_valid.reshape((res_y_valid.shape[0], -1))

    # Drop the padding for calculating true error
    if padded:
        drop = MIN_GENES - genes
        x_train = np.delete(x_train, np.s_[-drop * 3 - 1:-1], 1)
        y_train = np.delete(y_train, np.s_[-drop * trg_timepoints - 1:-1], 1)
        x_valid = np.delete(x_valid, np.s_[-drop * 3 - 1:-1], 1)
        y_valid = np.delete(y_valid, np.s_[-drop * trg_timepoints - 1:-1], 1)

        recon_train_res = np.delete(recon_train_res, np.s_[-drop * trg_timepoints - 1:-1], 1)
        recon_valid_res = np.delete(recon_valid_res, np.s_[-drop * trg_timepoints - 1:-1], 1)

    X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid = _correct_data(x_train, x_valid, y_train, y_valid,
                                                                                 recon_train_res, recon_valid_res,
                                                                                 norms_X,
                                                                                 norms_Y)

    train_mae = np.average(np.absolute(Y_train - recon_train))
    train_mse = np.average(np.square(Y_train - recon_train))
    val_mae = np.average(np.absolute(Y_valid - recon_valid))
    val_mse = np.average(np.square(Y_valid - recon_valid))

    return X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, np.array(
        [train_mae, train_mse, val_mae, val_mse])


def merge_autoencoders(inputs, encoder, decoder):
    model_input = Input(shape=(inputs,))
    encoder_output = encoder(model_input)
    decoder_output = decoder(encoder_output)
    network = Model(model_input, decoder_output)


    adam = Adam(lr=0.0005, decay=0.00001)
    network.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])
   
    return network


def mod_autoencoder_model(inputs): #  this defines the modified autoencoder model
    #  define three encoder layers
    input_layer = Input(shape=(inputs,))
    encoded = Dense(70, activation='relu')(input_layer)
    encoded = Dense(70, activation='relu', activity_regularizer=regularizers.l1(10e-7))(encoded)
    encoded = Dense(60, activation='relu', activity_regularizer=regularizers.l1(10e-9))(encoded)
    encoder = Model(input_layer, encoded)

    #  define three decoder layers
    encoded_input = Input(shape=(60,))
    decoded = Dense(70, activation='relu', activity_regularizer=regularizers.l1(10e-7))(encoded_input)
    decoded = Dense(70, activation='relu')(decoded)
    decoded = Dense(inputs, activation="sigmoid")(decoded)
    decoder = Model(encoded_input, decoded)

    # define full model network; input, endcoder layers, decoder layers, and output.
    model_input = Input(shape=(inputs,))
    encoder_output = encoder(model_input)
    decoder_output = decoder(encoder_output)
    network = Model(model_input, decoder_output)
    #define optimiser - Adam and loss function mean absolute error (mae)
    adam = Adam(lr=0.0005, decay=0.00001)
    network.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])

    return network, encoder, decoder


def _correct_data(X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, norms_X, norms_Y):
    # Denormalize all data
    X_train = denormalize_with_norms(X_train, norms_X)
    X_valid = denormalize_with_norms(X_valid, norms_X)
    Y_train = denormalize_with_norms(Y_train, norms_Y)
    Y_valid = denormalize_with_norms(Y_valid, norms_Y)
    recon_train = denormalize_with_norms(recon_train, norms_Y)
    recon_valid = denormalize_with_norms(recon_valid, norms_Y)

    # Decorrect for slopes, otherwise normalization won't be correct
    X_train = decorrect_slopes(X_train, x_vivo)
    X_valid = decorrect_slopes(X_valid, x_vivo)
    Y_train = decorrect_slopes(Y_train, y_vivo)
    Y_valid = decorrect_slopes(Y_valid, y_vivo)
    recon_train = decorrect_slopes(recon_train, y_vivo)
    recon_valid = decorrect_slopes(recon_valid, y_vivo)

    # Normalize again, but using min/max over entire dataset instead of per feature.
    # This is to have a consistent error measurement across different experiments.
    combined = np.append(X_train, X_valid, axis=0)
    combined, norms = normalize_total(combined)
    X_train = combined[:X_train.shape[0]]
    X_valid = combined[X_train.shape[0]:]

    combined = np.append(Y_train, Y_valid, axis=0)
    combined, norms = normalize_total(combined)
    Y_train = combined[:Y_train.shape[0]]
    Y_valid = combined[Y_train.shape[0]:]

    recon_train = normalize_with_norms(recon_train, norms)
    recon_valid = normalize_with_norms(recon_valid, norms)

    # Correct for slopes again
    X_train = correct_slopes(X_train, x_vivo)
    X_valid = correct_slopes(X_valid, x_vivo)
    Y_train = correct_slopes(Y_train, y_vivo)
    Y_valid = correct_slopes(Y_valid, y_vivo)
    recon_train = correct_slopes(recon_train, y_vivo)
    recon_valid = correct_slopes(recon_valid, y_vivo)

    return X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid


def train_network(network, X_train, X_valid, Y_train, Y_valid):
    batch_size = 32

    # Fitting the data
    network.fit(X_train, Y_train, epochs=5000, batch_size=batch_size, shuffle=False, validation_data=(X_valid, Y_valid),
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=100, verbose=0, mode='auto')])


# Given X, create Y such that the instances now have doubling (r1->r1, r1->r2, r2->r1, r2->r2)
def X_to_doubled(X):
    d1_r1 = np.repeat(X[0::4], 2, axis=0)  # split data into first and second replicate
    d1_r2 = np.repeat(X[2::4], 2, axis=0)

    # Change into r1, r2, r1, r2
    interwoven = np.ravel(np.column_stack((d1_r1, d1_r2))).reshape(X.shape)
    return interwoven


# Given Y, create X such that the instances now have doubling (r1->r1, r1->r2, r2->r1, r2->r2)
def Y_to_doubled(Y):
    X = np.zeros(Y.shape)
    X[0::4] = Y[0::4]
    X[1::4] = Y[2::4]
    X[2::4] = Y[1::4]
    X[3::4] = Y[3::4]
    return X


def train_naive_encoder(X_train, X_valid, Y_train, Y_valid, norms_X, norms_Y):  # Dan: for untrained AE
    inputs = X_train.shape[1]
    #trains naive encoder model

    # define the three layer of the encoder
    input_layer = Input(shape=(inputs,))
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(160, activation='relu', activity_regularizer=regularizers.l1(10e-7))(encoded)
    encoded = Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-9))(encoded)
    encoder = Model(input_layer, encoded)

    # deinfe the three layers of of the decoder
    encoded_input = Input(shape=(32,))
    decoded = Dense(96, activation='relu', activity_regularizer=regularizers.l1(10e-7))(encoded_input)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(Y_train.shape[1], activation="sigmoid")(decoded) # Dan: I changed this line!
    decoder = Model(encoded_input, decoded)

    # define total model - input, three encoder layers, three decoder layers, output.
    model_input = Input(shape=(inputs,))
    encoder_output = encoder(model_input)
    decoder_output = decoder(encoder_output)
    network = Model(model_input, decoder_output)
    #define adaptive learning rate, number of epochs
    lr_start = 0.01
    lr_end = 0.001
    epochs = 5000
    decay = (lr_start - lr_end)/epochs
    #define loss function (mean absolute error (mae)) and optimisation fucntion Adam.
    adam = Adam(lr=lr_start, decay=decay)
    network.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])

    batch_size = 128

    # Fitting the data
    network.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=(X_valid, Y_valid),
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=100, verbose=0, mode='auto')])

    recon_train = network.predict(X_train)
    recon_valid = network.predict(X_valid)

    X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid = _correct_data(X_train, X_valid, Y_train, Y_valid,
                                                                                 recon_train, recon_valid, norms_X,
                                                                                 norms_Y)

    train_mae = np.average(np.absolute(Y_train - recon_train))
    train_mse = np.average(np.square(Y_train - recon_train))
    val_mae = np.average(np.absolute(Y_valid - recon_valid))
    val_mse = np.average(np.square(Y_valid - recon_valid))
    
    print(
        'LOOV next result - train_mae: {:.4f} train_mse: {:.4f} val_mae: {:.4f} val_mse: {:.4f}'.format(
            train_mae,
            train_mse,
            val_mae,
            val_mse))

    return X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, np.array(
        [train_mae, train_mse, val_mae, val_mse])


def train_mod_ autoencoders(X_train, X_valid, Y_train, Y_valid, norms_X, norms_Y):
    # Create and train autoencoder for first domain
    print("Training autoencoder on first domain")

    d1_network, d1_encoder, d1_decoder = autoencoder_model(X_train.shape[1])
    train_network(d1_network, X_train, X_valid, X_train, X_valid)

    # Create and train autoencoder for second domain
    print("Training autoencoder on second domain")
    d2_network, d2_encoder, d2_decoder = autoencoder_model(Y_train.shape[1])

    train_network(d2_network, Y_train, Y_valid, Y_train, Y_valid)

    # Combine autoencoders and re-train network
    print("Training combined network")
    network = merge_autoencoders(X_train.shape[1], d1_encoder, d2_decoder)
    train_network(network, X_train, X_valid, Y_train, Y_valid)

    recon_train = network.predict(X_train)
    recon_valid = network.predict(X_valid)

    X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid = _correct_data(X_train, X_valid, Y_train, Y_valid,
                                                                                 recon_train, recon_valid, norms_X,
                                                                                 norms_Y)

    train_mae = np.average(np.absolute(Y_train - recon_train))
    train_mse = np.average(np.square(Y_train - recon_train))
    val_mae = np.average(np.absolute(Y_valid - recon_valid))
    val_mse = np.average(np.square(Y_valid - recon_valid))

    return X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, np.array(
        [train_mae, train_mse, val_mae, val_mse])


# Train benchmarking machine learning methods (random regression forest, elastic net or kNN) and output predictions for the test data
# note results of elastic net is not included in the manuscript - while elastic net out preforms kNN it's predictions are not as good as the random regression forest.
def train_base_model(X_train, X_valid, Y_train, Y_valid, norms_X, norms_Y, model='rf'):
    if model == 'rf': #random regression forest
        predictor = RandomForestRegressor(max_features=0.3, n_estimators=200, n_jobs=3)
    elif model == 'elastic': #elastic net (note the results for the elastic net in not included in the manuscript- it did not prefer as well as the random regression forest)
        predictor = MultiTaskElasticNet(alpha=0.003, l1_ratio=0.7)
    elif model == 'knn': #k-nearest neighbours.
        predictor = KNeighborsRegressor(2, weights='distance')
    else:
        raise ValueError('{} is not a valid model!'.format(model))

    predictor.fit(X_train, Y_train)

    recon_train = predictor.predict(X_train)
    recon_valid = predictor.predict(X_valid)

    X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid = _correct_data(X_train, X_valid, Y_train, Y_valid,
                                                                                 recon_train, recon_valid, norms_X,
                                                                                 norms_Y)

    train_mae = np.average(np.absolute(Y_train - recon_train))
    train_mse = np.average(np.square(Y_train - recon_train))
    val_mae = np.average(np.absolute(Y_valid - recon_valid))
    val_mse = np.average(np.square(Y_valid - recon_valid))

    return X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, np.array(
        [train_mae, train_mse, val_mae, val_mse])



def leave_one_out_evaluation(X, Y, compounds, model='autoencoder', x_vivo_arg=False, y_vivo_arg=False):
    print("Performing leave-one-out evaluation with '{}' model".format(model))

    #train the specified model using leave-one-compound-out cross validation
    
    global x_vivo, y_vivo
    x_vivo = x_vivo_arg
    y_vivo = y_vivo_arg

    total_errors = []
    total_X_train = None
    total_Y_train = None
    total_X_valid = None
    total_Y_valid = None
    total_recon_train = None
    total_recon_valid = None
    total_train_compounds = None
    total_valid_compounds = None
    unique_compounds = np.unique(compounds)

    counter = 0
    for i, exclude_compound in enumerate(unique_compounds):
        counter = counter + 1
        print("Excluding compound", exclude_compound, "[", counter, "/", len(unique_compounds), "]")
        X_train, X_valid, Y_train, Y_valid, norms_X, norms_Y, train_compounds, \
        valid_compounds = split_train_test(X, Y, compounds, x_vivo, y_vivo, exclude_compound=exclude_compound)

        if model == 'mod_autoencoder':
            X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, errors = train_mod_autoencoders(X_train, X_valid,
                                                                                                      Y_train, Y_valid,
                                                                                                      norms_X, norms_Y)
        elif model == 'cnn':
            X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, errors = cnn_model(
                X_train, X_valid,
                Y_train, Y_valid,
                norms_X, norms_Y)

        elif model == 'naive_encoder':
            X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, errors = train_naive_encoder(
                X_train, X_valid,
                Y_train, Y_valid,
                norms_X, norms_Y)
        else:
            X_train, X_valid, Y_train, Y_valid, recon_train, recon_valid, errors = train_base_model(X_train, X_valid,
                                                                                                    Y_train, Y_valid,
                                                                                                    norms_X, norms_Y,
                                                                                                    model)

        if i == 0:
            total_X_train = X_train
            total_Y_train = Y_train
            total_X_valid = X_valid
            total_Y_valid = Y_valid
            total_recon_train = recon_train
            total_recon_valid = recon_valid
            total_train_compounds = train_compounds
            total_valid_compounds = valid_compounds
        else:
            total_X_train = np.append(total_X_train, X_train, axis=0)
            total_Y_train = np.append(total_Y_train, Y_train, axis=0)
            total_X_valid = np.append(total_X_valid, X_valid, axis=0)
            total_Y_valid = np.append(total_Y_valid, Y_valid, axis=0)
            total_recon_train = np.append(total_recon_train, recon_train, axis=0)
            total_recon_valid = np.append(total_recon_valid, recon_valid, axis=0)
            total_train_compounds = np.append(total_train_compounds, train_compounds, axis=0)
            total_valid_compounds = np.append(total_valid_compounds, valid_compounds, axis=0)

        total_errors.append(errors)

    total_errors = np.array(total_errors)
    avg_errors = np.mean(total_errors, axis=0)
    print("Average Errors:")
    print("Training mae:{}, mse:{}".format(avg_errors[0], avg_errors[1]))
    print("Validation mae:{}, mse:{}".format(avg_errors[2], avg_errors[3]))

    store_mae = avg_errors[2]  # Dan: I added this

    print("Compounds sorted by validation MAE (compound, MAE):")
    mae = total_errors[:, 2]
    for i in np.argsort(mae):
        print(unique_compounds[i], mae[i])

    data = {
        'X_train': total_X_train, 'Y_train': total_Y_train, 'recon_train': total_recon_train,
        'X_valid': total_X_valid, 'Y_valid': total_Y_valid, 'recon_valid': total_recon_valid,
        'train_compounds': total_train_compounds, 'valid_compounds': total_valid_compounds
    }
    return data, store_mae


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    compound_list = CompoundLists.GENERAL_45
    #define data domains note: X_type or input domain is always rat_vitro.
    x_type = "rat_vitro"
    y_type = "human_vitro"  #change target domain here as desired "rat_vivo" or "human_vitro"
    #load and parse data (extract time series of gene expression for the specfied compounds and genes in the specified domains)

   
    global x_vivo, y_vivo
    x_vivo = x_type == "rat_vivo"
    y_vivo = y_type == "rat_vivo"


    all_maes = [];  #for storing results

    for k in range(20,21):  # number of genes in set (20, 35, 50, 65, 80)
        for i in range(26,29):  #identifier of gene sets
            """
            Change location of input files below if desired:
            - change 'Random' to 'Orth' for orthologs
            - note that for size 20 there is (by definition) no 'Nested' folder!
            - change domain name if desired
            """
            #import gene set 
            file1 = "Data/RatInVitro/%d"%(k) + "/Nested/Random%d"%(k) + "/data_X%d"%(k) + "_%d"%(i) + "_nest.p"
            file2 = "Data/HumanInVitro/%d"%(k) + "/Nested/Random%d"%(k) + "/data_%d"%(k) + "_%d"%(i) + "_human_nest.p"
            print(file1)
            print(file2)

            X, _, gene_list_x, _ = pickle.load(open(file1, "rb"))
            Y, data_compounds, gene_list_y,_ =  pickle.load(open(file2, "rb"))

            """ Dan: select desired method under 'model'
                rf -> random regression forest
                mod_autoencoder -> modified autoender
                naive_encoder -> naive encoder model
                cnn -> convolutional neural network
            """
            #leave-one-compound-out cross validation for user specifed compounds and genes for the given model
            data, mae = leave_one_out_evaluation(X, Y, data_compounds, model='rf',x_vivo_arg=x_vivo, y_vivo_arg=y_vivo)
            all_maes.append(mae)

    print("Here are all MAEs again:\n", all_maes)

    """ Dan: this block can be uncommented to store the prediction values as a matlab workspace
    filename = "predictions.mat"
    print("filename", out_file_name)
    data['genes'] = gene_list_y
    data['genes_input'] = gene_list_x
    sio.savemat(out_file_name, data)
    """

if __name__ == '__main__':
    main()
