import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import gc


nb_epoch = 100
batch_size = 32
len_train = 25903891
data_path = '../data/'
model_save = "autoencoder_model_1layer.h5"
log_dir = "./autoencoder-logs1layer"
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

print("Loading Data")
# train_df = pd.read_csv('train_df.csv')

train_df = pd.read_csv(data_path + "train_sample.csv.zip", compression='infer',
                       dtype=dtypes,
                       usecols=['ip', 'app', 'device', 'os', 'channel',
                                'click_time', 'is_attributed'])


print(train_df.is_attributed.value_counts())

# Print stats we might need
print("Stats:")
print(train_df.head())
print("Shape:", train_df.shape)

# Remove unwanted columns
train_df['hour'] = pd.to_datetime(
    train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(
    train_df.click_time).dt.day.astype('uint8')
train_df['wday'] = pd.to_datetime(
    train_df.click_time).dt.dayofweek.astype('uint8')
print('grouping by ip-day-hour combination....')
gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(
    by=['ip', 'day', 'hour'])[
    ['channel']].count().reset_index().rename(index=str,
                                              columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
del gp
gc.collect()
print('group by ip-app combination....')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])
[['channel']].count().reset_index().rename(index=str,
                                           columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
del gp
gc.collect()
print('group by ip-app-os combination....')
gp = train_df[['ip', 'app', 'os', 'channel']].groupby(
    by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(
    index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
del gp
gc.collect()
print("vars and data type....")
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
train_df.drop(['click_time', 'ip'], 1, inplace=True)

train_df[['app', 'device', 'os', 'channel', 'hour', 'day', 'wday']].apply(
    LabelEncoder().fit_transform)

print(train_df.head())

# Encode categorical labels using target encoding
enc = ce.TargetEncoder(cols=['app', 'device', 'os', 'channel',
                             'hour', 'day', 'wday']).fit(
    train_df, train_df.is_attributed)

train_df = enc.transform(train_df)
print(train_df.head())
# Split in 75% train and 25% test set
X_train, X_test = train_test_split(train_df, test_size=0.25,
                                   random_state=1984)

# Get the data ready for the Neural Network
X_train = X_train[X_train.is_attributed == 0]
X_train = X_train.drop(['is_attributed'], axis=1)

y_test = X_test['is_attributed']
X_test = X_test.drop(['is_attributed'], axis=1)

X_train = X_train.values
X_test = X_test.values

# starting number of neurons
input_dim = X_train.shape[1]
print(X_train.shape, X_test.shape)

# Shape based on unique data columns, e.g. channel, qty, ip_app_count
input_layer = Input(shape=(input_dim, ))
encoding_dim = 7
# Build Layers: encoders passed to decoders
encoder = Dense(int(7), activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(5), activation="tanh")(encoder)
decoder = Dense(int(7), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='tanh')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# if not os.path.isfile(model_save):

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
autoencoder.summary()
# Save model for later
checkpointer = ModelCheckpoint(filepath=model_save,
                               verbose=0,
                               save_best_only=True)
# Save Tensorboard data
tensorboard = TensorBoard(log_dir=log_dir,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# Train model
history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer, tensorboard]).history


print("Loading Model")
autoencoder = load_model(model_save)
predictions = autoencoder.predict(X_test)
print("Making predictions")
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                         'true_class': y_test})
print(error_df.describe())
# del train_df, y_train
# gc.collect()
