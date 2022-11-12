from keras import models, layers, losses, optimizers
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data_path = 'co2_efficiencies.csv'
data_ind = 'CO2_Efficiency'
x_ind = 'Year'
raw_data = pd.read_csv(data_path)
train_steps = 10
pred_steps = 100
lstm_cells = 100
learn_rate = 1e-3
train_epochs = 30
save_model_as = 'co2_model'
save_data_as = 'co2_eff_pred.csv'


processed_data = raw_data[data_ind]

max_val = max(processed_data)
min_val = min(processed_data)

processed_data = (processed_data - min_val) / (max_val - min_val)

training_data = []
for i in range(len(processed_data) - train_steps):
    input_set = []
    for j in range(train_steps):
        input_set.append(processed_data[i + j])
    training_data.append(np.array(input_set))
training_data = np.array(training_data)


timesteps = train_steps
feature = 1
model = models.Sequential([
    layers.LSTM(lstm_cells, input_shape=(timesteps, feature), return_sequences=False),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.rmsprop_v2.RMSProp(learning_rate=learn_rate), loss=losses.mean_squared_error)
model.fit(training_data, np.array(processed_data[train_steps:]), epochs=train_epochs)


preds = list(training_data)
data_list = list(processed_data).copy()
for i in range(pred_steps):
    res = model.predict(np.array([preds[-1]]))
    data_list.append(float(res[0][0]))
    input_set = []
    for j in range(-train_steps-1, -1):
        input_set.append(data_list[j])
    preds.append(np.array(input_set))


y_pred = model.predict(training_data) * (max_val - min_val) + min_val
y_true = raw_data[data_ind][train_steps:]
data_list = np.array(data_list) * (max_val - min_val) + min_val

plt.plot(list(raw_data[x_ind].iloc[train_steps:]), y_true, 'r')
plt.plot(list(raw_data[x_ind].iloc[train_steps:]), y_pred, 'b')
plt.plot(range(list(raw_data[x_ind])[-1], list(raw_data[x_ind])[-1]+pred_steps), data_list[len(y_true)+train_steps:], 'g')
plt.show()

if save_data_as:
    c = list(y_pred)
    c.extend(data_list[len(y_true)+train_steps:])
    d = range(list(raw_data[x_ind])[-1], list(raw_data[x_ind])[-1]+pred_steps)
    with open(save_data_as, mode='w+') as f:
        f.write('Year,'+data_ind+'\n')
        for i in range(len(d)):
            f.write(str(d[i])+','+str(c[i])+'\n')


if save_model_as:
    model.save(save_model_as, overwrite=False)
