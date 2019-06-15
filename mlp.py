from ann_visualizer.visualize import ann_viz
from keras import Sequential
from keras.layers import Dense, Dropout

from load_data import x, y, dimensions
from mccv_keras import mccv

hidden_layers = 5
plot = False  # Switch to True to output the architecture as a .png file.
name = ""  # If the above is set to True, this will be the name of the output file.
title = ""  # If the above is set to True, this will be the title of the graph.

model = Sequential()
model.add(Dense(16, input_dim=dimensions, dtype="float32", activation="relu"))
for i in range(hidden_layers - 1):
    model.add(Dense(16, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

if plot:
    ann_viz(model, filename="{}.gv".format(name), title=title)

mccv(x, y, model)
