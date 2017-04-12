from keras.models import load_model
from keras.utils import plot_model
model = load_model('./model.hp')

model.summary()
plot_model(model, to_file='imgs/model.png',show_shapes=True,show_layer_names=False)