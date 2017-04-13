from keras.models import load_model
from keras.utils import plot_model

model = load_model('./model.hp')

model.summary()
plot_model(model, to_file='imgs/model.png',show_shapes=True,show_layer_names=False)

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# svg_img = model_to_dot(model).create(prog='dot', format='svg')
# SVG(svg_img)