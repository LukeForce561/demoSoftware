import  numpy as np
import  keras
from    keras                       import backend as K
from    keras.layers.core           import Dense
from    keras.optimizers            import Adam 
from    keras.metrics               import categorical_crossentropy
from    keras.preprocessing.image   import ImageDataGenerator
from    keras.preprocessing         import image
from    keras.models                import Model
from    keras.applications          import imagenet_utils
from    sklearn.metrics             import confusion_matrix
import  intertools
import  matplotlib.pyplot           as plt
#matplotlib inline


mobile= keras.applications.mobilenet.MobileNet()

def prepare_image(file)
    image_path              = 'MobileNet-inference-images/'
    img - image.load_img(img_path + file, target_size=(224,224))
    img_array               = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
