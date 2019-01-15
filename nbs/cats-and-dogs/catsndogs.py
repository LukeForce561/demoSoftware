import  numpy as np
import  Keras
import  IPython
from    IPython                     import get_ipython
from    Keras                       import backend as K
from    Keras.layers                import Activation
from    Keras.layers.core           import Dense, Flatten
from    Keras.layers.normalization  import BatchNormalization
from    Keras.layers.convolution    import *
from    Keras.optimizers            import Adam 
from    Keras.metrics               import categorical_crossentropy
from    Keras.preprocessing.image   import ImageDataGenerator
#from    Keras.preprocessing         import image
from    Keras.models                import Sequential
#from    Keras.applications          import imagenet_utils
from    sklearn.metrics             import confusion_matrix
import  intertools
import  matplotlib.pyplot           as plt

get_ipython().run_line_magic('matplotlib', 'inline')

train_path      =    'cats-and-dogs/train' 
valid_path      =    'cats-and-dogs/valid' 
test_path       =    'cats-and-dogs/test'

train_batches   =   ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['dog','cat'], batch_size=10)
valid_batches   =   ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['dog','cat'], batch_size=4)
test_batches    =   ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['dog','cat'], batch_size=10)

#plots images with labels within jupyter notebook

def plots(ims, figsize=(12,6), rows=1, interp=False,titles=None):
            if type(ims[0]) is np.ndarray:
                ims = np.array(ims).astype(np.uint8)
                if(ims.shape[-1] !=3 ):
                    ims = ims.transpose((0,2,3,1))
            f = plt.figure(figsize=figsize)
            cols = len(ims)// rows if len(ims) % 2 == 0 else len(ims) // rows + 1
            for i in range(len(ims)):
                sp = f.add_subplot(rows,cols,i+1)
                sp.axis('Off')
                if titles is not None:
                    sp.set_title(titles[i],fontsize=16)
                plt.imshow(ims[i],interpolatgion=None if interp else 'none')

imgs, labels = nex(train_batches)
plots(imgs,titles=labels)

## Mobilenet image preparation

## def prepare_image(file)
    ## image_path              = 'MobileNet-inference-images/'
    ## img - image.load_img(img_path + file, target_size=(224,224))
    ## img_array               = image.img_to_array(img)
    ## img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    ## return Keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
