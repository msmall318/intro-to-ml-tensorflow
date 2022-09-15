import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import keras
from PIL import Image
# Ignore some warnings that are not relevant (you can remove this if you prefer)
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# Add a required, positional argument for the input data file name,
# and open in 'read' mode
# python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
parser.add_argument('--image', action='store', type=str, dest='img', default = 'test_images/cautleya_spicata.jpg', help = 'Provide path to the image file')
parser.add_argument('--mdl', action="store", type=str, dest="mdl", default = 'model.h5', help='Provide path to the model.h5 file')
parser.add_argument('--top-k', action="store", dest="top_k", type=int, default=3, help="Enter the number of features to display")
parser.add_argument('--category_names',action="store", dest="category_names",type = str, default=None, help="Provide path to the category names json file")

# Add an optional argument for the output file,
# open in 'write' mode and and specify encoding
parser.add_argument('--output', type=argparse.FileType('w', encoding='UTF-8'))

args = parser.parse_args()

#load the model
model = tf.keras.models.load_model('my_model.h5',  custom_objects={'KerasLayer':hub.KerasLayer})

# Done: Create the process_image function
def process_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 225
    return image

if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

# process image and make prediction
def predict(image_path, model, top_k):
    processed_test_image = process_image(image_path)
    image_batch = np.expand_dims(processed_test_image, axis=0)
    probs = model.predict(image_batch)
    index = [str(i) for i in range(len(probs[0]))]
    temp = sorted(zip(probs[0], index), reverse=True)[:top_k]
    probs = [i[0] for i in temp]
    classes = [i[1] for i in temp]
    if args.category_names:
        names = [class_names[i[1]] for i in temp]
        return probs, classes, names
    else:
        return probs, classes

if args.category_names:
    probs, classes, names = predict(args.img, model, args.top_k)
    print(probs)
    print(classes)
    print(names)
else:
    probs, classes = predict(args.img, model, args.top_k)
    print(probs)
    print(classes)