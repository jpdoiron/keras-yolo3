import numpy as np
import time as timer
from yolo3.utils import get_random_data
import matplotlib.pyplot as plt

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


annotation_path = 'Custom_set.txt'
classes_path = 'model_data/custom_classes.txt'
anchors_path = 'model_data/tiny_yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw

val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
batch_size = 32

#gen = train.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
start = timer.time()

a = get_random_data(lines[:num_train][0],input_shape,False,max_boxes=1)

end  = timer.time()
print(end-start)

plt.imshow(a[0])
plt.show()
