import cv2
import onnx
import onnxruntime as rt
import onnxruntime as rt
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import time
from src.config import *
from src.postprocessing import *
from src.preprocessing import *

import numpy as np
from PIL import Image

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

image = Image.open("/home/sri/Documents/RTR2LP34edit.jpg")
# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
print(image_size.shape)
print(image_data[0].shape)
print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed input
sess = rt.InferenceSession("/home/sri/Documents/tiny-yolov3-11.onnx")
final=np.array([1,2],dtype='float32')
print('final:',image_size)
outputs = sess.get_outputs()
print('output name:',outputs)
output_names = list(map(lambda output: output.name, outputs))
input_name = sess.get_inputs()
print('Input name:',input_name)
start = timer()
out_boxes, out_scores, out_classes = sess.run(output_names,{input_name[0].name:image_data,input_name[1].name:image_size})
print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
print(type(out_classes))
print(out_classes.shape)
print(out_classes[0])
print(out_boxes.shape,out_scores.shape,out_classes.shape)
for i, c in reversed(list(enumerate(out_classes))):
    print(c)

# thickness = (image.size[0] + image.size[1]) // 300
# for i, c in reversed(list(enumerate(out_classes))):
#     predicted_class=1
#     box = out_boxes[0][i]
#     print(box.shape)
#     score = out_scores[i]
#     #label = '{} {:.2f}'.format(predicted_class, score)
#     draw = ImageDraw.Draw(image)
#     #label_size = draw.textsize(label)
#
#     top, left, bottom, right = box
#     top = max(0, np.floor(top + 0.5).astype('int32'))
#     left = max(0, np.floor(left + 0.5).astype('int32'))
#     bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#     right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#     print(f'top:{top},left:{left},right:{right},bottom:{bottom}')
#     # print(label, (left, top), (right, bottom))
#     #
#     # if top - label_size[1] >= 0:
#     #     text_origin = np.array([left, top - label_size[1]])
#     # else:
#     #     text_origin = np.array([left, top + 1])
#     #
#     # # My kingdom for a good redistributable image drawing library.
#     for i in range(thickness):
#         draw.rectangle(
#             [left + i, top + i, right - i, bottom - i],
#             outline=(0, 255, 0))
#     # draw.rectangle(
#     #     [tuple(text_origin), tuple(text_origin + label_size)],
#     #     fill=self.colors[c])
#     # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
#     #del draw
#     print(type(image))
#     draw.save('out.jpg')
#
#
#
#
# end = timer()
# print(end - start)


