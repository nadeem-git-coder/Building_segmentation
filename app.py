# This was made by following this tutorial
# https://www.youtube.com/watch?v=i40ulpcacFM

import os
from os.path import join as pjoin
import cv2
import numpy as np
from tqdm import tqdm

from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from patchify import patchify, unpatchify

from keras import backend as K
from keras.models import load_model 
     
import segmentation_models as sm

import gradio as gr
     
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  

model_path = 'models/satellite_segmentation_100-epochs.h5'
saved_model = load_model(model_path,
                         custom_objects=({'dice_loss_plus_1focal_loss': total_loss, 
                                          'jaccard_coef': jaccard_coef}))


def process_input_image(test_image):
  test_dataset = []
  image_patch_size = 256
  scaler = MinMaxScaler()

  # crop images so that they are divisible by image_patch_size
  test_image = np.array(test_image)
  size_x = (test_image.shape[1]//image_patch_size)*image_patch_size
  size_y = (test_image.shape[0]//image_patch_size)*image_patch_size

  test_image = Image.fromarray(test_image)
  test_image = test_image.crop((0, 0, size_x, size_y))
            
  # patchify image so that each patch is size (image_patch_size,image_patch_size)
  test_image = np.array(test_image)
  image_patches = patchify(test_image, (image_patch_size,image_patch_size, 3), step = image_patch_size) # 3 should probably be a variable since we have have  many more channels than RGB

  # scale values so that they are between 0 to 1
  # here, we use MinMaxScaler from sklearn

  for i in range(image_patches.shape[0]):
    for j in range(image_patches.shape[1]):
      image_patch = image_patches[i,j,:,:]

      image_patch = scaler.fit_transform(image_patch.reshape(-1, image_patch.shape[-1])).reshape(image_patch.shape)
      
      image_patch = image_patch[0] # drop extra unessesary dimantion that patchify adds
      test_dataset.append(image_patch)

  test_dataset = [np.expand_dims(np.array(x), 0) for x in test_dataset]

  test_prediction = []
      
  for image in tqdm(test_dataset):
    prediction = saved_model.predict(image,verbose=0)
    predicted_image = np.argmax(prediction, axis=3)
    predicted_image = predicted_image[0,:,:]
    test_prediction.append(predicted_image)


  reconstructed_image = np.reshape(np.array(test_prediction),(image_patches.shape[0],image_patches.shape[1],image_patch_size,image_patch_size))
  reconstructed_image  =  unpatchify(reconstructed_image , (size_y,size_x))

  lookup = {'rgb': [np.array([ 60,  16, 152]),
    np.array([132,  41, 246]),
    np.array([110, 193, 228]),
    np.array([254, 221,  58]),
    np.array([226, 169,  41]),
    np.array([155, 155, 155])],
  'int': [0, 1, 2, 3, 4, 5]}

  rgb_image = np.zeros((reconstructed_image.shape[0],reconstructed_image.shape[1],3), dtype=np.uint8)

  for i,l in enumerate(lookup['int']):
    rgb_image[np.where(reconstructed_image==l)] = lookup['rgb'][i]
  return 'Predicted Masked Image', rgb_image


my_app = gr.Blocks()
with my_app:
  gr.Markdown("Statellite Image Segmentation Application UI with Gradio")
  with gr.Tabs():
    with gr.TabItem("Select your image"):
      with gr.Row():
        with gr.Column():
            img_source = gr.Image(label="Please select source Image")
            source_image_loader = gr.Button("Load above Image")
        with gr.Column():
            output_label = gr.Label(label="Image Info")
            img_output = gr.Image(label="Image Output")
    source_image_loader.click(
        process_input_image,
        [
            img_source
        ],
        [
            output_label,
            img_output
        ]
    )

my_app.launch(debug=True)