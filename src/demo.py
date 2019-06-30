from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import json         #added json import to save the output of the network into a .json file

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

def is_image(file_name):
  ext = file_name[file_name.rfind('.') + 1:].lower()
  return ext in image_ext

def get_filename(file_name):
    name = file_name[0: + file_name.rfind('.')]
    return name

def demo_image(image, model, opt):      #image name added as an input, so that the individual output files can be asctibed to the respective image
  s = max(image.shape[0], image.shape[1]) * 1.0
  c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
  trans_input = get_affine_transform(
      c, s, 0, [opt.input_w, opt.input_h])
  inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
  inp = (inp / 255. - mean) / std
  inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
  inp = torch.from_numpy(inp).to(opt.device)
  out = model(inp)[-1]
  pred = get_preds(out['hm'].detach().cpu().numpy())[0]
  pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))  #this step readjusts the 2D skeleton to the input image with center and scale
  pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]    #uses heatmap and depthmap to create 3D pose
  #pred_3d = transform_preds(pred_3d, c, s, (opt.output_w, opt.output_h,10))  #this step readjusts the 3D skeleton to the input image with center and scale
  
  '''
  debugger = Debugger()
  debugger.add_img(image)                   #adds image
  debugger.add_point_2d(pred, (255, 0, 0))  #adds 2D graph of joints to image
  debugger.add_point_3d(pred_3d, 'b')       #plots the 3D joint locations into the plot
  debugger.show_all_imgs(pause=False)       #this command enables showing the images
  debugger.show_3d()                        #this command shows the figures
  '''

  return pred_3d.flatten()

def main(opt):
  opt.heads['depth'] = opt.num_output
  if opt.load_model == '':
    opt.load_model = '../models/fusion_3d_var.pth'
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  else:
    opt.device = torch.device('cpu')
  
  model, _, _ = create_model(opt)
  model = model.to(opt.device)
  model.eval()
  
  path_dec = opt.demo.split("/")

  if os.path.isdir(opt.demo):
    ls = os.listdir(opt.demo)
    result_array = np.empty((0,48))
    for file_name in sorted(ls):
      if is_image(file_name):
        image_name = os.path.join(opt.demo, file_name)
        #print('Running {} ...'.format(image_name))
        image = cv2.imread(image_name)
        pred = demo_image(image, model, opt)                             #shows images and their 3D models
        result_array = np.append(result_array, [pred], axis=0)

    #result_array = result_array-avgarr
    print(os.path.join('coords_pavlakos', path_dec[-1] + '.npy'))
    np.save(os.path.join('coords_pavlakos', path_dec[-1] + '.npy'), result_array)    #saves 3D coordinates of the 16 joints that are extracted from the image
  elif is_image(opt.demo):
    print('Running {} ...'.format(opt.demo))
    image = cv2.imread(opt.demo)
    demo_image(image, model, opt, file_name)

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
