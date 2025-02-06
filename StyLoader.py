import sys
import os
import cv2
import torch
import numpy
basePathEZ = 'E:\StyleByExample\Stylizing-Video-by-Example'
sys.path.append(basePathEZ)
sys.path.append(basePathEZ+'\core')
from main_called import ezscalled
from lib import utils

import imageio

def synthesizeImage(targetImage,conten0,conten1,index =0):
    inputDir = "./temp/input"
    os.makedirs(inputDir, exist_ok=True)
    edgeDir = "./temp/input/edge"
    os.makedirs(edgeDir, exist_ok=True)
    styDir = "./temp/styDir"
    os.makedirs(styDir, exist_ok=True)
    outputDir = "./temp/Output"
    os.makedirs(outputDir, exist_ok=True)
    
    
    rgb = conten0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb8 = utils.to8b(rgb)
    filename = (inputDir+ '/' + '{:03d}.png'.format(0))
    print(filename)
    imageio.imwrite(filename, rgb8)
 
    rgb = conten1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb8 = utils.to8b(rgb)
    filename = (inputDir+ '/' + '{:03d}.png'.format(1))
    print(filename)
    imageio.imwrite(filename, rgb8)
    
    rgb = targetImage.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb8 = utils.to8b(rgb)
    filename = (styDir+ '/' + '{:03d}.png'.format(0))
    print(filename)
    imageio.imwrite(filename, rgb8)

    
    output = ezscalled(styDir+'/000.png',inputDir,index=index)

    
    
    imgs = cv2.imread(output) 
    print(imgs.shape)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)/256.0 #,dtype=torch.float32
    

    
    noGradRGBTarget = torch.tensor(imgs,dtype=torch.float32 )
    noGradRGBTarget = noGradRGBTarget.permute(2,0,1).unsqueeze(0)

    return noGradRGBTarget
    
    
    
    