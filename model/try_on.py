# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from dataloaders.cp_dataset import CPDataset, CPDataLoader
from models.networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from models.visualization import board_add_image, board_add_images, save_images
import numpy as np

from PIL import Image
import timeit

global test_loader_GMM
global model_GMM
global test_loader_TOM
global model_TOM

def test_gmm(test_loader, model):
    model.cuda()
    model.eval()

    
    
    save_dir = os.path.join('data', 'input')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)

    warp_mask_dir = os.path.join(save_dir, 'warp-mask')    
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    '''
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)
    '''

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        #c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        #im_pose = inputs['pose_image'].cuda()
        #im_h = inputs['head'].cuda()
        #shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        #im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        #shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        #warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        #overlay = 0.7 * warped_cloth + 0.3 * im
        '''
        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        '''
        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)

        '''
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)
        '''
        if (step+1) % 1 == 0:
            #board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('GMM : step: %8d, time: %.3f\n' % (step+1, t))#, flush=True)


def test_tom(test_loader, model):
    model.cuda()
    model.eval()

    
    
    save_dir = os.path.join('data', 'output')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    '''    
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data
    '''


    print('Dataset size: %05d!\n' % (len(test_loader.dataset)))#, flush=True)


    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        #im = inputs['image'].cuda()
        #im_pose = inputs['pose_image']
        #im_h = inputs['head']
        #shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        
        save_images(p_tryon, im_names, try_on_dir)
        '''
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data
        '''
        if (step+1) % 1 == 0:
            
            t = time.time() - iter_start_time
            print('TOM : step: %8d, time: %.3f\n' % (step+1, t))#, flush=True)


def load_preparse(file):
    im_parse = Image.open(file).convert('L')   # updated new segmentation  
    parse_array = np.array(im_parse)
    return parse_array

def pre_load():
    global test_loader_GMM
    global model_GMM
    global test_loader_TOM
    global model_TOM

    

    # gmm loader
    test_dataset_GMM = CPDataset('GMM')    
    test_loader_GMM = CPDataLoader(test_dataset_GMM)
    # gmm model
    model_GMM = GMM()
    load_checkpoint(model_GMM, 'data/pretrained/gmm_final.pth')    
    
    # tom loader
    test_dataset_TOM = CPDataset('TOM')
    test_loader_TOM = CPDataLoader(test_dataset_TOM)    
    # tom model
    model_TOM = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
    load_checkpoint(model_TOM, 'data/pretrained/tom_final.pth')    
    
    print("PREPARED")


def fit():
    global test_loader_GMM
    global model_GMM
    global test_loader_TOM
    global model_TOM
   

    start_time = timeit.default_timer()
    RUN(test_loader_GMM, model_GMM, test_loader_TOM, model_TOM)
    end_time = timeit.default_timer()
    print('total time : '+ str(end_time - start_time))


def RUN(test_loader_GMM, model_GMM, test_loader_TOM, model_TOM):
    print("RUN")
    with torch.no_grad():
        test_gmm( test_loader_GMM, model_GMM )
    print('Finished test %s, named: %s!' % ('GMM', 'GMM'))    
    
    with torch.no_grad():
        test_tom( test_loader_TOM, model_TOM )

    print('Finished test %s, named: %s!' % ('TOM', 'TOM'))

   
def main():

    pre_load()
    while(1):
        a = input()
        if a == 'r' or a=='f':
            fit()
        if a=='q' or a=='c':
            break


if __name__ == "__main__":
    main()


# python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/GMM/gmm_final.pth

# python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/TOM/tom_final.pth