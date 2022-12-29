# coding=utf-8
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import cv2

#parse
from dataloaders import cihp
from dataloaders import custom_transforms as tr
from models import deeplab_xception_transfer, graph
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import timeit

#mask
from models.masking import masking_img

#pose
from models.pose_body import Body
from models import util

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def parse_generator_dataloader():
    ## multi scale
    scale_list=[1,0.5,0.75,1.25,1.5,1.75]
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_(pv),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_(pv),
            tr.HorizontalFlip(),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        voc_val = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts)
        voc_val_f = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts_flip)

        testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=0)
        testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=0)

        testloader_list.append(copy.deepcopy(testloader))
        testloader_flip_list.append(copy.deepcopy(testloader_flip))
    
    return testloader_list, testloader_flip_list


def preparing_img_parsing():
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()


    testloader_list, testloader_flip_list = parse_generator_dataloader()


    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20, os=16,
                                                                                     hidden_layers=128, source_classes=7,
                                                                                     )
    net.cuda()
    x = torch.load('data/pretrained/inference.pth')
    net.load_source_model(x)
    adj = [adj1_test, adj2_test, adj3_test]
    return net, testloader_list, testloader_flip_list, adj

   


class CPDataset(data.Dataset):
    """Dataset for CP-VTON+.
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting        
        self.dataroot = "data"
        self.datamode = "input"  # train or test or self-defined
        self.stage = opt  # GMM or TOM
        self.data_list = "test_pairs.txt"
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        self.data_path = osp.join(self.dataroot, self.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])

        self.pose_extractor = Body('data/pretrained/body_pose_model.pth')
        
        
        self.parsing_net, self.parsing_loaders, self.parsing_flip_loaders, self.adj = preparing_img_parsing()
        

        self.im_names = ""
        self.c_names = ""
        self.load_data_list()
        # load data list   

    def load_data_list(self):
        im_names = []
        c_names = []
        with open(osp.join(self.data_path, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

            
    def name(self):
        return "CPDataset"


    def img_parsing(self, net, testloader_list, testloader_flip_list, adj):
        
        start_time = timeit.default_timer()
        net.eval()
        adj1_test, adj2_test, adj3_test = adj[0], adj[1], adj[2]
        results = []
        for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
            print(ii)
            #1 0.5 0.75 1.25 1.5 1.75 ; flip:
            sample1 = large_sample_batched[:6]
            sample2 = large_sample_batched[6:]
            for iii,sample_batched in enumerate(zip(sample1,sample2)):
                inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
                inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
                inputs = torch.cat((inputs,inputs_f),dim=0)
                if iii == 0:
                    _,_,h,w = inputs.size()
                # assert inputs.size() == inputs_f.size()

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

                with torch.no_grad():
                    
                    inputs, labels = inputs.cuda(), labels.cuda()
                    # outputs = net.forward(inputs)                
                    outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                    outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                    outputs = outputs.unsqueeze(0)

                    if iii>0:
                        outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
                        outputs_final = outputs_final + outputs
                    else:
                        outputs_final = outputs.clone()
            ################ plot pic
            
            predictions = torch.max(outputs_final, 1)[1]
            results.append(predictions.cpu().numpy())
            #cv2.imwrite(opts.output_path + 'cihp_output/{}.png'.format(img_list[ii]), results[0,:,:])
            
        end_time = timeit.default_timer()
        print('Parsing time use for '+str(ii) + ' is :' + str(end_time - start_time))
        

        
        return results[0]

    def img_preprocessing(self, human_img, cloth_img, pose_model, im_name):

        start_time = timeit.default_timer()
        
        #mask
        
        cloth_mask = masking_img(np.array(cloth_img.convert('L')))

        #pose
        candidate, subset = pose_model(np.array(human_img))
        cand = np.delete(candidate, [3],axis=1)
        for x in np.where(subset == -1)[1] :
            cand = np.insert(cand,[x],[0,0,0],axis=0)
        dict_form = {"version": 1.0, "people": [{"face_keypoints": [], "pose_keypoints": [], "hand_right_keypoints": [], "hand_left_keypoints": []}]} 
        dict_form['people'][0]['pose_keypoints'] = cand.reshape(-1).tolist()         


        with open('data/output/debug/' + im_name.replace('.jpg', '_keypoints.json'), "w") as json_file:
            json.dump(dict_form, json_file)
        human_img = cv2.cvtColor(np.array(human_img), cv2.COLOR_BGR2RGB)

        canvas = copy.deepcopy(human_img)
        canvas = util.draw_bodypose(canvas, candidate, subset)


        cv2.imwrite('data/output/debug/pose_'+im_name.replace('.jpg', '.png'), canvas)
        
        end_time = timeit.default_timer()
        print('masking, pose time : '+ str(end_time - start_time))
        return cloth_mask, dict_form


    def __getitem__(self, index):

        self.load_data_list()
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # person image
        im = Image.open(osp.join(self.data_path, im_name))
        im = im.resize((self.fine_width, self.fine_height), Image.LANCZOS)

        print("mask, pose...")
        if self.stage == 'GMM':
            
            c = Image.open(osp.join(self.data_path, c_name))            
            c = c.resize((self.fine_width, self.fine_height),Image.LANCZOS)
            cloth_mask, pose = self.img_preprocessing(im, c, self.pose_extractor, im_name) 
            cm = Image.fromarray(cloth_mask).convert('L')
            #cm = Image.open(osp.join(self.data_path, 'cloth-mask' + c_name)).convert('L')
            
            cv2.imwrite("data/output/debug/mask-"+c_name.replace('.jpg', '.png'), cloth_mask)
        else:

            c = Image.open(osp.join(self.data_path, 'warp-cloth', im_name))    # c_name, if that is used when saved
            cloth_mask, pose = self.img_preprocessing(im, c, self.pose_extractor, im_name) 
            cm = Image.open(osp.join(self.data_path, 'warp-mask', im_name)).convert('L')    # c_name, if that is used when saved
        
        print("done!")
        im = self.transform(im)  # [-1,1]

        
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)

        
        
        """
        LIP labels
        
        [(0, 0, 0),    # 0=Background
         (128, 0, 0),  # 1=Hat
         (255, 0, 0),  # 2=Hair
         (0, 85, 0),   # 3=Glove
         (170, 0, 51),  # 4=SunGlasses
         (255, 85, 0),  # 5=UpperClothes
         (0, 0, 85),     # 6=Dress
         (0, 119, 221),  # 7=Coat
         (85, 85, 0),    # 8=Socks
         (0, 85, 85),    # 9=Pants
         (85, 51, 0),    # 10=Jumpsuits
         (52, 86, 128),  # 11=Scarf
         (0, 128, 0),    # 12=Skirt
         (0, 0, 255),    # 13=Face
         (51, 170, 221),  # 14=LeftArm
         (0, 255, 255),   # 15=RightArm
         (85, 255, 170),  # 16=LeftLeg
         (170, 255, 85),  # 17=RightLeg
         (255, 255, 0),   # 18=LeftShoe
         (255, 170, 0)    # 19=RightShoe
         (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
         ]
         """

        print('parse...')
        # load parsing image
        
        if os.path.isfile('data/output/debug/parse-'+im_name.replace('.jpg','.png')) :
            im_parse = Image.open('data/output/debug/parse-'+im_name.replace('.jpg','.png')).convert('L')   # updated new segmentation  
            parse_array = np.array(im_parse)
        else :
            im_parse = self.img_parsing(self.parsing_net, self.parsing_loaders, self.parsing_flip_loaders, self.adj)                        
            parse_array = im_parse.reshape(256,192)
            cv2.imwrite('data/output/debug/parse-'+im_name.replace('.jpg','.png'), parse_array)
            self.parse = parse_array
        
            
            
        print('done!')

        
        '''
        parse_name = im_name.replace('.jpg', '.png')        
        im_parse = Image.open(
            # osp.join(self.data_path, 'image-parse', parse_name)).convert('L')
            osp.join(self.data_path, 'image-parse-new'+ parse_name)).convert('L')   # updated new segmentation
        
        parse_array = np.array(im_parse)
        '''
        
        #im_mask = Image.open(osp.join(self.data_path, 'image-mask'+ parse_name)).convert('L')
        #mask_array = np.array(im_mask)
        
        # parse_shape = (parse_array > 0).astype(np.float32)  # CP-VTON body shape
        # Get shape from body mask (CP-VTON+)
        


        parse_shape = (parse_array > 0).astype(np.float32)
        image_mask = (parse_shape * 255).astype(np.uint8)
        cv2.imwrite("data/output/debug/mask-"+im_name.replace('.jpg', '.png'), image_mask)
        if self.stage == 'GMM':
            parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(
                    np.float32)  # CP-VTON+ GMM input (reserved regions)
        else:
            parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 16).astype(np.float32) + \
                (parse_array == 17).astype(
                np.float32)  # CP-VTON+ TOM input (reserved regions)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32)    # upper-clothes labels

        # shape downsample
        parse_shape_ori = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize(
            (self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
        parse_shape_ori = parse_shape_ori.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
        
        try:
            shape_ori = self.transform(parse_shape_ori)  # [-1,1]
        except:
            shape_ori = self.transform2(parse_shape_ori)  # [-1,1]

        try:
            shape = self.transform(parse_shape)  # [-1,1]
        except:
            shape = self.transform2(parse_shape)  # [-1,1]

        phead = torch.from_numpy(parse_head)  # [0,1]
        # phand = torch.from_numpy(parse_hand)  # [0,1]
        pcm = torch.from_numpy(parse_cloth)  # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm)  # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead)  # [-1,1], fill -1 for other parts

        # load pose points
        
        '''
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, '_pose'+ pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        '''

        
        pose_data = pose['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))
        
    


        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')

            try:
                one_map = self.transform(one_map)
            except:
                one_map = self.transform2(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        try:
            im_pose = self.transform(im_pose)
        except:
            im_pose = self.transform2(im_pose)

        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0)

        if self.stage == 'GMM':
            
            im_g = Image.open(osp.join(self.data_path, osp.join('default','grid.png')))
            im_g = self.transform(im_g)
        else:
            im_g = ''

        pcm.unsqueeze_(0)  # CP-VTON+

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            'parse_cloth_mask': pcm,     # for CP-VTON+, TOM input
            'shape_ori': shape_ori,     # original body shape without resize
        }
        
        
        return result

    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, dataset):
        super(CPDataLoader, self).__init__()

        
        train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
    def __parse__(self):
        import pdb; pdb.set_trace()

    
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed
    embed()
