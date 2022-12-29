from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_cihp import Path
import random

class VOCSegmentation(Dataset):
    """
    CIHP dataset
    """

    def __init__(self,
                 base_dir='./data/input',
                 split='train',
                 transform=None,
                 flip=False,
                 ):
        """
        :param base_dir: path to CIHP dataset directory
        :param split: train/val/test
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._flip_flag = flip

        self._base_dir = base_dir
        self._cat_dir = os.path.join(self._base_dir, 'Category_ids')
        self._flip_dir = os.path.join(self._base_dir,'Category_rev_ids')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform
        

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []

        self.load_txt()
        

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def load_txt(self) :

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []

        with open(os.path.join(os.path.join(self._base_dir, 'test_pairs.txt')), "r") as f:
                lines = f.read().splitlines()
            
        for ii, line in enumerate(lines):
            
            
            _image = os.path.join(self._base_dir, line.split(' ')[0])            
            _cat = os.path.join(self._base_dir, 'default/parse.png')
            assert os.path.isfile(_image)
            self.im_ids.append(line.split(' ')[0])
            self.images.append(_image)
            self.categories.append(_cat)
            self.flip_categories.append(_cat)

        assert (len(self.images) == len(self.categories))
        assert len(self.flip_categories) == len(self.categories)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        self.load_txt()
        
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        _img = _img.resize((192, 256), Image.LANCZOS)
        if self._flip_flag:
            if random.random() < 0.5:
                _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
                _target =_target.resize((192, 256), Image.LANCZOS)
        else:
            _target = Image.open(self.categories[index])
            _target =_target.resize((192, 256), Image.LANCZOS)

        return _img, _target

    def __str__(self):
        return 'CIHP(split=' + str(self.split) + ')'



