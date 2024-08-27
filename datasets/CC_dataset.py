import os.path
from datasets.base_dataset import BaseDataset, get_transform, get_params
from util.util import make_dataset
from PIL import Image
import cv2, numpy as np

import torchvision.transforms as transforms

class CCDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images
     - '/path/to/data/raw'  raw underwater images
     - '/path/to/data/ref'  ref enhancement images

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare three directories:
    '/path/to/data/raw', '/path/to/data/ref' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.isTrain = opt.isTrain

        self.img_size = opt.crop_size
        self.dir_raw = os.path.join(opt.dataroot, opt.phase, 'raw')

        if self.isTrain:
            self.dir_ref = os.path.join(opt.dataroot, opt.phase, 'ref')

        self.raw_paths = sorted(make_dataset(self.dir_raw, opt.max_dataset_size))  # load images from '/path/to/data/raw'

        self.raw_size = len(self.raw_paths)  # get the size of dataset A

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

    def __imreadCvtLabNorm(self, img_path):
        raw_img = cv2.imread(img_path) # HWC
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2Lab)

        min_Lab = np.array([0., 42., 20.])
        norm_Lab = np.array([255.,184.,203.]) #min_Lab = np.array([0.,42.,20.]), max_Lab = np.array([255.,226.,223.])

        raw_img = raw_img.astype(np.float32)
        for i in range(3):
            raw_img[:, :, i] = (raw_img[:, :, i] - min_Lab[i])/norm_Lab[i]
        # print(f'type(raw_img): {type(raw_img)}')
        return Image.fromarray(np.uint8(raw_img*255))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains raw, ref, mask, A_paths, B_paths and C_paths
            raw (tensor)       -- an image in the input domain
            ref (tensor)       -- its corresponding image in the target domain
            raw_paths (str)    -- image paths
            ref_paths (str)    -- image paths
        """
        raw_path = self.raw_paths[index % self.raw_size]  # make sure index is within then range
        raw_name = os.path.split(raw_path)[1]
        base_name = raw_name.split('.')[0]
        if self.isTrain:
            ref_path = os.path.join(self.dir_ref, raw_name)

        raw_img = self.__imreadCvtLabNorm(raw_path)
        if self.isTrain:
            ref_img = self.__imreadCvtLabNorm(ref_path)

        # apply image transformation
        if self.isTrain:
            transform_params = get_params(self.opt, raw_img.size)
            # transform_params = get_params(self.opt, (raw_img.shape[1], raw_img.shape[0]))
            self.transform_raw = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            self.transform_ref = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        else:
            self.transform_raw = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        raw = self.transform_raw(raw_img)

        if self.isTrain:
            ref = self.transform_ref(ref_img)

        if self.isTrain:
            return {'raw': raw, 'ref': ref, 'raw_paths': raw_path, 'ref_paths': ref_path}
        else:
            return {'raw': raw, 'raw_paths': raw_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.raw_size
