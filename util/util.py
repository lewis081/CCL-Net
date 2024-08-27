"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

# lab:[-1,1], CHW
# rgb:[0,255], HWC
def lab2rgb(image_numpy):
    # CHW -> HWC; [-1,1] -> [0,1]
    image_numpy_norm01 = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0  # cvt Lab channels from [-1,1] to [0,1]

    # norm Lab[0,1] to Lab[0,255;42,226;20,223]
    min_Lab = np.array([0., 42., 20.])
    norm_Lab = np.array([255., 184., 203.])  # min_Lab = np.array([0.,42.,20.]), max_Lab = np.array([255.,226.,223.])
    for i in range(3):
        image_numpy_norm01[:, :, i] = image_numpy_norm01[:, :, i] * norm_Lab[i] + min_Lab[i]

    image_numpy_norm01 = image_numpy_norm01.astype(np.uint8)

    # Lab to RGB[0,255]
    image_numpy_norm01 = cv2.cvtColor(image_numpy_norm01, cv2.COLOR_Lab2RGB)
    return image_numpy_norm01

def tensor2im(input_image, imtype=np.uint8, mask=False, inputIsLabFormat = False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if inputIsLabFormat:
            image_numpy = lab2rgb(image_numpy)
        else:
            if len(image_numpy.shape) < 3:
                image_numpy = np.expand_dims(image_numpy, axis=0)
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def masks2RGBimg(input_image):
    """
    Converts a 8 channels mask tensor to a numpy image array.
    :param input_image: the input image tensor array with 8 channels
    :return: 3 channels numpy image array
    """
    cmap = np.zeros([8, 3]).astype(np.uint8)
    cmap[0, :] = np.array([0, 0, 0])
    cmap[1, :] = np.array([0, 0, 255])
    cmap[2, :] = np.array([0, 255, 0])
    cmap[3, :] = np.array([0, 255, 255])
    cmap[4, :] = np.array([255, 0, 0])
    cmap[5, :] = np.array([255, 0, 255])
    cmap[6, :] = np.array([255, 255, 0])
    cmap[7, :] = np.array([255, 255, 255])

    image_tensor = input_image.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy[image_numpy > 0.5] = 1
    image_numpy[image_numpy <= 0.5] = 0
    color_img = np.zeros((3, image_numpy.shape[1], image_numpy.shape[2]))
    for i in range(image_numpy.shape[0]):
        gray_image = image_numpy[i]

        mask = gray_image == 1
        color_img[0][mask] = cmap[i][0]
        color_img[1][mask] = cmap[i][1]
        color_img[2][mask] = cmap[i][2]
    color_img = np.transpose(color_img.astype(np.uint8), (1, 2, 0))
    return color_img

def get_saliency(divide_masks):
    salience = torch.argmax(divide_masks, dim=1)

    for i in range(divide_masks.shape[1]):
        if i == 0 or i == 7:
            salience[salience == i] = 0
        else:
            salience[salience == i] = 1
    salience = salience.float()
    # salience.requires_grad = True
    salience = torch.unsqueeze(salience, 0)
    return salience

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


###################################
# dataset process
###################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


def getRobotFishHumanReefWrecks(mask):
    # for categories: HD, RO, FV, RI, WR
    # mask = np.array(mask)
    # mask[mask > 0.5] = 1
    # mask[mask <= 0.5] = 0
    imw, imh = mask.shape[1], mask.shape[2]
    Human = np.zeros((imw, imh))
    Robot = np.zeros((imw, imh))
    Fish = np.zeros((imw, imh))
    Reef = np.zeros((imw, imh))
    Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[0, i, j] == 0 and mask[1, i, j] == 0 and mask[2, i, j] == 1):
                Human[i, j] = 1
            elif (mask[0, i, j] == 1 and mask[1, i, j] == 0 and mask[2, i, j] == 0):
                Robot[i, j] = 1
            elif (mask[0, i, j] == 1 and mask[1, i, j] == 1 and mask[2, i, j] == 0):
                Fish[i, j] = 1
            elif (mask[0, i, j] == 1 and mask[1, i, j] == 0 and mask[2, i, j] == 1):
                Reef[i, j] = 1
            elif (mask[0, i, j] == 0 and mask[1, i, j] == 1 and mask[2, i, j] == 1):
                Wreck[i, j] = 1
            else:
                pass
    mask_indiv = np.stack((Robot, Fish, Human, Reef, Wreck), 0)
    return torch.from_numpy(mask_indiv).float()

def getSemanticSegmentation(image_numpy):
    imw, imh = image_numpy.shape[1], image_numpy.shape[2]
    mask = np.zeros((3, imw, imh))
    for i in range(imw):
        for j in range(imh):
            pass


def to_image(x):
    return (x * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)


def visualize_feature_map(img_batch, image_name, suffix='', max_chs=4):
    save_dir = './results/demo/vis_fm/d_r_386_inferno'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_batch = img_batch.cpu().numpy()
    feature_map = np.squeeze(img_batch, axis=0)
    feature_map = feature_map.transpose((1, 2, 0))
    print(feature_map.shape)

    feature_map_combination = []
    # plt.figure()

    num_pic = feature_map.shape[2]
    print('feature map channels:%d' % num_pic)
    # row, col = get_row_col(num_pic)
    #
    for i in range(0, num_pic):
        if i > max_chs:
            break
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        plt.figure(figsize=(6.4, 4.2))
        plt.imshow(feature_map_split)
        plt.axis('off')
        plt.title('feature_map_{}'.format(i))
        plt.savefig(os.path.join(save_dir, '{}_{}_feature_map_{}.png'.format(image_name, suffix, i)))
    # plt.imshow(feature_map[:, :, 227])
    # plt.savefig('/data/qiqi/project/Underwater_Co-Image-style_Enhance/results/demo/vis_fm/feature_map3_227.png')
    # plt.show()
    #
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig(os.path.join(save_dir, "{}_{}_feature_map_sum.png".format(image_name, suffix)))




