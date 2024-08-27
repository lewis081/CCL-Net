import torch

# def rgb_to_lab(srgb):
#     # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
#     with tf.name_scope('rgb_to_lab'):
#         srgb = check_image(srgb)
#         srgb_pixels = tf.reshape(srgb, [-1, 3])
#         with tf.name_scope('srgb_to_xyz'):
#             linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
#             exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
#             rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
#             rgb_to_xyz = tf.constant([
#                 #    X        Y          Z
#                 [0.412453, 0.212671, 0.019334], # R
#                 [0.357580, 0.715160, 0.119193], # G
#                 [0.180423, 0.072169, 0.950227], # B
#             ])
#             xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
#
#         # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
#         with tf.name_scope('xyz_to_cielab'):
#             # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)
#
#             # normalize for D65 white point
#             xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
#
#             epsilon = 6/29
#             linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
#             exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
#             fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask
#
#             # convert to lab
#             fxfyfz_to_lab = tf.constant([
#                 #  l       a       b
#                 [  0.0,  500.0,    0.0], # fx
#                 [116.0, -500.0,  200.0], # fy
#                 [  0.0,    0.0, -200.0], # fz
#             ])
#             lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
#
#         return tf.reshape(lab_pixels, tf.shape(srgb))


# https://blog.csdn.net/bby1987/article/details/109522126
# output
# L, min: 0.000000, max: 100.000000
# a, min: -86.183030, max: 98.233054
# b, min: -107.857300, max: 94.478122
def rgb_to_lab(srgb, useGPU=True):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c

    # srgb = check_image(srgb)
    # srgb_pixels = tf.reshape(srgb, [-1, 3])
    if len(srgb.shape) == 4:
        srgb_pixels = srgb.permute(0,2,3,1) # b,c,h,w -> b,h,w,c
    elif len(srgb.shape) == 3:
        srgb_pixels = srgb.permute(1, 2, 0)  # c,h,w -> h,w,c
    else:
        print("len(srgb.shape) is 3 or 4, quit function rgb_to_lab")
        return

    linear_mask = (srgb_pixels <= 0.04045).to(dtype=torch.float) # type bool cvt to float
    exponential_mask = (srgb_pixels > 0.04045).to(dtype=torch.float) #https://blog.csdn.net/a15608445683/article/details/124573132
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334], # R
        [0.357580, 0.715160, 0.119193], # G
        [0.180423, 0.072169, 0.950227], # B
    ])
    if useGPU and torch.cuda.is_available():
        rgb_to_xyz = rgb_to_xyz.cuda()
    xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz) #https://blog.csdn.net/weixin_42065178/article/details/119517404

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions

    # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

    # normalize for D65 white point
#     white = torch.tensor([1/0.950456, 1.0, 1/1.088754])
    white = torch.tensor([1/0.95047, 1.0, 1/1.08883])
    if useGPU and torch.cuda.is_available():
        white = white.cuda()
    xyz_normalized_pixels = torch.multiply(xyz_pixels, white)

    epsilon = 6/29
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).to(dtype=torch.float)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).to(dtype=torch.float)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ])
    if useGPU and torch.cuda.is_available():
        fxfyfz_to_lab = fxfyfz_to_lab.cuda()

    unkown_const = torch.tensor([-16.0, 0.0, 0.0])
    if useGPU and torch.cuda.is_available():
        unkown_const = unkown_const.cuda()
    lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + unkown_const

#     print("lab_pixels.shape: ", lab_pixels.shape)
#     print("srgb.shape: ", srgb.shape)

#     return torch.reshape(lab_pixels, srgb.shape)

    lab_pixels = lab_norm(lab_pixels, useGPU)

    if len(srgb.shape) == 4:
        return lab_pixels.permute(0,3,1,2) # b,h,w,c -> b,c,h,w
    elif len(srgb.shape) == 3:
        return lab_pixels.permute(2,0,1) # h,w,c -> c,h,w

#input
# lab : b,c,h,w or c,h,w
# L=L/100.0
# a=(a+86.183030)/184.416084
# b=(b+107.857300)/202.335422
# def lab_norm1(lab, useGPU=True):
#     if len(lab.shape) == 4:
#         return lab_pixels.permute(0,3,1,2) # b,h,w,c -> b,c,h,w
#     elif len(lab.shape) == 3:
#         l, a, b = torch.split(lab, 1, dim=0)
#         l = l/100.0
#         a = (a+86.183030)/184.416084
#         b = (b + 107.857300) / 202.335422
#         return torch.cat([],0)
#     pass

def lab_norm(lab, useGPU=True):
    lab_to_norm = torch.tensor([
        # l_n  a_n       b_n
        [0.01, 0.0,      0.0],      # l
        [0.0,  0.005423, 0.0],      # a
        [0.0,  0.0,      0.004942], # b
    ])
    if useGPU and torch.cuda.is_available():
        lab_to_norm = lab_to_norm.cuda()

    unkown_const2 = torch.tensor([0.0, 0.467329, 0.533062])
    if useGPU and torch.cuda.is_available():
        unkown_const2 = unkown_const2.cuda()
    lab_n = torch.matmul(lab, lab_to_norm) + unkown_const2

    return lab_n
    pass