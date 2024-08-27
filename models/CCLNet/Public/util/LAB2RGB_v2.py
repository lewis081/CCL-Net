import torch

# def lab_to_rgb(lab):
#     with tf.name_scope('lab_to_rgb'):
#         lab = check_image(lab)
#         lab_pixels = tf.reshape(lab, [-1, 3])
#         # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
#         with tf.name_scope('cielab_to_xyz'):
#             # convert to fxfyfz
#             lab_to_fxfyfz = tf.constant([
#                 #   fx      fy        fz
#                 [1/116.0, 1/116.0,  1/116.0], # l
#                 [1/500.0,     0.0,      0.0], # a
#                 [    0.0,     0.0, -1/200.0], # b
#             ])
#             fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)
#
#             # convert to xyz
#             epsilon = 6/29
#             linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
#             exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
#             xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
#
#             # denormalize for D65 white point
#             xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])
#
#         with tf.name_scope('xyz_to_srgb'):
#             xyz_to_rgb = tf.constant([
#                 #     r           g          b
#                 [ 3.2404542, -0.9692660,  0.0556434], # x
#                 [-1.5371385,  1.8760108, -0.2040259], # y
#                 [-0.4985314,  0.0415560,  1.0572252], # z
#             ])
#             rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
#             # avoid a slightly negative number messing up the conversion
#             rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
#             linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
#             exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
#             srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask
#
#         return tf.reshape(srgb_pixels, tf.shape(lab))


# input: if not needInverseNorm
# L, min: 0.000000, max: 100.000000
# a, min: -86.183030, max: 98.233054
# b, min: -107.857300, max: 94.478122
# input: if needInverseNorm
# Lab: [0,1.0]

# output:
# RGB: [0,1.0]

class Lab2RGB():
    def __init__(self, useGPU=True):
        self.lab_to_inorm = torch.tensor([
        # l_in  a_in     b_in
        [100.0 , 0.0,      0.0],      # l_n
        [0.0,  184.4161,   0.0],      # a_n
        [0.0,  0.0,      202.3354],   # b_n
        ])
        if useGPU and torch.cuda.is_available():
            self.lab_to_inorm = self.lab_to_inorm.cuda()

        self.unkown_const2 = torch.tensor([0.0, -86.1830, -107.8573])
        if useGPU and torch.cuda.is_available():
            self.unkown_const2 = self.unkown_const2.cuda()

        self.lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
        ])
        if useGPU and torch.cuda.is_available():
            self.lab_to_fxfyfz = self.lab_to_fxfyfz.cuda()

        self.unkown_const = torch.tensor([16.0/116, 16.0/116, 16.0/116])
        if useGPU and torch.cuda.is_available():
            self.unkown_const = self.unkown_const.cuda()

        self.white = torch.tensor([0.950456, 1.0, 1.088754])
        if useGPU and torch.cuda.is_available():
            self.white = self.white.cuda()

        self.xyz_to_rgb = torch.tensor([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
        ]) # inverse of M_{RGB2XYZ}^T
        if useGPU and torch.cuda.is_available():
            self.xyz_to_rgb = self.xyz_to_rgb.cuda()

        # print('init done')
    def lab_inverse_norm(self, lab_n):
        lab_in = torch.matmul(lab_n, self.lab_to_inorm) + self.unkown_const2

        return lab_in

    def lab_to_rgb(self, lab, needInverseNorm=True):
        if len(lab.shape) == 4:
            lab_pixels = lab.permute(0, 2, 3, 1)  # b,c,h,w -> b,h,w,c
        elif len(lab.shape) == 3:
            lab_pixels = lab.permute(1, 2, 0)  # c,h,w -> h,w,c
        else:
            print("len(lab.shape) is 3 or 4, quit function lab_to_rgb")
            return

        if needInverseNorm:
            # inverse norm from [0,1] to [[0,100], [-86.18,98.23], [-107.86,94.48]]
            lab_pixels = self.lab_inverse_norm(lab_pixels)

        # convert to fxfyfz
        # lab_to_fxfyfz = torch.tensor([
        #     #   fx      fy        fz
        #     [1/116.0, 1/116.0,  1/116.0], # l
        #     [1/500.0,     0.0,      0.0], # a
        #     [    0.0,     0.0, -1/200.0], # b
        # ])
        # if useGPU and torch.cuda.is_available():
        #     lab_to_fxfyfz = lab_to_fxfyfz.cuda()

        # unkown_const = torch.tensor([16.0/116, 16.0/116, 16.0/116])
        # if useGPU and torch.cuda.is_available():
        #     unkown_const = unkown_const.cuda()

        fxfyfz_pixels = torch.matmul(lab_pixels, self.lab_to_fxfyfz) + self.unkown_const

        # print(f'fxfyfz_pixels: {fxfyfz_pixels}' )

        # convert to xyz
        epsilon = 6. / 29
        linear_mask = (fxfyfz_pixels <= epsilon).to(dtype=torch.float)
        exponential_mask = (fxfyfz_pixels > epsilon ).to(dtype=torch.float)
        # fxfyfz_pixels = (fxfyfz_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
        # fxfyfz_pixels ** (1 / 3)) * exponential_mask
        xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4. / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

        # print(f'xyz_pixels: {xyz_pixels}')

        # denormalize for D65 white point
        # white = torch.tensor([0.950456, 1.0, 1.088754])
        # if useGPU and torch.cuda.is_available():
        #     white = white.cuda()
        xyz_pixels = torch.multiply(xyz_pixels, self.white)
        # print(xyz_pixels) // xyz

        # xyz_to_srgb
        # xyz_to_rgb = torch.tensor([
        #     #     r           g          b
        #     [ 3.2404542, -0.9692660,  0.0556434], # x
        #     [-1.5371385,  1.8760108, -0.2040259], # y
        #     [-0.4985314,  0.0415560,  1.0572252], # z
        # ]) # inverse of M_{RGB2XYZ}^T
        # if useGPU and torch.cuda.is_available():
        #     xyz_to_rgb = xyz_to_rgb.cuda()
        rgb_pixels = torch.matmul(xyz_pixels, self.xyz_to_rgb)
        # print(f'rgb_pixels: {rgb_pixels}')

        # # avoid a slightly negative number messing up the conversion
        # rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0) // why need this step at here in UColor srccode?
        rgb_pixels = torch.clip(rgb_pixels, 0.0, 1.0)
        linear_mask = (rgb_pixels <= 0.0031308).to(dtype=torch.float)
        exponential_mask = (rgb_pixels > 0.0031308).to(dtype=torch.float)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        # print(f'srgb_pixels: {srgb_pixels}')
        # srgb_pixels = torch.clip(srgb_pixels, 0.0, 1.0)

        if len(lab.shape) == 4:
            return srgb_pixels.permute(0,3,1,2) # b,h,w,c -> b,c,h,w
        elif len(lab.shape) == 3:
            return srgb_pixels.permute(2,0,1) # h,w,c -> c,h,w

    # lab: [-1,1]
    # rgb: [-1,1]
    def labn12p1_to_rgbn12p1(self, lab):
        #[-1,1] to [0,1]
        lab_021 = (lab + 1)/2
        rgb_021 = self.lab_to_rgb(lab_021)

        #[0,1] to [-1,1]
        return 2*rgb_021 - 1

# import numpy as np
# from skimage import color
# if __name__ == '__main__':
#     print('START@@: lab_inverse_norm test ')
#     lab_n = torch.tensor([[[[0., 0., 0.]]]])
#     print(lab_n.shape)
#     print(lab_inverse_norm(lab_n)) # tensor([[[[   0.0000,  -86.1830, -107.8573]]]])
#
#     lab_n = torch.tensor([[[[1.0, 1.0, 1.0]]]])
#     print(lab_inverse_norm(lab_n))  # tensor([[[ 100.0000, 98.2331, 94.4781]]])
#
#     print('END@@: lab_inverse_norm test ')
#     print('START@@: lab_to_rgb test ')
#     lab_ = torch.tensor([[[[60.15671581, -14.50893467,  63.59532313]]]]) # [L,a,b]
#     print(
#         255*lab_to_rgb(lab_.permute(0, 3, 1, 2), needInverseNorm=False))  # # tensor([[[[   149.9973,  150.0014, 0.0000]]]])
#
#     lab_np = np.array([[[ 60.15671581, -14.50893467,  63.59532313]]], dtype=np.float32)
#     # print(color.lab2xyz(lab_np)) #[[[0.2348503  0.2829767  0.04224898]]]
#     rgb2_ = np.round(255.0 * np.clip(color.lab2rgb(lab_np), 0, 1)).astype(np.uint8) #[[[150   150   0]]]
#     print(rgb2_)
#
#     lab_n = torch.tensor([[[[0.5324, 0.9017, 0.8652]]]])  # [L,a,b]
#     print(
#         lab_to_rgb(lab_n.permute(0, 3, 1, 2)))  # tensor([[[1.0000 0.0000 0.0000]]])
#     print(
#         255 * lab_to_rgb(lab_n.permute(0, 3, 1, 2)))  # tensor([[[255.0000 0.0000 0.0000]]])
#     print('END@@: lab_to_rgb test ')
