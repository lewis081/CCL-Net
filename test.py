import os
from options.test_options import TestOptions
from datasets import create_dataset
from models import create_model
from util.util import save_image, tensor2im

from datetime import datetime
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create result directory
    save_dir = os.path.join(opt.results_dir, opt.results_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visual_names = model.get_current_visual_names()
    save_dirs = {}
    for v_name in visual_names:
        save_dirs[v_name] = os.path.join(save_dir, v_name)
        if not os.path.exists(save_dirs[v_name]):
            os.makedirs(save_dirs[v_name])

    if opt.eval:
        model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i, data in enumerate(dataset):
        start_time = datetime.now()
        starter.record()

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals, _ = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print(f'processing {i}-th image... {img_path}: [CPU]{datetime.now() - start_time} ms')

        ender.record()
        torch.cuda.synchronize()
        infer_used_time = starter.elapsed_time(ender)
        print(f'processing {i}-th image... : [GPU]{infer_used_time} ms')

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save result
        img_name = os.path.basename(img_path[0])
        # im = tensor2im(visuals.get('pred_enhancement'))
        # save_image(im, os.path.join(save_dir, img_name))

        for v_name, v_img in visuals.items():
            im = tensor2im(v_img)
            save_image(im, os.path.join(save_dirs[v_name], img_name))


    print('Test Done!')
