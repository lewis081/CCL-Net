import time
from options.train_options import TrainOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

from models.CCLNet.Public.util.AverageMeter import AverageMeter

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    print(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    best_loss = 9999.0
    best_epoch = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        losses_cnt = AverageMeter()

        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print(f'i: {i}')
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            #need hide
            lss = 0.
            for k,v in model.get_current_losses().items():
                lss += v
            losses_cnt.update(lss)

        if np.isnan(losses_cnt.avg):
            visualizer.print_msg("losses_cnt.avg is nan, jump out loop of epoch")
            break

        if losses_cnt.avg < best_loss:
            best_loss = losses_cnt.avg
            best_epoch = epoch
            model.save_networks("best")

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            visualizer.print_msg('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        losses = model.get_current_losses()
        losses['loss_avg'] = losses_cnt.avg
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, None, losses)
        visualizer.print_msg('End Epoch %d. Avg Loss: %f -------- ' % (epoch, losses_cnt.avg))
        visualizer.print_msg('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    visualizer.print_msg('Finish Training. best_loss: %f, best_epoch: %d' % (best_loss, best_epoch))

    visualizer.print_msg('Train Done!')
