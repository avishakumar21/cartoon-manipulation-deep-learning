import random
import pdb
import sys
import torch
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from logger import Logger
from torchvision.utils import make_grid
#from trainers.pix2pix_trainer import Pix2PixTrainer
from trainers import create_trainer

# parse options
opt = TrainOptions().parse()


# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader_train, dataloader_val = data.create_dataloader_trainval(opt)

# create trainer for our model
trainer = create_trainer(opt)
model = trainer.pix2pix_model
trainer.save('latest')

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader_train))

# create tool for visualization
writer = Logger(f"output/{opt.name}")

for epoch in iter_counter.training_epochs():
    def display(data_i):
        losses = trainer.get_latest_losses()
        for k,v in losses.items():
            writer.add_scalar(k,v.mean().item(), iter_counter.total_steps_so_far)
        writer.write_console(epoch, iter_counter.epoch_iter, iter_counter.time_per_iter)
        num_print = min(4, data_i['image'].size(0))
        infer_out,_ = trainer.pix2pix_model.forward(data_i, mode='inference')
        vis = (make_grid(infer_out[:num_print])+1)/2
        vis = torch.clamp(vis, 0,1)
        writer.add_single_image('infer_out',
                vis,
                iter_counter.total_steps_so_far)
        writer.add_single_image('inputs',
                (make_grid(trainer.get_latest_inputs()[:num_print])+1)/2,
                iter_counter.total_steps_so_far)
        generated = trainer.get_latest_generated()
        for k,v in generated.items():
            if v.size(1) == 3:
                vis = (make_grid(v[:num_print])+1)/2
            else:
                vis = make_grid(v[:num_print])
            writer.add_single_image(k,
                    vis,
                    iter_counter.total_steps_so_far)
        writer.write_html()
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader_train, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        flag = random.randint(0,1)
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i, i)

        if iter_counter.needs_displaying() and flag:
            display(data_i)

        if iter_counter.needs_validation():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('epoch%d_step%d'%
                    (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
            print("doing validation")
            model.eval()
            num = 0
            psnr_total = 0
            for ii, data_ii in enumerate(dataloader_val):
                mask = data_ii['mask']
                with torch.no_grad():
                    generated,_ = model(data_ii, mode='inference')
                    generated = generated.cpu()
                generated = (generated+1)/2*255
                gt = data_ii['image']
                bsize, c, h, w = gt.shape
                gt = (gt+1)/2*255
                mse = ((generated-gt)**2*mask).sum(3).sum(2).sum(1)
                mse /= (c*h*w)
                psnr = 10*torch.log10(255.0*255.0 / (mse+1e-8))
                psnr_total += psnr.sum().item()
                num += bsize
            psnr_total /= num
            writer.add_scalar("val.psnr", psnr_total, iter_counter.total_steps_so_far)
            writer.write_scalar("val.psnr", psnr_total, iter_counter.total_steps_so_far)
            writer.write_html()
            model.train()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

print('Training was successfully finished.')
