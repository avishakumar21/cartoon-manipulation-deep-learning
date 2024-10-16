import torchvision.transforms as transforms
import torch
from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
from data.image_folder import make_dataset
import os
import pdb


class ValEdgeDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--output_dir', type=str, required=False,
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_image_list', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_mask_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_edge_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_image_postfix', type=str, default=".jpg",
                            help='path to the directory that contains photo images')
        parser.add_argument('--val_mask_postfix', type=str, default=".png",
                            help='path to the directory that contains photo images')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if opt.output_dir is not None and not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)

        image_paths, edge_paths, mask_paths, output_paths = self.get_paths(opt)
        self.output_paths = output_paths

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.edge_paths = edge_paths

        size = len(self.image_paths)
        self.dataset_size = size
        transform_list = [
                transforms.Resize((opt.crop_size, opt.crop_size), 
                    interpolation=Image.NEAREST),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size),interpolation=Image.NEAREST),
            transforms.ToTensor()
            ])

    def get_paths(self, opt):
        image_dir = opt.val_image_dir
        image_list = opt.val_image_list
        names = open(image_list).readlines()
        filenames = list(map(lambda x: x.strip('\n').replace(opt.val_image_postfix, ""), names))
        image_paths = list(map(lambda x: os.path.join(image_dir, x+opt.val_image_postfix), filenames))
        mask_paths = list(map(lambda x: os.path.join(opt.val_mask_dir, x+opt.val_mask_postfix), filenames))
        edge_paths = list(map(lambda x: os.path.join(opt.val_edge_dir, x+opt.val_mask_postfix), filenames))
        if opt.output_dir is not None:
            output_paths = list(map(lambda x: x+opt.val_image_postfix, filenames))
        else:
            output_paths = None
        return image_paths, edge_paths, mask_paths, output_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        if self.output_paths is not None:
            output_path = self.output_paths[index]
        else:
            output_path =0
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        image_tensor = self.image_transform(image)
        # mask image
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = mask.resize((w,h))
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor>0).float()

        edge_path = self.edge_paths[index]
        edge = Image.open(edge_path)
        edge = edge.convert("L")
        edge = edge.resize((w,h))
        edge_tensor = self.mask_transform(edge)
        edge_tensor = (edge_tensor>0).float()
        input_dict = {
                      'image': image_tensor,
                      'mask': mask_tensor,
                      'edge': edge_tensor,
                      'path': output_path,
                      }

        return input_dict
