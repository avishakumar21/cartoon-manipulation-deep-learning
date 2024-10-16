from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
from data.image_folder import make_dataset
import os
import pdb


class TrainEdgeDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_edge_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_list', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_postfix', type=str, default=".jpg",
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_mask_postfix', type=str, default=".png",
                            help='path to the directory that contains photo images')
        return parser

    def initialize(self, opt):
        self.opt = opt
        image_paths, mask_paths = self.get_paths(opt)

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        size = len(self.image_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        image_dir = opt.train_image_dir
        mask_dir = opt.train_edge_dir
        image_list = opt.train_image_list
        names = open(image_list).readlines()
        filenames = list(map(lambda x: x.strip('\n').replace(opt.train_image_postfix, ""), names))
        image_paths = list(map(lambda x: os.path.join(image_dir, x+opt.train_image_postfix), filenames))
        mask_paths = list(map(lambda x: os.path.join(mask_dir, x+opt.train_mask_postfix), filenames))
        return image_paths, mask_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = mask.convert('L')

        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        transform_mask = get_transform(self.opt, params, normalize=False)
        mask_tensor = transform_mask(mask)

        input_dict = {
                      'image': image_tensor,
                      'edge': mask_tensor,
                      'path': image_path,
                      }
        return input_dict
