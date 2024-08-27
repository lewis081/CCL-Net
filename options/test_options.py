from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        # image resize
        parser.add_argument('--image_height', type=int, default=480, help='resize images to this size') #?
        parser.add_argument('--image_width', type=int, default=640, help='resize images to this size') #?
        # rewrite devalue values
        parser.set_defaults(model='test') #?
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.set_defaults(preprocess='resize')
        parser.set_defaults(no_flip=True)
        self.isTrain = False
        return parser
