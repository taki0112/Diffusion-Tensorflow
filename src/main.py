from Diffusion import Diffusion
import argparse
from utils import *

def parse_args():
    desc = "Tensorflow implementation of Diffusion"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='dataset_name')

    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--iteration', type=int, default=100000, help='The total iterations')
    parser.add_argument('--print_freq', type=int, default=2000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--n_samples', type=int, default=16, help='The number of generated images')

    parser.add_argument('--timesteps', type=int, default=1000, help='The number of generated images')
    parser.add_argument('--objective', type=str, default='ddpm', help='[ddpm, ddim, general]')
    parser.add_argument('--eta', type=float, default=1.0, help='for reverse general form')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='ema_decay')
    parser.add_argument('--beta_schedule', type=str, default='cosine', help='linear or cosine')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():

    args = vars(parse_args())

    strategy = tf.distribute.MirroredStrategy()
    NUM_GPUS = strategy.num_replicas_in_sync
    batch_size = args['batch_size'] * NUM_GPUS  # global batch size

    # training parameters
    training_parameters = {
        **args,
        'batch_size': batch_size,
        'NUM_GPUS' : NUM_GPUS,
    }

    # automatic_gpu_usage()
    with strategy.scope():
        diffusion = Diffusion(training_parameters, strategy)

        # build graph
        diffusion.build_model()

        if args['phase'] == 'train' :
            diffusion.train()
            print(" [*] Training finished!")


if __name__ == '__main__':
    main()