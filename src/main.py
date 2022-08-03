from DDGAN import DDGAN
import argparse
from utils import *

def parse_args():
    desc = "Tensorflow implementation of Denoising Diffusion GANs"
    parser = argparse.ArgumentParser(description=desc)

    # training
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='dataset_name')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')
    parser.add_argument('--use_geometric', type=str2bool, default=False)
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate g')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='decay rate for EMA')
    parser.add_argument('--r1_gamma', type=float, default=1., help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=10, help='lazy regulariation.')


    # generator & disriminator settings
    parser.add_argument('--g_nf', type=int, default=64, help='number of initial channels in generator')
    parser.add_argument('--d_nf', type=int, default=64, help='number of initial channels in discriminator')
    parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--conditional', type=str2bool, default=True, help='noise conditional')
    parser.add_argument('--fir', type=str2bool, default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', type=str2bool, default=True, help='skip rescale')

    # dimension
    parser.add_argument('--timesteps', type=int, default=4)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)

    # misc
    parser.add_argument('--cal_fid', type=str2bool, default=False, help='calculate fid in training')
    parser.add_argument('--print_freq', type=int, default=2000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')

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
        diffusion = DDGAN(training_parameters, strategy)

        # build graph
        diffusion.build_model()

        if args['phase'] == 'train' :
            diffusion.train()
            print(" [*] Training finished!")

        if args['phase'] == 'test' :
            diffusion.test()
            print(" [*] Test finished!")


if __name__ == '__main__':
    main()