import argparse
import six

def print_arguments(args, log):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')

def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters')

    # Model Data's path
    parser.add_argument('--save_path', type=str, default='checkpoints/',
                        help='Path to save the trained model')
    parser.add_argument('--log_path', type=str, default='logs/',
                        help='Path to save the training log')

    # Training Parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--max_seq_len', type=int, default=11,
                        help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # Model Parameters
    parser.add_argument('--enc_emb_dim', type=int, default=16,
                        help='Encoder embedding dimension')
    parser.add_argument('--dec_emb_dim', type=int, default=16,
                        help='Decoder embedding dimension')
    parser.add_argument('--enc_hid_dim', type=int, default=32,
                        help='Encoder hidden dimension')
    parser.add_argument('--dec_hid_dim', type=int, default=32,
                        help='Decoder hidden dimension')


    return parser.parse_args()