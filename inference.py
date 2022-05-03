import torch
from models.seq2seq import Seq2Seq
import pickle

def inference():

    # Load the model and dataset
    with open('./checkpoints/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    with open('./checkpoints/args.pkl', 'rb') as f:
        args = pickle.load(f)
    model = Seq2Seq(args, dataset.start_token, dataset.end_token)
    model.load_state_dict(torch.load('./checkpoints/seq2seq.pth'))


    while True:
        src = input('Enter the source date (format like %y-%m-%d, eg. 31-04-18): ')
        src_id = dataset.str2idx(src)
        dst = model.inference(torch.tensor(src_id).unsqueeze(0))
        print('The destination date is: %s \n' % dataset.idx2str(dst[0]))



if __name__ == '__main__':
    inference()