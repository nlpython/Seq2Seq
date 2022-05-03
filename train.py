import torch
from torch import nn
from dataset import DateDataset
from torch.utils.data import DataLoader
from hyperParams import print_arguments, get_parser
from loguru import logger
from models.seq2seq import Seq2Seq
import pickle

def train():

    logger.add("logs/train.log", rotation="1 day")

    # load args
    args = get_parser()
    print_arguments(args, logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # load dataset
    dataset = DateDataset(4000)
    print("Chinese time order: yy/mm/dd ",dataset.date_cn[:3],"\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
                f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}")

    args.enc_vocab_size = dataset.num_word
    args.dec_vocab_size = dataset.num_word
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Seq2Seq(args, dataset.start_token, dataset.end_token).to(device)

    optimeizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # save dataset and args
    with open('checkpoints/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open('checkpoints/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    # train
    logger.info("Start training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            global_step += 1
            # to device
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            logits = model(batch['enc_input'], batch['dec_input'])
            loss = loss_func(logits.reshape(-1, args.dec_vocab_size), batch['dec_output'].reshape(-1))

            optimeizer.zero_grad()
            loss.backward()
            optimeizer.step()

            if global_step % 100 == 0:
                model.eval()
                logger.info(f"Epoch {epoch} step {step} loss {loss.item():.4f}")
                target = dataset.idx2str(batch['dec_input'][0, 1:].cpu().numpy())
                pred = model.inference(batch['enc_input'][0:1])
                res = dataset.idx2str(pred[0].cpu().numpy())
                src = dataset.idx2str(batch['enc_input'][0].cpu().numpy())
                logger.info(f"Inference | target: {target}, pred:{res}\n")
                model.train()

    logger.info("Training finished.")
    logger.info('Saving model...')
    torch.save(model.state_dict(), 'checkpoints/seq2seq.pth')
    logger.info('Successfully saved.')


if __name__ == '__main__':
    train()