from torch import nn
import torch

class Seq2Seq(nn.Module):

    def __init__(self, args, SOS, EOS):
        super().__init__()
        # Encoder
        self.enc_emb = nn.Embedding(args.enc_vocab_size, args.enc_emb_dim)
        self.enc_emb.weight.data.normal_(0, 0.1)
        self.enc_lstm = nn.LSTM(args.enc_emb_dim, args.enc_hid_dim, num_layers=1, batch_first=True)

        # Decoder
        self.dec_emb = nn.Embedding(args.dec_vocab_size, args.dec_emb_dim)
        self.dec_emb.weight.data.normal_(0, 0.1)
        self.dec_cell = nn.LSTMCell(args.dec_emb_dim, args.dec_hid_dim)
        self.dec_linear = nn.Linear(args.dec_hid_dim, args.dec_vocab_size)

        self.SOS = SOS
        self.EOS = EOS
        self.max_seq_len = args.max_seq_len

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size, seq_len]
        :param dec_inputs: [batch_size, seq_len+1(because of <sos>)]
        :return:
        """
        # Encode
        enc_emb = self.enc_emb(enc_inputs)
        _, (enc_h, enc_c) = self.enc_lstm(enc_emb)

        # Decode
        h, c = enc_h[0], enc_c[0]
        dec_embed = self.dec_emb(dec_inputs).permute(1, 0, 2)

        outputs = []
        for i in range(dec_embed.shape[0]):
            h, c = self.dec_cell(dec_embed[i], (h, c))
            output = self.dec_linear(h)
            outputs.append(output)
        # outputs.shape: [seq_len, batch_size, dec_vocab_size]
        outputs = torch.stack(outputs, dim=0)
        return outputs.permute(1, 0, 2)


    def inference(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, seq_len]
        :return:
        """
        # Encode
        enc_emb = self.enc_emb(enc_inputs)
        _, (h, c) = self.enc_lstm(enc_emb)
        h, c = h[0], c[0]

        # Decode
        batch_size = enc_inputs.size(0)
        start = torch.ones((batch_size, 1), device=enc_inputs.device).long()
        start[:, 0] = torch.tensor(self.SOS)
        dec_embed = self.dec_emb(start).permute(1, 0, 2)  # [1, batch_size, dec_emb_dim]
        dec_in = dec_embed[0]

        outputs = []
        for i in range(self.max_seq_len):
            h, c = self.dec_cell(dec_in, (h, c))
            o = self.dec_linear(h)
            o = o.argmax(dim=-1).view(-1, 1)
            outputs.append(o)
            dec_in = self.dec_emb(o).permute(1, 0, 2)[0]

        outputs = torch.stack(outputs, dim=0)
        return outputs.permute(1, 0, 2).view(-1, self.max_seq_len)

