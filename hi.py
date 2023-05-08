import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project the query, key, and value into num_heads subspaces
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # Compute the scaled dot-product attention for each head
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add a dimension for the head
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)

        # Merge the num_heads and d_model dimensions back together
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.depth)
        output = self.output_projection(context)

        return output, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob, max_seq_len=1000):
        super().__init__()

        self.dropout_prob = dropout_prob

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros((1, max_seq_len, d_model))
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout_prob=dropout_prob)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        residual = x

        x, attention = self.multi_head_attention(x, x, x, mask=attention_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)

        return x, attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob):
        super().__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout_prob=dropout_prob)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, x, encoder_output, decoder_mask=None, encoder_mask=None):
        residual = x

        x, self_attn = self.multi_head_attention1(x, x, x, mask=decoder_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)

        residual = x

        # Pass the output of the first attention block through the encoder-decoder attention block
        x, encoder_attn = self.multi_head_attention2(x, encoder_output, encoder_output, mask=encoder_mask)
        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = self.layer_norm3(residual + x)

        return x, self_attn, encoder_attn


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers, dropout_prob):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout_prob=dropout_prob)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_prob=dropout_prob)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_prob=dropout_prob)
            for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_seq, target_seq):

        input_seq = self.embedding(input_seq)
        input_seq = input_seq * math.sqrt(self.d_model)
        input_seq = self.positional_encoding(input_seq)

        # Mask to avoid using future information in the decoder
        target_mask = (target_seq != 0).unsqueeze(1)
        decoder_mask = torch.ones(target_seq.size(1), target_seq.size(1)).tril()
        decoder_mask = decoder_mask.unsqueeze(0).expand(target_seq.size(0), -1, -1).to(target_seq.device)

        encoder_output, encoder_attention = self.encode(input_seq)
        decoder_output, self_attention, encoder_decoder_attention = self.decode(target_seq, encoder_output, target_mask=target_mask, decoder_mask=decoder_mask)

        output = self.output_projection(decoder_output)
        return output, self_attention, encoder_decoder_attention

    def encode(self, x):
        attention = []

        for layer in self.encoder_layers:
            x, attn = layer(x)
            attention.append(attn)

        return x, attention

    def decode(self, x, encoder_output, target_mask=None, decoder_mask=None):
        self_attention = []
        encoder_decoder_attention = []

        for layer in self.decoder_layers:
            x, self_attn, enc_dec_attn = layer(x, encoder_output, decoder_mask=decoder_mask, encoder_mask=target_mask)
            self_attention.append(self_attn)
            encoder_decoder_attention.append(enc_dec_attn)

        return x, self_attention, encoder_decoder_attention


