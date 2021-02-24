import pandas as pd
import spacy
import numpy as np
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer


from sklearn.model_selection import train_test_split
from torchtext import data

from dataset.dataframe_dataset import DataFrameDataset

# Import Data
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

# to clean data
def normalise_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

train["text"]=normalise_text(train["text"])

# split data into train and validation
train_df, valid_df = train_test_split(train)

tokenizer = get_tokenizer("spacy")

#Create Fields
TEXT = data.Field(tokenize = tokenizer, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)
fields = [('text',TEXT), ('label',LABEL)]

#create dataframe dataset
train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)

#Build Vocab
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_ds,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.200d',
                 unk_init = torch.Tensor.zero_)

LABEL.build_vocab(train_ds)

# Hyperparameters
num_epochs = 1
learning_rate = 0.001

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_ds, val_ds),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)

#build model
class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))

        # hidden = [batch size, hid dim * num directions]

        return output

# creating instance of our LSTM_net class

model = LSTM_net(INPUT_DIM,
                 EMBEDDING_DIM,
                 HIDDEN_DIM,
                 OUTPUT_DIM,
                 N_LAYERS,
                 BIDIRECTIONAL,
                 DROPOUT,
                 PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

#  to initiaise padded to zeros
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

model.to(device)  # CNN to GPU

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# training function
def train(model, iterator):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.text

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label)

            epoch_acc += acc.item()

    return epoch_acc / len(iterator)


loss = []
acc = []
val_acc = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evaluate(model, valid_iterator)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Acc: {valid_acc * 100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)

