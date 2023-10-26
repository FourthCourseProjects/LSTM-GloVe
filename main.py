from torchtext import datasets
from torchtext.data import to_map_style_dataset
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import torch
from torchtext.vocab import GloVe
import torch.nn as nn
import time


# OBTENCIÓN DEL DATASET
train_iter, test_iter = datasets.AG_NEWS(split=('train', 'test'))
train_ds = to_map_style_dataset(train_iter)
test_ds = to_map_style_dataset(test_iter)
train = np.array(train_ds)
test = np.array(test_ds)


# TOKENIZACIÓN
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=['<pad>','<unk>'])
vocab.set_default_index(vocab["<unk>"])


print("Tamaño del vocabulario:", len(vocab), "tokens")
print("Tokenización de la frase 'Here is an example sentence':", tokenizer("Here is an example sentence"))
print("Índices de las palabras 'here', 'is', 'an', 'example', 'supercalifragilisticexpialidocious':", vocab(['here', 'is', 'an', 'example', 'supercalifragilisticexpialidocious']))
print("Palabras correspondientes a los índices 475, 21, 30, 5297, 0:", vocab.lookup_tokens([475, 21, 30, 5297, 0]))
print("Las diez primeras palabras del vocabulario:", vocab.get_itos()[:10])


# INDEXACIÓN DE LOS TOKENS
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
print("Tokenización de la frase 'Here is an example sentence':", text_pipeline("Here is an example sentence"))


# FUNCIÓN PARA PREPROCESAR LOS CONJUNTOS DE TRAIN Y TEST
# Para ello, crea los tokens para dichos textos, crea un tensor con ellos, y rellena los heucos vacíos con padding
def collate_batch(batch):
    label_list, text_list = [], []
    for sample in batch:
        label, text = sample
        text_list.append(torch.tensor(text_pipeline(text), dtype=torch.long))
        label_list.append(label_pipeline(label))
    return torch.tensor(label_list, dtype=torch.long), torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])

train_dataloader = DataLoader(
    train_iter, batch_size=64, shuffle=True, collate_fn=collate_batch
)

test_dataloader = DataLoader(
    test_iter, batch_size=64, shuffle=True, collate_fn=collate_batch
)

for batch in train_dataloader:
    print(batch[1][:4])
    print("\n")
    print(batch[0][:4])
    print("\n")
    break


class LSTMTextClassificationModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_class):
        super(LSTMTextClassificationModel, self).__init__()
        self.embedding = GloVe(dim=embed_dim)  # <-- Capa de embedding genérica (no pre-entrenada)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text):
        embedded = self.embedding().get_vecs_by_tokens(text, lower_case_backup=True)  # <-- Tras pasar por la capa de embedding, las palabras se representan como vectores
        lstm_out, _ = self.lstm(embedded)
        # Tomar la última salida de la secuencia LSTM
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


model = LSTMTextClassificationModel(32, 64, 4)
model.train()

for batch in train_dataloader:
    predicted_label = model(batch[1])
    label = batch[0]
    break
print(batch[1][:4])
print(predicted_label[:4])
print(label[:4])


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


def train(dataloader):
    model.train()
    total_acc, total_count, max_acc = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| {:5d} batches '
                  '| accuracy {:8.3f}'.format(idx, total_acc / total_count))

            if max_acc < total_acc / total_count:
                max_acc = total_acc / total_count

            total_acc, total_count = 0, 0
            start_time = time.time()
    return max_acc

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()

    accu_train = train(train_dataloader)
    accu_val = evaluate(test_dataloader)

    # if accu_train > accu_val:
    #    scheduler.step()

    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)