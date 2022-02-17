import os
import math
import torch
import argparse
from transformers import GPT2Tokenizer, AdamW
from module import DiscretePromptLearning
from utils import rouge_score, bleu_score, DataLoader, Batchify2, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for the model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify2(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size, shuffle=True)
val_data = Batchify2(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size)
test_data = Batchify2(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
model = DiscretePromptLearning.from_pretrained('gpt2')
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        seq, mask, prompt = data.next_batch()  # data.step += 1
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)
        prompt = prompt.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs = model(prompt, seq, mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = seq.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | {:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step, data.total_step))
            text_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            seq, mask, prompt = data.next_batch()  # data.step += 1
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            prompt = prompt.to(device)
            outputs = model(prompt, seq, mask)
            loss = outputs.loss

            batch_size = seq.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    input_features = []
    with torch.no_grad():
        while True:
            seq, _, prompt = data.next_batch()  # data.step += 1
            prompt = prompt.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(prompt, text, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)
            input_features.extend(prompt.tolist())

            if data.step == data.total_step:
                break
    return idss_predict, input_features


# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted, features = generate(test_data)
tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
tokens_context = [tokenizer.decode(ids) for ids in features]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
    text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
