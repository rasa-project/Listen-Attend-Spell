from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import os
from io import open
from local.Load_feat import Voc, Loader
import local.kaldi_io.kaldi_io as kaldi_io
from local.Model import EncoderRNN
from local.Model import DecoderRNN
import datetime as datetime
import sys
import argparse

parser = argparse.ArgumentParser(description="speech training")
parser.add_argument('--label-file', metavar='DIR',
                    help='path to label', default='data/text')
parser.add_argument('--scp-file', metavar='DIR',
                    help='path to scp feats', default='data/text')
parser.add_argument('--corpus-name', help='corpus name', default='WSJ')
parser.add_argument('--seed', default=7, type=int, help='Seed to generators')
parser.add_argument('--lr', '--learning-rate', default=4e-5,
                    type=float, help='initial learning rate')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers used in data-loading')
parser.add_argument('--reduce', default=1, type=int,
                    help='Number of workers used in data-loading')
parser.add_argument('--input-size', default=120,
                    type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-size', default=320,
                    type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=4,
                    type=int, help='Number of RNN layers')
parser.add_argument('--output-size', default=31,
                    type=int, help='output Character')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of training epochs')
parser.add_argument('--save-folder', default='models/',
                    help='Location to save epoch models')
parser.add_argument('--save-every', default=1, type=int,
                    help='Save model every x steps')
parser.add_argument('--clip', default=50.0, type=float, help='Clip gradient')
parser.add_argument('--print-every', default=100,
                    type=int, help='print progress every')
parser.add_argument('--lambda-loss', default=0.8,
                    type=float, help='Lambda weight')
parser.add_argument('--batch-size', default=150,
                    type=int, help='Number of sample')
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MAX_LENGTH = 50

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable,
               encoder, encoder_optimizer, clip,
               batch_size, decoder, decoder_optimizer, mask, max_target_len):
    SOS_token = 29
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    loss = 0
    n_totals = 0
    use_teacher_forcing = True
    print_losses = []
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    del encoder_outputs
    del decoder_input
    del loss
    torch.cuda.empty_cache()
    return sum(print_losses) / n_totals


def trainIters(model_name, subset_training, voc, encoder, encoder_optimizer,
               encoder_n_layers, hidden_size, batch_size, num_workers,
               save_dir, n_iteration, print_every, save_every,
               clip, corpus_name, loadFilename,
               embedding, decoder, decoder_optimizer, decoder_n_layers):
    def iterator_subset(subset_training, voc):
        for subset in subset_training:
            set_train = Loader(subset[0], subset[1], voc)
            dataloader_train = set_train.get_loader(
                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
            yield set_train, dataloader_train
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    print('Read all dataloader from subsets')
    print('Number of subsets:'+str(len(subset_training)))
    dataloader_subsets = []
    for _, dataloader_train in iterator_subset(subset_training, voc):
        dataloader_subsets.append(dataloader_train)

    print('Training ...')
    for iteration in range(start_iteration, n_iteration + 1):
        for dataloader_train in dataloader_subsets:
            loss_avg = 0
            i = 1
            for training_batch in dataloader_train:
                keys, input_variable, lengths, target_variable, mask, max_target_len = training_batch
                mask = mask.cuda()
                input_variable = input_variable.cuda()
                lengths = lengths.cuda()
                target_variable = target_variable.cuda(non_blocking=True)
                loss = train(input_variable, lengths, target_variable,
                             encoder, encoder_optimizer, clip,
                             batch_size, decoder, decoder_optimizer, mask, max_target_len)
                loss_avg += loss
                i += 1
                if i % print_every == 0:
                    print_loss_avg = loss_avg/i
                    print("Iteration:{}, Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                        iteration, iteration/n_iteration * 100, print_loss_avg))
                del input_variable
                del lengths
                del target_variable
            loss_avg = loss_avg/i
            print("==============================")
            print("Subsetiter => Iteration:{}, Percent complete: {:.1f}%; Average loss: {:.4f}.format(
                iteration, iteration/n_iteration * 100, loss_avg))
            print("==============================")

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}_{}_{}_{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'embedding': embedding.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'voc_dict':voc.__dict__,
                'loss': loss_avg,
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    print("END Training")


if __name__ == "__main__":
    args = parser.parse_args()
    print("PYTORCH VERSION:"+torch.__version__)
    print("Seed to generators: "+str(args.seed))
    print("[INFO] Corpus")
    print("Corpus name: " + args.corpus_name)
    print("Label File: " + args.label_file)
    print("SCP File: " + args.scp_file)
    print("Nb worker for data-loading: " + str(args.num_workers))
    print("[INFO] EXP")
    print("Learning rate: " + str(args.lr))
    print("Number of training epochs: " + str(args.epochs))
    print("Number of sample per Batch: " + str(args.batch_size))
    print("Dim reduction : " + str(args.reduce))
    print("Model Information")
    print("Input size: " + str(args.input_size))
    print("Number of RNN layers: " + str(args.hidden_layers))
    print("Hidden size of GRU: " + str(args.hidden_size))
    print("Output: " + str(args.output_size))
    print("[INFO] Save Folder")
    print("Path: " + args.save_folder)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("Load Voc")
    voc = Voc(args.corpus_name)
    loadFilename = None
    print('Building encoder and decoder ...')
    voc.num_words = 31
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    dropout = 0.1

    encoder = EncoderRNN(args.input_size*args.reduce, args.hidden_size,
                             encoder_n_layers, (args.reduce != 1))

    decoder = DecoderRNN(embedding, args.hidden_size, voc.num_words, decoder_n_layers, dropout)

    if torch.cuda.device_count() > 1:
        print("Multi-GPU option activated")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    loadFilename = None
    encoder.train()
    decoder.train()
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    training_subsets = [[args.label_file, args.scp_file]]
    model_name = 'model_exp'
    print("Start Training: " + str(model_name))
    time_begin = datetime.datetime.now()
    print(time_begin)

    trainIters(model_name, training_subsets, voc, encoder, encoder_optimizer,
               encoder_n_layers, args.hidden_size, args.batch_size, args.num_workers,
               args.save_folder, args.epochs, args.print_every, args.save_every, args.clip, args.corpus_name, loadFilename,
               embedding, decoder, decoder_optimizer, decoder_n_layers)

    time_end = datetime.datetime.now()
    print(time_end-time_begin)


