import os
import time
import json

import argparse
from distillation.opts import *
from distillation.model.sturent_BRNN_from_BOS import BinaryRNN
from distillation.model.teacher_BLSTM import BinaryLSTM
from distillation.model.teacher_BL3LSTM import BinaryL3LSTM
from distillation.model.teacher_BATLSTM import BinaryLSTMWithAttention
from distillation.model.teacher_BBiATLSTM import BiLSTMWithAttention
from distillation.model.teacher_BBi2ATLSTM import BiLSTM2WithAttention
from distillation.utils.checkpoint import *
from distillation.utils.seed import set_seed
from distillation.trainers.teacher_trainer import build_optimizer, Teacher_Trainer
from distillation.utils.data_loader import build_data_loader

def build_model(args, model_name):
    if model_name == "BinaryRNN":
        return BinaryRNN(args)
    if model_name == "BinaryLSTM":
        return BinaryLSTM(args)
    if model_name == "BinaryL3LSTM":
        return BinaryL3LSTM(args)
    if model_name == "BinaryLSTMWithAttention":
        return BinaryLSTMWithAttention(args)
    if model_name == "BiLSTMWithAttention":
        return BiLSTMWithAttention(args)
    if model_name == "BiLSTM2WithAttention":
        return BiLSTM2WithAttention(args)
    raise ValueError("Unknown model: {}".format(model_name))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument("--dataset", default="ISCXVPN2016")
    parser.add_argument("--model_name", default="BiLSTMWithAttention",
                        choices=["BinaryRNN", "BinaryLSTM", "BinaryL3LSTM", "BinaryLSTMWithAttention",
                                 "BiLSTM2WithAttention","BiLSTMWithAttention"])
        
    # Model options
    model_opts(parser)
    # Training options
    training_opts(parser)
    
    args = parser.parse_args()
    # Set dataset & model path according to options
    args.train_path = './dataset/{}/json/train.json'.format(args.dataset)
    args.test_path = './dataset/{}/json/test.json'.format(args.dataset)
    args.output_dir = './save/{}/{}//brnn_len{}_ipd{}_ev{}_hidden{}_{}/'.format(
        args.dataset, args.model_name,
        args.pkt_len_embed_bits, args.ipd_embed_bits, args.embed_dim_bits, args.rnn_hidden_state_bits,
        str(args.ce_loss_weight) + '_' + str(args.focal_gamma) + '_' + str(args.lr))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(args.output_dir)
    with open('./dataset/{}/json/statistics.json'.format(args.dataset)) as fp:
        statistics = json.load(fp)
        args.num_classes = statistics['label_num']

        class_weights = [1] * args.num_classes
        args.class_weights = class_weights
        print('class weights: {}'.format(class_weights))
    
    set_seed(args.random_seed)

    # Build the binary RNN model & initialize parameters
    model = build_model(args, args.model_name)
    initialize_parameters(args, model)

    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer(args, model)

    # Assign gpu
    gpu_id = args.cuda_device_id
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)
        print("Using GPU %d for training." % args.cuda_device_id)
    else:
        print("Using CPU mode for training.")

    # Build data loader
    train_loader = build_data_loader(args, args.train_path, args.train_batch_size, is_train=True, shuffle=True)
    test_loader  = build_data_loader(args, args.test_path,  args.train_batch_size, is_train=False, shuffle=False)

    trainer = Teacher_Trainer(args)
    trainer.train(args, train_loader, test_loader, model, optimizer)
    

if __name__ == "__main__":
    main()

