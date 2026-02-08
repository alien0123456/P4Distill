import os
import json
import argparse
import torch

from distillation.opts import model_opts, training_opts
from distillation.opts import distill_opts

from distillation.model.sturent_BRNN_from_BOS import BinaryRNN
from distillation.model.teacher_BLSTM import BinaryLSTM
from distillation.model.teacher_BL3LSTM import BinaryL3LSTM
from distillation.model.teacher_BATLSTM import BinaryLSTMWithAttention
from distillation.model.teacher_BBiATLSTM import BiLSTMWithAttention
from distillation.model.teacher_BBi2ATLSTM import BiLSTM2WithAttention

from distillation.utils.checkpoint import initialize_parameters, load_model
from distillation.utils.seed import set_seed, build_generator, seed_worker
from distillation.trainers.student_trainer import DistillTrainer
from distillation.trainers.teacher_trainer import build_optimizer
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
    if model_name == "BiLSTM2WithAttention":
        return BiLSTM2WithAttention(args)
    if model_name == "BiLSTMWithAttention":
        return BiLSTMWithAttention(args)
    raise ValueError("Unknown model: {}".format(model_name))


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument("--dataset", default="ISCXVPN2016")
    parser.add_argument("--teacher_model", default="BiLSTMWithAttention",
                        choices=["BinaryRNN", "BinaryLSTM", "BinaryL3LSTM", "BinaryLSTMWithAttention",
                                 "BiLSTM2WithAttention","BiLSTMWithAttention"])
    
    # Model options
    model_opts(parser)
    # Training options
    training_opts(parser)
    # Distillation options
    distill_opts(parser)

    args = parser.parse_args()

    # Set dataset & model path according to options
    args.train_path = './dataset/{}/json/train.json'.format(args.dataset)
    args.test_path = './dataset/{}/json/test.json'.format(args.dataset)

    args.output_dir = './save_kd/{}/T_{}_/S_BRNN/len{}_ipd{}_ev{}_hidden{}_/kd_a{}_t{}_lr{}/'.format(
        args.dataset, args.teacher_model,
        args.pkt_len_embed_bits, args.ipd_embed_bits, args.embed_dim_bits, args.rnn_hidden_state_bits,
        args.kd_alpha, args.kd_temperature, args.lr
    )
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
    dl_generator = build_generator(args.random_seed)

    # Build student (binary RNN)
    student = BinaryRNN(args)
    initialize_parameters(args, student)
    init_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_states")
    os.makedirs(init_dir, exist_ok=True)
    init_path = os.path.join(init_dir, f"BRNN_{args.dataset}_seed{args.random_seed}.pt")
    if os.path.exists(init_path):
        student.load_state_dict(torch.load(init_path, map_location="cpu"))
    else:
        torch.save(student.state_dict(), init_path)
 
    # Build teacher (variable model)
    teacher = build_model(args, args.teacher_model)
    load_model(teacher, args.teacher_ckpt_path)

    # Build optimizer (student only)
    optimizer, scheduler = build_optimizer(args, student)

    # Assign gpu
    gpu_id = args.cuda_device_id
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        student.cuda(gpu_id)
        teacher.cuda(gpu_id)
        print("Using GPU %d for distillation." % args.cuda_device_id)
    else:
        print("Using CPU mode for distillation.")

    # Build data loader
    train_loader = build_data_loader(
        args, args.train_path, args.train_batch_size,
        is_train=True, shuffle=True,
        generator=dl_generator, worker_init_fn=seed_worker
    )
    test_loader = build_data_loader(
        args, args.test_path, args.train_batch_size,
        is_train=False, shuffle=True,
        generator=dl_generator, worker_init_fn=seed_worker
    )

    trainer = DistillTrainer(args)
    trainer.train(args, train_loader, test_loader, student, teacher, optimizer)


if __name__ == "__main__":
    main()

