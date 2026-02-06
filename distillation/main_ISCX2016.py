import os
import time
import json
import argparse

from opts_iscx2016 import *

from model.model_BRNN import BinaryRNN
from model.Binary.model_BLSTM import BinaryLSTM
from model.Binary.model_BL3LSTM import BinaryL3LSTM
from model.Binary.model_BATLSTM import BinaryLSTMWithAttention
from model.Binary.model_BBiATLSTM import BiLSTMWithAttention
from model.Binary.model_BBi2ATLSTM import BiLSTM2WithAttention

from utils.model_rwi import *
from utils.seed import set_seed
from trainer2 import build_optimizer, build_data_loader, STrainer

def load_model(model, model_path):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model


'''
python main_ISCX2016.py --teacher_model=BinaryLSTM
python main_ISCX2016.py --teacher_model=BinaryLSTMWithAttention
python main_ISCX2016.py --teacher_model=BinaryLSTM
python main_ISCX2016.py --teacher_model=BinaryLSTM
python main_ISCX2016.py --teacher_model=BinaryLSTM
'''


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument("--dataset", default="ISCXVPN2016",
                        choices=["ISCXVPN2016", "BOTIOT", "CICIOT2022", "PeerRush"])
    # Teacher model selection.
    parser.add_argument("--teacher_model", default="BinaryLSTM",
                        choices=["BinaryLSTM", "BinaryLSTMWithAttention", "BinaryL3LSTM",
                                "BiLSTMWithAttention","BiLSTM2WithAttention"])
    parser.add_argument("--loss_type", default="KL",
                        choices=["KL"])


    # Student model options.
    model_opts(parser)
    # Training options (shared between teacher and student for a dataset).
    training_opts(parser)
    args = parser.parse_args()
                
   
    # Set dataset & model path according to options
    args.train_path = '../dataset/{}/json/train.json'.format(args.dataset)
    args.test_path = '../dataset/{}/json/test.json'.format(args.dataset)
    args.output_dir = './save/{}/{}/{}/studentbrnn_len{}_ipd{}_ev{}_hidden{}_{}_T{}_a{}/'.format(
        args.dataset,
        args.teacher_model,
        args.loss_type,
        args.len_embedding_bits, 
        args.ipd_embedding_bits, 
        args.embedding_vector_bits, 
        args.rnn_hidden_bits,
        str(args.loss_factor) + '_' + str(args.focal_loss_gamma) + '_' + args.loss_type + '_' + str(args.learning_rate),
        args.T,
        args.a)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    # print(args.output_dir)
    with open('../dataset/{}/json/statistics.json'.format(args.dataset)) as fp:
        statistics = json.load(fp)
        args.labels_num = statistics['label_num']

        class_weights = [1] * args.labels_num
        args.class_weights = class_weights
        # print('class weights: {}'.format(class_weights))

    set_seed(args.seed)

    args.teacher_bestmodel_path = 'teacher_bm_path/{}/{}/teacher-brnn-best'.format(
        args.dataset,
        args.teacher_model)
    
    if args.teacher_model == "BinaryLSTM":
        teacher_model = BinaryLSTM(args)
    elif args.teacher_model == "BinaryL3LSTM":
        teacher_model = BinaryL3LSTM(args)
    elif args.teacher_model == "BinaryLSTMWithAttention":
        teacher_model = BinaryLSTMWithAttention(args)
    elif args.teacher_model == "BiLSTM2WithAttention":
        teacher_model = BiLSTM2WithAttention(args)
    elif args.teacher_model == "BiLSTMWithAttention":
        teacher_model = BiLSTMWithAttention(args)
    else:
        print("check model name, it's wrong", args.teacher_model)
        
    # Load the best teacher checkpoint.

    best_model_path = args.teacher_bestmodel_path
    
    load_model(teacher_model, best_model_path)
    
    teacher_model.eval()
    
    # Student Model
    student_model = BinaryRNN(args)
    initialize_parameters(args, student_model)

    optimizer, scheduler = build_optimizer(args, student_model)
    
    gpu_id = args.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        student_model.cuda(gpu_id)
        teacher_model.cuda(gpu_id)
        print("Using GPU %d for training." % args.gpu_id)
    else:
        print("Using CPU mode for training.")

    train_loader = build_data_loader(args, args.train_path, args.batch_size, is_train=True, shuffle=True)
    test_loader = build_data_loader(args, args.test_path, args.batch_size, is_train=False, shuffle=True)
    

    trainer = STrainer(args)
    trainer.Strain(args, train_loader, test_loader, teacher_model, student_model, optimizer, scheduler)
    
    


