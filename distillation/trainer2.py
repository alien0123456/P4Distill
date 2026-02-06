import time
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.model_rwi import *
from utils.early_stopping import *
from torch.utils.data import DataLoader
from utils.data_loader import FlowDataset
from utils.metric import metric_from_confuse_matrix
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight



def build_optimizer(args, model):
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    return optimizer, None


def build_data_loader(args, filename, batch_size, is_train=False, shuffle=True):
    dataset = FlowDataset(args, filename)
    data_loader = DataLoader(dataset, batch_size=batch_size, 
                             shuffle=shuffle)
    print('The size of {}_set is {}.'.format(
            'train' if is_train else 'test', len(dataset)))
    return data_loader


def batch2segs(args, batch):
    len_x_batch = []
    ipd_x_batch = []
    label_batch = []
    for i in range(len(batch[0])):
        label = batch[1][i]
        seqs = batch[0][i].split(';')
        len_seq = eval(seqs[0])
        ipd_seq = eval(seqs[1])
        flow_packets = len(len_seq)
        if flow_packets < args.window_size:
            raise Exception('Flow packets < window size!!!')
            # continue
        else:
            segs_idx = [idx for idx in range(0, flow_packets - args.window_size + 1)]
            batch_segs_idx = segs_idx

            for idx in batch_segs_idx:
                len_x_batch.append(len_seq[idx: idx + args.window_size])
                ipd_x_batch.append(ipd_seq[idx: idx + args.window_size])
                label_batch.append(label)

    len_x_batch = torch.LongTensor(len_x_batch)
    ipd_x_batch = torch.LongTensor(ipd_x_batch)
    label_batch = torch.tensor(label_batch)

    if args.gpu_id is not None:
        len_x_batch = len_x_batch.cuda(args.gpu_id)
        ipd_x_batch = ipd_x_batch.cuda(args.gpu_id)
        label_batch = label_batch.cuda(args.gpu_id)

    return len_x_batch, ipd_x_batch, label_batch

# def kl_divergence(teacher_logits, student_logits, T):
#     teacher_probs = torch.softmax(teacher_logits/T, dim=-1)
#     student_probs = torch.softmax(student_logits/T, dim=-1)
#     return F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

def kl_divergence(teacher_logits, student_logits, T):
    log_student = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(log_student, teacher_probs, reduction='batchmean') * (T * T)
    return loss



def save_checkpoint(output_dir, model_name, model, result_log):
    print('Saving model: {}'.format(output_dir + model_name))
    save_model(model, output_dir + model_name)
    with open(output_dir + model_name + '-result.txt', 'w') as fp:
        for line in result_log:
            # print(line)
            fp.writelines(line + '\n')

def get_loss(t_logits, s_logits, label, alpha, T, loss_type):

    task_loss = nn.CrossEntropyLoss()(s_logits, label)
    
    if loss_type == "KL":
        distillation_loss = kl_divergence(t_logits, s_logits, T)
        
    
    total_loss = alpha * task_loss + (1 - alpha) * distillation_loss
    
    return total_loss, distillation_loss, task_loss


def compute_kl_div(teacher_logits, student_logits, T=5):
    # T: temperature
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    q_student = F.log_softmax(student_logits / T, dim=1)
    kl = F.kl_div(q_student, p_teacher, reduction='batchmean') * (T * T)
    return kl.item()

def compute_cosine_sim(teacher_logits, student_logits):
    return F.cosine_similarity(teacher_logits, student_logits, dim=1).mean().item()

def compute_l2_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2)).item()

class STrainer(object):
    def __init__(self, args):
        
        self.current_epoch = 0
        self.total_epochs = args.total_epochs
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        
        self.labels_num = args.labels_num
        self.output_dir = args.output_dir

        self.loss_factor = args.loss_factor
        self.focal_loss_gamma = args.focal_loss_gamma
        self.loss_type = args.loss_type

        
    def forward_propagation(self, len_x_batch, ipd_x_batch, label_batch, model):
        """
        Compute task loss (CrossEntropy) and logits.
        Used only in validation/test, not in distillation training.
        """
        logits = model(len_x_batch, ipd_x_batch)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, label_batch)
        
        return loss, logits


    def Svalidate(self, args, test_loader, model):
        model.eval()

        test_samples = 0
        test_total_loss = 0

        conf_mat_test = np.zeros([args.labels_num, args.labels_num])
        with torch.no_grad():
            for batch in test_loader:
                len_x_batch, ipd_x_batch, label_batch = batch2segs(args, batch)
                loss, logits= self.forward_propagation(len_x_batch, ipd_x_batch, label_batch, model)
                
                test_samples += len_x_batch.shape[0]
                test_total_loss += loss.item() * len_x_batch.shape[0]
                
                pred = logits.max(dim=1, keepdim=True)[1]
                for i in range(len(pred)):
                    conf_mat_test[label_batch[i].cpu(), pred[i].cpu()] += 1
        
        return conf_mat_test, test_total_loss, test_samples
            
    

    def Strain(self, args, train_loader, test_loader, teacher_model, student_model, optimizer, scheduler):
        learning_curves = {
            'train_loss': [],
            'train_total_loss1':[],
            'train_total_loss2':[],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'test_loss': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
            'cosine_similarity':[],
            'KL_divergence':[],
            'L2_distance':[]
        }

        first_recs = 0
        
        early_stopping = EarlyStopping(patience=50, delta=0, verbose=False)
        while True:
            self.current_epoch += 1
            if self.current_epoch == self.total_epochs + 1:
                return
            start_time = time.time()
            
            # Train Student for an epoch
            student_model.train()
            teacher_model.eval()
            
            train_samples = 0
            train_total_loss = 0
            train_total_loss1 = 0
            train_total_loss2 = 0
            conf_mat_train = np.zeros([args.labels_num, args.labels_num])
            
            kl_all =[]
            cos_sim_all = []
            l2_dis_all = []
            
            for batch in train_loader:
                len_x_batch, ipd_x_batch, label_batch = batch2segs(args, batch)
               
                # Teacher Predict
                with torch.no_grad():  
                    teacher_logits = teacher_model(len_x_batch, ipd_x_batch)
                    
                # Student Model Forward Propagation
                student_logits = student_model(len_x_batch, ipd_x_batch)
                    
                loss, distillation_loss, task_loss = get_loss(teacher_logits, student_logits, label_batch, args.a, args.T, args.loss_type)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_samples += len_x_batch.shape[0]
                train_total_loss += loss.item() * len_x_batch.shape[0]
                train_total_loss1 += distillation_loss.item() * len_x_batch.shape[0]
                train_total_loss2 += task_loss.item() * len_x_batch.shape[0]
                
                pred = student_logits.max(dim=1, keepdim=True)[1]
                for i in range(len(pred)):
                    conf_mat_train[label_batch[i].cpu(), pred[i].cpu()] += 1

                kl_loss = compute_kl_div(teacher_logits, student_logits, T=args.T)
                cos_sim = compute_cosine_sim(teacher_logits, student_logits)
                l2_dis = compute_l2_distance(teacher_logits, student_logits)
                kl_all.append(kl_loss)
                cos_sim_all.append(cos_sim)
                l2_dis_all.append(l2_dis)
    
            conf_mat_test, test_total_loss, test_samples  = self.Svalidate(args, test_loader, student_model)
            
            # Report losses
            train_avg_loss = train_total_loss / train_samples
            train_avg_loss1 = train_total_loss1 / train_samples
            train_avg_loss2 = train_total_loss2 / train_samples
            test_avg_loss = test_total_loss / test_samples

            epoch_kl = np.mean(kl_all)
            epoch_cos_sim = np.mean(cos_sim_all)
            epoch_l2_dis = np.mean(l2_dis_all)           

            pres_test, recs_test, f1s_test, logs_test = metric_from_confuse_matrix(conf_mat_test)
            pres_train, recs_train, f1s_train, logs_train = metric_from_confuse_matrix(conf_mat_train)

            print("| {:5d}/{:5d} epochs ({:5.2f} s, lr {:8.5f})"
                    "| Train segs {:7d}, Test segs {:7d} "
                    "| Train loss {:7.2f} "
                    "| Train loss1 {:7.2f} "
                    "| Train loss2 {:7.2f} "
                    "| Train pres {:7.2f}"
                    "| Train recs {:7.2f}"
                    "| Test loss {:7.2f} "
                    "| Test pres {:7.2f}"
                    "| Test recs {:7.2f}"
                    "| cosine_similarity {:7.2f}"
                    "| KL_divergence {:7.2f}"
                    "| L2_distance {:7.2f}".format(
                    self.current_epoch, self.total_epochs, time.time() - start_time, optimizer.param_groups[0]['lr'],
                    train_samples, test_samples,
                    train_avg_loss, train_avg_loss1, train_avg_loss2,
                    np.mean(pres_train), np.mean(recs_train),
                    test_avg_loss,
                    np.mean(pres_test), np.mean(recs_test),
                    epoch_cos_sim,
                    epoch_kl,
                    epoch_l2_dis
                ))

                
            # Early Stopping
            status = early_stopping(test_total_loss / test_samples)
            if status == EARLY_STOP:
                return
            
            # Save model
            if status == BEST_SCORE_UPDATED or self.current_epoch % self.save_checkpoint_epochs == 0:
                pres_train, recs_train, f1s_train, logs_train = metric_from_confuse_matrix(conf_mat_train)
                pres_test, recs_test, f1s_test, logs_test = metric_from_confuse_matrix(conf_mat_test)
                logs = ['Training set: {} segs, average loss {}'.format(train_samples, train_avg_loss)]
                logs.extend(logs_train)
                logs.append('Testing set: {} segs, average loss {}'.format(test_samples, test_avg_loss))
                logs.extend(logs_test)
                if status == BEST_SCORE_UPDATED:
                    save_checkpoint(output_dir = self.output_dir, 
                                    model_name = 'student-best',
                                    model=student_model,
                                    result_log=logs)
                if self.current_epoch % self.save_checkpoint_epochs == 0:
                    save_checkpoint(output_dir = self.output_dir, 
                                    model_name = 'student-' + str(self.current_epoch),
                                    model=student_model,
                                    result_log=logs)
            
            learning_curves['train_loss'].append(train_avg_loss)
            learning_curves['train_total_loss1'].append(train_avg_loss1)
            learning_curves['train_total_loss2'].append(train_avg_loss2)
            learning_curves['test_loss'].append(test_avg_loss)
            learning_curves['cosine_similarity'].append(epoch_cos_sim)
            learning_curves['KL_divergence'].append(epoch_kl)
            learning_curves['L2_distance'].append(epoch_l2_dis)

            with open(self.output_dir + 'learning_curves.json', 'w') as fp:
                json.dump(learning_curves, fp, indent=1)
