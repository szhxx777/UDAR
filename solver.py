import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_factory.data_loader import get_loader_segment
from metrics.metrics import combine_all_evaluation_scores
from model.udar import UDAR
from utils.utils import *


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, run_id=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.run_id = run_id

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        checkpoint_name = f'{self.dataset}_run{self.run_id}_checkpoint.pth'
        torch.save(model.state_dict(), os.path.join(path, checkpoint_name))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.decomp_kernel = config.get('decomp_kernel', 25)
        self.period = config.get('period', 25)
        self.run_id = config.get('run_id', 0)

        self.train_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='train', dataset=self.dataset
        )
        self.vali_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='val', dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='test', dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='thre', dataset=self.dataset
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = UDAR(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=3,
            decomp_kernel=self.decomp_kernel,
            period=self.period,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for input_data, _ in vali_loader:
            input_tensor = input_data.float().to(self.device)
            final_rec, final_residual, residual_rec, series, prior, sigmas = self.model(input_tensor)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                    1, 1, 1, self.win_size
                )
                series_loss += (
                    torch.mean(my_kl_loss(series[u], normalized_prior.detach())) +
                    torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))
                )
                prior_loss += (
                    torch.mean(my_kl_loss(normalized_prior, series[u].detach())) +
                    torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
                )
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(residual_rec, final_residual)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("====================== TRAIN MODE (Run {}) ======================".format(self.run_id))

        time_now = time.time()
        path = f"{self.model_save_path}/{self.dataset}_run_id={self.run_id}"
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(
            patience=3,
            verbose=True,
            dataset_name=self.dataset,
            run_id=self.run_id,
        )
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_tensor = input_data.float().to(self.device)

                final_rec, final_residual, residual_rec, series, prior, sigmas = self.model(input_tensor)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )
                    series_loss += (
                        torch.mean(my_kl_loss(series[u], normalized_prior.detach())) +
                        torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))
                    )
                    prior_loss += (
                        torch.mean(my_kl_loss(normalized_prior, series[u].detach())) +
                        torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
                    )
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(residual_rec, final_residual)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        checkpoint_name = f'{self.dataset}_run{self.run_id}_checkpoint.pth'
        path = f"{self.model_save_path}/{self.dataset}_run_id={self.run_id}"
        self.model.load_state_dict(torch.load(os.path.join(str(path), checkpoint_name)))
        self.model.eval()
        temperature = 50

        print("====================== TEST MODE (Run {}) ======================".format(self.run_id))

        test_start_time = time.time()
        output_file = os.path.join(path, f"{self.dataset}_run{self.run_id}_test_results.txt")

        with open(output_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("UDAR TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")

            f.write("MODEL PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Window Size: {self.win_size}\n")
            f.write(f"Input Channels: {self.input_c}\n")
            f.write(f"Output Channels: {self.output_c}\n")
            f.write(f"Decomposition Kernel: {self.decomp_kernel}\n")
            f.write(f"Period: {self.period}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Learning Rate: {self.lr}\n")
            f.write(f"Anomaly Ratio: {self.anormly_ratio}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Test Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_start_time))}\n")
            f.write("\n")

        criterion = nn.MSELoss(reduction='none')

        attens_energy = []
        with torch.no_grad():
            for input_data, _ in self.train_loader:
                input_tensor = input_data.float().to(self.device)
                final_rec, final_residual, residual_rec, series, prior, sigmas = self.model(input_tensor)

                loss = torch.mean(criterion(final_residual, residual_rec), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )
                    if u == 0:
                        series_loss = my_kl_loss(series[u], normalized_prior.detach()) * temperature
                        prior_loss = my_kl_loss(normalized_prior, series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], normalized_prior.detach()) * temperature
                        prior_loss += my_kl_loss(normalized_prior, series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for input_data, labels in self.thre_loader:
                input_tensor = input_data.float().to(self.device)
                final_rec, final_residual, residual_rec, series, prior, sigmas = self.model(input_tensor)

                loss = torch.mean(criterion(final_residual, residual_rec), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )
                    if u == 0:
                        series_loss = my_kl_loss(series[u], normalized_prior.detach()) * temperature
                        prior_loss = my_kl_loss(normalized_prior, series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], normalized_prior.detach()) * temperature
                        prior_loss += my_kl_loss(normalized_prior, series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred (adjusted): ", pred.shape)
        print("gt:   ", gt.shape)

        print("\n--- All Metrics Evaluation (Run {}) ---".format(self.run_id))
        all_metrics = combine_all_evaluation_scores(pred, gt)

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time

        with open(output_file, 'a') as f:
            f.write("TEST RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Threshold: {thresh:.6f}\n")
            f.write(f"Prediction shape: {pred.shape}\n")
            f.write(f"Ground truth shape: {gt.shape}\n\n")

            f.write("DETAILED METRICS:\n")
            f.write("-" * 30 + "\n")
            for k, v in sorted(all_metrics.items()):
                metric_line = "{}: {:0.4f}".format(k, v)
                print(metric_line)
                f.write(metric_line + "\n")
            f.write("\n")

            f.write("SELECTED CORE METRICS:\n")
            f.write("-" * 30 + "\n")
            aff_f = all_metrics.get("affiliation_F1", 0.0)
            pa_f = all_metrics.get("pa_f_score", 0.0)
            a_roc = all_metrics.get("AUC_ROC", 0.0)
            a_pr = all_metrics.get("AUC-PR", 0.0)

            core_metrics = [
                ("Aff-F (Affiliation F1)", aff_f),
                ("PA-F (Point-Adjusted F1)", pa_f),
                ("A-ROC (AUC-ROC)", a_roc),
                ("A-PR (AUC-PR)", a_pr)
            ]

            core_metrics_to_return = {
                "affiliation_F1": aff_f,
                "pa_f_score": pa_f,
                "AUC_ROC": a_roc,
                "AUC-PR": a_pr,
            }

            for name, value in core_metrics:
                metric_line = "{} : {:0.4f}".format(name, value)
                print(metric_line)
                f.write(metric_line + "\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("TEST COMPLETION INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Test End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_end_time))}\n")
            f.write(f"Test Duration: {test_duration:.2f} seconds\n")
            f.write(f"Results saved to: {output_file}\n")
            f.write("=" * 50 + "\n")

        print("\n" + "=" * 50)
        print(f"Test completed in {test_duration:.2f} seconds")
        print(f"Results saved to: {output_file}")
        print("=" * 50)

        return all_metrics, core_metrics_to_return
