import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import os
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

import model
import lossfunction

class Trainer():
    def __init__(self, GEN: model.GraphEmbeddingNetwork, GENdataLoader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-6, betas=(0.9, 0.999), with_cuda=True, cuda_devices=None, log_freq: int = 5, lossFunction = lossfunction.CosSimLoss()):
        # test if cuda could be used
        cuda_condition = torch.cuda.is_available() and with_cuda
        print(f'CUDA available: {cuda_condition}')
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # upload model to device
        self.model = GEN.to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # store dataset
        self.train_data = GENdataLoader
        self.test_data = test_dataloader

        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas)
        self.criterion = lossFunction
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, batch_size=10):
        # Setting the tqdm progress bar
        total = len(self.train_data)
        update_interval = max(1, total // 10)  # 每1%更新一次，确保至少为1

        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP_%s:%d" % ("train", epoch),
                              total=total,
                              bar_format="{l_bar}{r_bar}",
                              leave=True)
        epoch_loss = []

        for i, data in data_iter:
            if isinstance(data, dict):
                data = {key: value.to(self.device) for key, value in data.items()}
            else:
                print(f"Unexpected data format at index {i}: {data}")
                continue

            tensor_u1 = torch.zeros(batch_size, data["adj_tensor1"].size(1), self.model.embedding_size).to(self.device)
            tensor_u2 = torch.zeros(batch_size, data["adj_tensor2"].size(1), self.model.embedding_size).to(self.device)

            attribute_vector1 = self.model.forward(data["attr_tensor1"], data["adj_tensor1"], tensor_u1)
            attribute_vector2 = self.model.forward(data["attr_tensor2"], data["adj_tensor2"], tensor_u2)
            loss = self.criterion.forward(attribute_vector1, attribute_vector2, data["label"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss.append(loss.item())
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss:": loss.item()
            }
            if i % update_interval == 0:
                data_iter.set_postfix(post_fix)
                data_iter.update(update_interval - (i % update_interval))

        # 确保进度条在处理完所有数据后显示 100%
        if total % update_interval != 0:
            data_iter.update(total % update_interval)

        return epoch_loss

    def save(self, epoch, file_path="output/parameter.model"):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def test(self, epoch, batch_size=10, num_tests=1):
        all_epoch_losses = []
        all_recalls = []
        all_accuracies = []
        all_precisions = []
        all_f1s = []

        for _ in range(num_tests):
            total = len(self.test_data)
            update_interval = max(1, total // 10)  # 每1%更新一次，确保至少为1

            data_iter = tqdm.tqdm(enumerate(self.test_data),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=total,
                                  bar_format="{l_bar}{r_bar}",
                                  leave=True)
            epoch_loss = []
            all_labels = []
            all_predictions = []

            for i, data in data_iter:
                if isinstance(data, dict):
                    data = {key: value.to(self.device) for key, value in data.items()}
                else:
                    print(f"Unexpected data format at index {i}: {data}")
                    continue

                tensor_u1 = torch.zeros(batch_size, data["adj_tensor1"].size(1), self.model.embedding_size).to(
                    self.device)
                tensor_u2 = torch.zeros(batch_size, data["adj_tensor2"].size(1), self.model.embedding_size).to(
                    self.device)

                attribute_vector1 = self.model.forward(data["attr_tensor1"], data["adj_tensor1"], tensor_u1)
                attribute_vector2 = self.model.forward(data["attr_tensor2"], data["adj_tensor2"], tensor_u2)
                loss = self.criterion.forward(attribute_vector1, attribute_vector2, data["label"])

                epoch_loss.append(loss.item())

                # 确保 predictions 和 labels 数量一致
                predictions = torch.argmax(attribute_vector1, dim=1)
                labels = data["label"].cpu().numpy()

                if len(labels) != len(predictions):
                    print(f"Mismatch in number of labels and predictions at batch {i}")
                    continue

                all_labels.extend(labels)
                all_predictions.extend(predictions.cpu().numpy())

                recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
                accuracy = accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "loss": loss.item(),
                    "recall": recall,
                    "accuracy": accuracy,
                    "precision": precision,
                    "f1": f1
                }
                if i % update_interval == 0:
                    data_iter.set_postfix(post_fix)
                    data_iter.update(update_interval - (i % update_interval))

            # 确保进度条在处理完所有数据后显示 100%
            if total % update_interval != 0:
                data_iter.update(total % update_interval)

            all_epoch_losses.append(np.mean(epoch_loss))
            all_recalls.append(recall)
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_f1s.append(f1)

        # Average metrics over all tests
        avg_loss = np.mean(all_epoch_losses)
        avg_recall = np.mean(all_recalls)
        avg_accuracy = np.mean(all_accuracies)
        avg_precision = np.mean(all_precisions)
        avg_f1 = np.mean(all_f1s)

        print(
            f"Test Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Recall: {avg_recall:.4f}, Avg Accuracy: {avg_accuracy:.4f}, Avg Precision: {avg_precision:.4f}, Avg F1: {avg_f1:.4f}")

        return avg_loss, avg_recall, avg_accuracy, avg_precision, avg_f1

    def load(self, file_path):
        """
        Loading the model from file_path

        :param file_path: model input path
        :return: None
        """
        if os.path.exists(file_path):
            self.model = torch.load(file_path)
            self.model.to(self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
            print("Model Loaded from:", file_path)
        else:
            raise FileNotFoundError(f"No model found at {file_path}")
