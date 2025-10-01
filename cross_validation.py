import numpy as np
import datetime
import os
import csv
import h5py
import copy as cp
import os.path as osp
from train_model import *
from utils import Averager
from sklearn.model_selection import KFold
import os
from xlrd import open_workbook
from xlutils.copy import copy
from xlwt import Workbook

ROOT = os.getcwd()

import networks
from utils import *
from eeg_dataset import *
from torch.utils.data import DataLoader
import random
from scipy.io import loadmat

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        self.log_file = "results.txt"

    def load_per_subject_seediv(self, sub,band):
        np.random.seed(0)
        X = np.empty([0, 62, 5])
        y = np.empty([0, 1])
        j = sub
        session_label1=[[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],[2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],[1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]

        for i in range(3):

            EEGPath = './SEED-IV/eeg_feature_smooth/' + str(i + 1) + '/' + str(j) + '.mat'
            print('processing {}'.format(EEGPath))
            # every subject DE feature
            subject_data = loadmat(EEGPath)
            trial_data_list = []
            trial_label_list = []
            session_label = session_label1[i]
            for k in range(24):  # 这里假设循环5次，可以根据需要修改
                key = "de_LDS" + str(k + 1)

                trial_data = subject_data[key].swapaxes(0, 1)  # 调换维度并按行拼接
                trial_data_list.append(trial_data)
                label = session_label[k]
                trial_label_list.extend([label] * trial_data.shape[0])

                # 按行拼接数据列表中的所有矩阵
            DE_feature = np.concatenate(trial_data_list, axis=0)
            label = np.array(trial_label_list).reshape(-1, 1)

            
            X = np.vstack([X, DE_feature])
            y = np.vstack([y, label])
        numbers = list(range(0, len(X)))
        random.shuffle(numbers)
        numbers = np.array(numbers)
        data = X[numbers]

        label = y[numbers]
        new_array = np.zeros_like(data)
        new_array[:,:,band]=data[:,:,band]
        print('>>> Data:{} Label:{}'.format(new_array.shape, label.shape))
        return new_array, label

    
    def load_per_subject_seed(self, sub,band):
        np.random.seed(0)
        X = np.empty([0, 62, 5])
        y = np.empty([0, 1])

        directory = '/root/ExtractedFeatures'

        files = [f for f in os.listdir(directory) if f.endswith('.mat')]

        # Dictionary to store grouped files
        subject_files = {}

        # Group files by subject ID
        for file in files:
            if file == 'label.mat':
                continue
            subject_id = file.split('_')[0]
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file)
        files = subject_files[str(sub)]
        session_label=[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        for file in files:
            EEGPath = directory + '/' + file
            print('processing {}'.format(EEGPath))

            subject_data = loadmat(EEGPath)
            trial_data_list = []
            trial_label_list = []

            for k in range(15):  # 这里假设循环5次，可以根据需要修改
                key = "de_LDS" + str(k + 1)

                trial_data = subject_data[key].swapaxes(0, 1)  # 调换维度并按行拼接
                trial_data_list.append(trial_data)
                label = session_label[k]
                trial_label_list.extend([label] * trial_data.shape[0])

                # 按行拼接数据列表中的所有矩阵
            DE_feature = np.concatenate(trial_data_list, axis=0)
            label = np.array(trial_label_list).reshape(-1, 1)

            
            X = np.vstack([X, DE_feature])
            y = np.vstack([y, label])
        numbers = list(range(0, len(X)))
        random.shuffle(numbers)
        numbers = np.array(numbers)
        data = X[numbers]

        label = y[numbers]
        new_array = np.zeros_like(data)
        new_array[:,:,band]=data[:,:,band]
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return new_array, label

    
    def prepare_data(self, idx_train, idx_test, data, label):
        
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        data_train, data_test = self.normalize(
            train=data_train, test=data_test)

        # Prepare the data format for training the model
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def prepare_data1(self, data_train, data_test, label_train, label_test,band):
        
        data_train, data_test = self.normalize(
            train=data_train, test=data_test,band=band)

        # Prepare the data format for training the model
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test,band):
        
        mean = 0
        std = 0
        for bands in band:
            mean = np.mean(train[:, :, bands])
            std = np.std(train[:, :, bands])
            train[:, :, bands] = (train[:, :, bands] - mean) / std
            test[:, :, bands] = (test[:, :, bands] - mean) / std
        return train, test

    

    def split_balance_class1(self, data, label, batch):
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1

        train = data
        train_label = label
        data_train = [train[i:i + batch] for i in range(0, len(train), batch)]
        data_label = [train_label[i:i + batch]
                      for i in range(0, len(train_label), batch)]
        return data_train, data_label
    
    
    

    def domain_generalization_train_experts(self, subject, domain_num, fold=5, reproduce=False):
        subjects = [i for i in range(1, domain_num + 2)]
        
        m = 0
        if self.args.bands=='allband':
            band=[0,1,2,3,4]
        else:
            band=int(self.args.bands[-1])-1
            band=[band]
        if self.args.dataset=='seed':
            ephoc_split=1697
            
            data_test, label_test = self.load_per_subject_seed(subject,band)
            epoch_split = int((10182)/ 1697)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seed(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seed(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        if self.args.dataset=='seediv':
            ephoc_split=167
            data_test, label_test = self.load_per_subject_seediv(subject,band)
            
            epoch_split = int((2505+ 0)/ 167)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seediv(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seediv(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        #data = np.array(data)
        #label = np.array(label)
        data_train = []
        data_label = []
        
        
        for patch in range(epoch_split):
            for domain in range(domain_num):
                
                data_train.append(data[domain * epoch_split + patch].reshape(-1, 62, 5))
                data_label.append(label[domain * epoch_split + patch])

        data_train = np.array(data_train).reshape(-1, 62, 5)
        data_label = np.array(data_label).reshape(-1)

        data = data_train
        label = data_label
        data_test = np.array(data_test)
        label_test = np.array(label_test)

        sub = subject
        data_train, label_train, data_test, label_test = self.prepare_data1(
            data_train=data, data_test=data_test, label_train=label, label_test=label_test,band=band)
        data_train = dt2_mapping(data_train,region=self.args.region)
        data_test = dt2_mapping(data_test,region=self.args.region)

        max_val_acc, max_f1_val = train_experts(args=self.args,
                                                data_train=data_train,
                                                label_train=label_train,
                                                data_val=data_test,
                                                label_val=label_test,
                                                subject=sub,
                                                fold=1,
                                                domain_num=domain_num
                                                )
        content = [max_val_acc, max_f1_val]
        self.log2txt(sub, content)
    def cross_sub_train_model(self, subject, domain_num, fold=5, reproduce=False):
        subjects = [i for i in range(1, domain_num + 2)]
        
        m = 0
        if self.args.bands=='allband':
            band=[0,1,2,3,4]
        else:
            band=int(self.args.bands[-1])-1
            band=[band]
        if self.args.dataset=='seed':
            ephoc_split=1697
            
            data_test, label_test = self.load_per_subject_seed(subject,band)
            epoch_split = int((10182)/ 1697)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seed(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seed(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        if self.args.dataset=='seediv':
            ephoc_split=167
            data_test, label_test = self.load_per_subject_seediv(subject,band)
            
            epoch_split = int((2505+ 0)/ 167)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seediv(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seediv(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        #data = np.array(data)
        #label = np.array(label)
        data_train = []
        data_label = []

        
        for patch in range(epoch_split):
            for domain in range(domain_num):
                data_train.append(data[domain * epoch_split + patch].reshape(-1, 62, 5))
                data_label.append(label[domain * epoch_split + patch])

        data_train = np.array(data_train).reshape(-1, 62, 5)
        data_label = np.array(data_label).reshape(-1)
        data = data_train
        label = data_label
        data_test = np.array(data_test)
        label_test = np.array(label_test)
        
        sub = subject
        data_train, label_train, data_test, label_test = self.prepare_data1(
            data_train=data, data_test=data_test, label_train=label, label_test=label_test,band=band)
        data_train = dt2_mapping(data_train,region=self.args.region)
        data_test = dt2_mapping(data_test,region=self.args.region)


        max_val_acc, max_f1_val = train_models(args=self.args,
                                                data_train=data_train,
                                                label_train=label_train,
                                                data_val=data_test,
                                                label_val=label_test,
                                                subject=sub,
                                                fold=1,
                                                domain_num=domain_num
                                                )
        content = [max_val_acc, max_f1_val]
        self.log2txt(sub, content)

    def attention_based_generalization(self, subject, domain_num, fold=5, reproduce=False):
        subjects = [i for i in range(1, domain_num + 2)]
        
        m = 0
        if self.args.bands=='allband':
            band=[0,1,2,3,4]
        else:
            
            band=int(self.args.bands[-1])-1
            band=[band]
        if self.args.dataset=='seed':
            ephoc_split=1697
            
            data_test, label_test = self.load_per_subject_seed(subject,band)
            epoch_split = int((10182)/ 1697)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seed(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seed(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        if self.args.dataset=='seediv':
            ephoc_split=167
            data_test, label_test = self.load_per_subject_seediv(subject,band)
            
            epoch_split = int((2505+ 0)/ 167)
            for sub in subjects:
                if sub != subject and m == 1:
                    data1, label1 = self.load_per_subject_seediv(sub,band)
                    data1, label1 = self.split_balance_class1(
                        data=data1, label=label1, batch=ephoc_split)
                    data = np.concatenate([data, data1], axis=0)
                    label = np.concatenate([label, label1], axis=0)
                elif sub != subject:
                    m = m + 1
                    data, label = self.load_per_subject_seediv(sub,band)
                    data, label = self.split_balance_class1(
                        data=data, label=label, batch=ephoc_split)

        #data = np.array(data)
        #label = np.array(label)
        data_train = []
        data_label = []

        
        for patch in range(epoch_split):
            for domain in range(domain_num):
                
                data_train.append(data[domain * epoch_split + patch].reshape(-1, 62, 5))
                data_label.append(label[domain * epoch_split + patch])

        data_train = np.array(data_train).reshape(-1, 62, 5)
        data_label = np.array(data_label).reshape(-1)

        data = data_train
        label = data_label
        data_test = np.array(data_test)
        label_test = np.array(label_test)

        sub = subject
        data_train, label_train, data_test, label_test = self.prepare_data1(
            data_train=data, data_test=data_test, label_train=label, label_test=label_test,band=band)
        data_train = dt2_mapping(data_train,region=self.args.region)
        data_test = dt2_mapping(data_test,region=self.args.region)
        max_val_acc, max_f1_val = train(args=self.args,
                                        data_train=data_train,
                                        label_train=label_train,
                                        data_val=data_test,
                                        label_val=label_test,
                                        subject=sub,
                                        fold=1,
                                        domain_num=domain_num
                                        )
        content = [max_val_acc, max_f1_val]
        self.log2txt(sub, content)

    
    def file_name(file_dir="./save/acc_result/"):
        for root, dirs, files in os.walk(file_dir):
            print(files)
        return files

    def log2txt(self, sub, content):
        files = os.listdir("./")

        save_file_name = self.args.dataset+ "_" +self.args.region + "_" + self.args.bands + ".xls"
        if save_file_name not in files:
            book = Workbook(encoding='utf-8')
            sheet1 = book.add_sheet('result')

            sheet1.write(0, 0, "sub")
            sheet1.write(0, 1, "DRM_acc")  # 交叉验证最大acc
            sheet1.write(0, 2, "DRM_f1")
            sheet1.write(0, 3, "Prompt_DRM_acc")  # 交叉验证平均acc
            sheet1.write(0, 4, "Prompt_DRM_f1")  # 交叉验证平均acc_std
            sheet1.write(0, 5, "acc")  # 最大验证准确
            sheet1.write(0, 6, "f1")  # 最大验证f1

            for i in range(1, 16):
                sheet1.write(i, 0, "s" + "%02d" % i)
            # 保存Excel book.save('path/文件名称.xls')
            book.save("./" + save_file_name)
        rexcel = open_workbook(
            "./" + save_file_name)  # 用wlrd提供的方法读取一个excel文件
        excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
        table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

        if self.args.sub_dependent_or_not == 'cross_sub_train_model':
            table.write(sub, 1, content[0])
            table.write(sub, 2, content[1])
        if self.args.sub_dependent_or_not == 'cross_sub_train_experts':
            table.write(sub, 3, content[0])
            table.write(sub, 4, content[1])
        if self.args.sub_dependent_or_not == 'cross_sub_attention_generalization':
            
            table.write(sub, 5, content[0])
            table.write(sub, 6, content[1])
        excel.save("./" + save_file_name)
        
    def log2txt1(self, sub, content):
        files = os.listdir("./")

        save_file_name = self.args.bands + ".xls"
        if save_file_name not in files:
            book = Workbook(encoding='utf-8')
            sheet1 = book.add_sheet('result')

            sheet1.write(0, 0, "avg")
            sheet1.write(1, 0, "band1")  
            sheet1.write(2, 0, "band2")
            sheet1.write(3, 0, "band3")  
            sheet1.write(4, 0, "band4")
            sheet1.write(5, 0, "band5")
            sheet1.write(0, 1, "frontal(6)")  
            sheet1.write(0, 2, "frontal(10)")
            sheet1.write(0, 3, "temporal(6)")  
            sheet1.write(0, 4, "temporal(10)")
            
            
            book.save("./" + save_file_name)
        rexcel = open_workbook(
            "./" + save_file_name)  # 用wlrd提供的方法读取一个excel文件
        excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
        table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

        band=int(self.args.bands[-1])
        regions=['frontal6','frontal10','temporal6','temporal10']
        for i in range(4):
            if self.region==regions[i]:
                region=i+1
            
        
        table.write(band, region, content)
        
       
        excel.save("./" + save_file_name)

def dt2_mapping(data,region):
    # data = np.load("./processedData/data_3d.npy")
    # print("data.shape: ", data.shape)  # [325680, 17, 5] -> [325680, 6, 9, 5]

    img_rows, img_cols, num_chan = 9, 9, 5
    # [samples, height, width, channels]->[325680, 6, 9, 5]
    data_4d = np.zeros((len(data), img_rows, img_cols, num_chan))
    print("data_4d.shape :", data_4d.shape)

    # 2D map for 17 channels
    # 'FT7'(channel1) :
    if region=='allregion':
        data_4d[:, 0, 2, :] = data[:, 0, :]
        # 'FT8'(channel2) :
        data_4d[:, 0, 4, :] = data[:, 1, :]
        # 'T7' (channel3) :
        data_4d[:, 0, 6, :] = data[:, 2, :]
        # 'T8' (channel4) :
        data_4d[:, 1, 2, :] = data[:, 3, :]
        # 'TP7'(channel5) :
        data_4d[:, 1, 6, :] = data[:, 4, :]
        # 'TP8'(channel6) :
        data_4d[:, 2, 0, :] = data[:, 5, :]
        # 'CP1'(channel7) :
        data_4d[:, 2, 1, :] = data[:, 6, :]
        # 'CP2'(channel8) :
        data_4d[:, 2, 2, :] = data[:, 7, :]
        # 'P1' (channel9) :
        data_4d[:, 2, 3, :] = data[:, 8, :]
        # 'PZ' (channel10):
        data_4d[:, 2, 4, :] = data[:, 9, :]
        # 'P2' (channel11):
        data_4d[:, 2, 5, :] = data[:, 10, :]
        # 'PO3'(channel12):
        data_4d[:, 2, 6, :] = data[:, 11, :]
        # 'POZ'(channel13):
        data_4d[:, 2, 7, :] = data[:, 12, :]
        # 'PO4'(channel14):
        data_4d[:, 2, 8, :] = data[:, 13, :]
        # 'PO4'(channel14):
        data_4d[:, 3, 0, :] = data[:, 14, :]
        # 'CP1'(channel7) :
        data_4d[:, 3, 1, :] = data[:, 15, :]
        # 'CP2'(channel8) :
        data_4d[:, 3, 2, :] = data[:, 16, :]
        # 'P1' (channel9) :
        data_4d[:, 3, 3, :] = data[:, 17, :]
        # 'PZ' (channel10):
        data_4d[:, 3, 4, :] = data[:, 18, :]
        # 'P2' (channel11):
        data_4d[:, 3, 5, :] = data[:, 19, :]
        # 'PO3'(channel12):
        data_4d[:, 3, 6, :] = data[:, 20, :]
        # 'POZ'(channel13):
        data_4d[:, 3, 7, :] = data[:, 21, :]
        # 'PO4'(channel14):
        data_4d[:, 3, 8, :] = data[:, 22, :]
        data_4d[:, 4, 0, :] = data[:, 23, :]
        # 'CP1'(channel7) :
        data_4d[:, 4, 1, :] = data[:, 24, :]
        # 'CP2'(channel8) :
        data_4d[:, 4, 2, :] = data[:, 25, :]
        # 'P1' (channel9) :
        data_4d[:, 4, 3, :] = data[:, 26, :]
        # 'PZ' (channel10):
        data_4d[:, 4, 4, :] = data[:, 27, :]
        # 'P2' (channel11):
        data_4d[:, 4, 5, :] = data[:, 28, :]
        # 'PO3'(channel12):
        data_4d[:, 4, 6, :] = data[:, 29, :]
        # 'POZ'(channel13):
        data_4d[:, 4, 7, :] = data[:, 30, :]
        # 'PO4'(channel14):
        data_4d[:, 4, 8, :] = data[:, 31, :]
        data_4d[:, 5, 0, :] = data[:, 32, :]
        # 'CP1'(channel7) :
        data_4d[:, 5, 1, :] = data[:, 33, :]
        # 'CP2'(channel8) :
        data_4d[:, 5, 2, :] = data[:, 34, :]
        # 'P1' (channel9) :
        data_4d[:, 5, 3, :] = data[:, 35, :]
        # 'PZ' (channel10):
        data_4d[:, 5, 4, :] = data[:, 36, :]
        # 'P2' (channel11):
        data_4d[:, 5, 5, :] = data[:, 37, :]
        # 'PO3'(channel12):
        data_4d[:, 5, 6, :] = data[:, 38, :]
        # 'POZ'(channel13):
        data_4d[:, 5, 7, :] = data[:, 39, :]
        # 'PO4'(channel14):
        data_4d[:, 5, 8, :] = data[:, 40, :]
        data_4d[:, 6, 0, :] = data[:, 41, :]
        # 'CP1'(channel7) :
        data_4d[:, 6, 1, :] = data[:, 42, :]
        # 'CP2'(channel8) :
        data_4d[:, 6, 2, :] = data[:, 43, :]
        # 'P1' (channel9) :
        data_4d[:, 6, 3, :] = data[:, 44, :]
        # 'PZ' (channel10):
        data_4d[:, 6, 4, :] = data[:, 45, :]
        # 'P2' (channel11):
        data_4d[:, 6, 5, :] = data[:, 46, :]
        # 'PO3'(channel12):
        data_4d[:, 6, 6, :] = data[:, 47, :]
        # 'POZ'(channel13):
        data_4d[:, 6, 7, :] = data[:, 48, :]
        # 'PO4'(channel14):
        data_4d[:, 6, 8, :] = data[:, 49, :]
        # data_4d[:, 7, 0, :]
        # 'CP1'(channel7) :
        data_4d[:, 7, 1, :] = data[:, 50, :]
        # 'CP2'(channel8) :
        data_4d[:, 7, 2, :] = data[:, 51, :]
        # 'P1' (channel9) :
        data_4d[:, 7, 3, :] = data[:, 52, :]
        # 'PZ' (channel10):
        data_4d[:, 7, 4, :] = data[:, 53, :]
        # 'P2' (channel11):
        data_4d[:, 7, 5, :] = data[:, 54, :]
        # 'PO3'(channel12):
        data_4d[:, 7, 6, :] = data[:, 55, :]
        # 'POZ'(channel13):
        data_4d[:, 7, 7, :] = data[:, 56, :]
        # 'PO4'(channel14):
        # data_4d[:, 7, 8, :]
        data_4d[:, 8, 2, :] = data[:, 57, :]
        # 'PZ' (channel10):
        data_4d[:, 8, 3, :] = data[:, 58, :]
        # 'P2' (channel11):
        data_4d[:, 8, 4, :] = data[:, 59, :]
        # 'PO3'(channel12):
        data_4d[:, 8, 5, :] = data[:, 60, :]
        # 'POZ'(channel13):
        data_4d[:, 8, 6, :] = data[:, 61, :]
    if region=='frontal6':
    
        data_4d[:, 0, 2, :] = data[:, 0, :]
        # 'FT8'(channel2) :
        
        # 'T7' (channel3) :
        data_4d[:, 0, 6, :] = data[:, 2, :]
        # 'T8' (channel4) :
        data_4d[:, 1, 2, :] = data[:, 3, :]
        # 'TP7'(channel5) :
        data_4d[:, 1, 6, :] = data[:, 4, :]
        # 'TP8'(channel6) :
        data_4d[:, 2, 0, :] = data[:, 5, :]
        # 'CP1'(channel7) :
        data_4d[:, 2, 1, :] = data[:, 6, :]
        # 'CP2'(channel8) :
        data_4d[:, 2, 2, :] = data[:, 7, :]
        # 'P1' (channel9) :
        data_4d[:, 2, 3, :] = data[:, 8, :]
        # 'PZ' (channel10):
        
        # 'P2' (channel11):
        data_4d[:, 2, 5, :] = data[:, 10, :]
        # 'PO3'(channel12):
        data_4d[:, 2, 6, :] = data[:, 11, :]
        # 'POZ'(channel13):
        data_4d[:, 2, 7, :] = data[:, 12, :]
        # 'PO4'(channel14):
        data_4d[:, 2, 8, :] = data[:, 13, :]
    if region=='frontal10':
        data_4d[:, 0, 2, :] = data[:, 0, :]
        # 'FT8'(channel2) :
        
        # 'T7' (channel3) :
        data_4d[:, 0, 6, :] = data[:, 2, :]
        # 'T8' (channel4) :
        data_4d[:, 1, 2, :] = data[:, 3, :]
        # 'TP7'(channel5) :
        data_4d[:, 1, 6, :] = data[:, 4, :]
        # 'TP8'(channel6) :
        data_4d[:, 2, 0, :] = data[:, 5, :]
        # 'CP1'(channel7) :
        data_4d[:, 2, 1, :] = data[:, 6, :]
        # 'CP2'(channel8) :
        data_4d[:, 2, 2, :] = data[:, 7, :]
        # 'P1' (channel9) :
        data_4d[:, 2, 3, :] = data[:, 8, :]
        # 'PZ' (channel10):
        
        # 'P2' (channel11):
        data_4d[:, 2, 5, :] = data[:, 10, :]
        # 'PO3'(channel12):
        data_4d[:, 2, 6, :] = data[:, 11, :]
        # 'POZ'(channel13):
        data_4d[:, 2, 7, :] = data[:, 12, :]
        # 'PO4'(channel14):
        data_4d[:, 2, 8, :] = data[:, 13, :]
        data_4d[:, 3, 0, :] = data[:, 14, :]
        # 'CP1'(channel7) :
        data_4d[:, 3, 1, :] = data[:, 15, :]
        # 'CP2'(channel8) :
        data_4d[:, 3, 2, :] = data[:, 16, :]
        # 'P1' (channel9) :
        data_4d[:, 3, 3, :] = data[:, 17, :]
        data_4d[:, 3, 5, :] = data[:, 19, :]
        # 'PO3'(channel12):
        data_4d[:, 3, 6, :] = data[:, 20, :]
        # 'POZ'(channel13):
        data_4d[:, 3, 7, :] = data[:, 21, :]
        # 'PO4'(channel14):
        data_4d[:, 3, 8, :] = data[:, 22, :]
    if region=='temporal6':
        data_4d[:, 3, 0, :] = data[:, 14, :]
        # 'CP1'(channel7) :
        data_4d[:, 3, 1, :] = data[:, 15, :]
        data_4d[:, 3, 7, :] = data[:, 21, :]
        # 'PO4'(channel14):
        data_4d[:, 3, 8, :] = data[:, 22, :]
        data_4d[:, 4, 0, :] = data[:, 23, :]
        # 'CP1'(channel7) :
        data_4d[:, 4, 1, :] = data[:, 24, :]
        data_4d[:, 4, 7, :] = data[:, 30, :]
        # 'PO4'(channel14):
        data_4d[:, 4, 8, :] = data[:, 31, :]
        data_4d[:, 5, 0, :] = data[:, 32, :]
        # 'CP1'(channel7) :
        data_4d[:, 5, 1, :] = data[:, 33, :]
        data_4d[:, 5, 7, :] = data[:, 39, :]
        # 'PO4'(channel14):
        data_4d[:, 5, 8, :] = data[:, 40, :]
    if region=='temporal9':
        data_4d[:, 3, 0, :] = data[:, 14, :]
        # 'CP1'(channel7) :
        data_4d[:, 3, 1, :] = data[:, 15, :]
        data_4d[:, 3, 7, :] = data[:, 21, :]
        # 'PO4'(channel14):
        data_4d[:, 3, 8, :] = data[:, 22, :]
        data_4d[:, 4, 0, :] = data[:, 23, :]
        # 'CP1'(channel7) :
        data_4d[:, 4, 1, :] = data[:, 24, :]
        data_4d[:, 4, 7, :] = data[:, 30, :]
        # 'PO4'(channel14):
        data_4d[:, 4, 8, :] = data[:, 31, :]
        data_4d[:, 5, 0, :] = data[:, 32, :]
        # 'CP1'(channel7) :
        data_4d[:, 5, 1, :] = data[:, 33, :]
        data_4d[:, 5, 7, :] = data[:, 39, :]
        # 'PO4'(channel14):
        data_4d[:, 5, 8, :] = data[:, 40, :]
        data_4d[:, 3, 2, :] = data[:, 16, :]
        data_4d[:, 3, 6, :] = data[:, 20, :]
        data_4d[:, 4, 2, :] = data[:, 25, :]
        data_4d[:, 4, 6, :] = data[:, 29, :]
        data_4d[:, 5, 2, :] = data[:, 34, :]
        data_4d[:, 5, 6, :] = data[:, 38, :]
    
    data_4d_reshape = np.swapaxes(data_4d, 1, 3)

    data_4d_reshape = torch.from_numpy(data_4d_reshape).float()
    # data_4d_reshape = np.swapaxes(data_4d_reshape, 2, 3)
    #print("data_4d_reshape.shape: ", data_4d_reshape.shape)
    return data_4d_reshape
