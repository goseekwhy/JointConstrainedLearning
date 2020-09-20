import tqdm
import time
import datetime
import random
import numpy as np
from document_reader import *
from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
import sys
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model_aug import *
from metric import metric, CM_metric
from exp import *

torch.manual_seed(42)
debugging = False

### Read parameters ###
if len(sys.argv) > 1:
    gpu_num, batch_size, learning_rate, rst_file_name, epochs, dataset, add_loss, finetune = sys.argv[1][-1], int(sys.argv[2][6:]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5][6:]), sys.argv[6], int(sys.argv[7][9:]), int(sys.argv[8][9:])
    
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
cuda = torch.device('cuda')
writer = SummaryWriter(comment=rst_file_name.replace(".rst", ""))

### restore model ###
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
MATRES_best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt") # for storing model param in training
load_model_path = "./model_params/" + dataset + "_best/" # for test

if dataset == "HiEve" or dataset == "Joint":
    # ========================
    #       HiEve Dataset
    # ========================
    dir_name = "./hievents_v2/processed/"
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    train_range = range(0, 60)
    valid_range = range(60, 80)
    test_range = range(80, 100)
    train_set_HIEVE = []
    valid_set_HIEVE = []
    test_set_HIEVE = []
    undersmp_ratio = 0.4

    t0 = time.time()
    doc_id = -1
    for file_name in tqdm.tqdm(onlyfiles):
        doc_id += 1
        if doc_id in train_range:
            my_dict = tsvx_reader(dir_name, file_name)
            num_event = len(my_dict["event_dict"])
            # range(a, b): [a, b)
            for x in range(1, num_event+1):
                for y in range(x+1, num_event+1):
                    for z in range(y+1, num_event+1):
                        x_sent_id = my_dict["event_dict"][x]["sent_id"]
                        y_sent_id = my_dict["event_dict"][y]["sent_id"]
                        z_sent_id = my_dict["event_dict"][z]["sent_id"]

                        x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
                        y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])
                        z_sent = padding(my_dict["sentences"][z_sent_id]["roberta_subword_to_ID"])

                        x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                        y_position = my_dict["event_dict"][y]["roberta_subword_id"]
                        z_position = my_dict["event_dict"][z]["roberta_subword_id"]

                        x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                        y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
                        z_sent_pos = padding(my_dict["sentences"][z_sent_id]["roberta_subword_pos"], pos = True)

                        xy = my_dict["relation_dict"][(x, y)]["relation"]
                        yz = my_dict["relation_dict"][(y, z)]["relation"]
                        xz = my_dict["relation_dict"][(x, z)]["relation"]

                        to_append = x, y, z, \
                                    x_sent, y_sent, z_sent, \
                                    x_position, y_position, z_position, \
                                    x_sent_pos, y_sent_pos, z_sent_pos, \
                                    xy, yz, xz, 0 # 0 means HiEve

                        if xy == 3 and yz == 3:
                            pass
                        elif xy == 3 or yz == 3 or xz == 3:
                            if random.uniform(0, 1) < 0.05:
                                train_set_HIEVE.append(to_append)
                        else:
                            train_set_HIEVE.append(to_append)
        else:
            my_dict = tsvx_reader(dir_name, file_name)
            num_event = len(my_dict["event_dict"])
            for x in range(1, num_event+1):
                for y in range(x+1, num_event+1):
                    x_sent_id = my_dict["event_dict"][x]["sent_id"]
                    y_sent_id = my_dict["event_dict"][y]["sent_id"]

                    x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
                    y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])

                    x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                    y_position = my_dict["event_dict"][y]["roberta_subword_id"]

                    x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                    y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)

                    xy = my_dict["relation_dict"][(x, y)]["relation"]

                    to_append = x, y, x, \
                                x_sent, y_sent, x_sent, \
                                x_position, y_position, x_position, \
                                x_sent_pos, y_sent_pos, x_sent_pos, \
                                xy, xy, xy, 0

                    if doc_id in valid_range:
                        if xy == 3:
                            if random.uniform(0, 1) < undersmp_ratio:
                                valid_set_HIEVE.append(to_append)
                        else:
                            valid_set_HIEVE.append(to_append)
                    else:
                        if xy == 3:
                            if random.uniform(0, 1) < undersmp_ratio:
                                test_set_HIEVE.append(to_append)
                        else:
                            test_set_HIEVE.append(to_append)

    elapsed = format_time(time.time() - t0)
    print("HiEve Preprocessing took {:}".format(elapsed))
    
if dataset == "MATRES" or dataset == "Joint":
    # ========================
    #       MATRES Dataset
    # ========================
    t0 = time.time()
    train_set_MATRES = []
    valid_set_MATRES = []
    test_set_MATRES = []

    for fname in tqdm.tqdm(eiid_pair_to_label.keys()):
        file_name = fname + ".tml"
        if file_name in onlyfiles_TB:
            dir_name = mypath_TB
        elif file_name in onlyfiles_AQ:
            dir_name = mypath_AQ
        elif file_name in onlyfiles_PL:
            dir_name = mypath_PL
        else:
            continue
        my_dict = tml_reader(dir_name, file_name)
        eiid_to_event_trigger_dict = eiid_to_event_trigger[fname]
        if file_name in onlyfiles_TB:
            for eiid1 in eiid_to_event_trigger_dict.keys():
                for eiid2 in eiid_to_event_trigger_dict.keys():
                    for eiid3 in eiid_to_event_trigger_dict.keys():
                        if eiid1!=eiid2 and eiid2!=eiid3 and eiid1!=eiid3:
                            if (eiid1, eiid2) in eiid_pair_to_label[fname].keys() and (eiid2, eiid3) in eiid_pair_to_label[fname].keys() and (eiid1, eiid3) in eiid_pair_to_label[fname].keys():

                                xy = eiid_pair_to_label[fname][(eiid1, eiid2)]
                                yz = eiid_pair_to_label[fname][(eiid2, eiid3)]
                                xz = eiid_pair_to_label[fname][(eiid1, eiid3)]

                                x = my_dict["eiid_dict"][eiid1]["eID"]
                                y = my_dict["eiid_dict"][eiid2]["eID"]
                                z = my_dict["eiid_dict"][eiid3]["eID"]

                                x_sent_id = my_dict["event_dict"][x]["sent_id"]
                                y_sent_id = my_dict["event_dict"][y]["sent_id"]
                                z_sent_id = my_dict["event_dict"][z]["sent_id"]

                                x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
                                y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])
                                z_sent = padding(my_dict["sentences"][z_sent_id]["roberta_subword_to_ID"])

                                x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                                y_position = my_dict["event_dict"][y]["roberta_subword_id"]
                                z_position = my_dict["event_dict"][z]["roberta_subword_id"]

                                x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                                y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
                                z_sent_pos = padding(my_dict["sentences"][z_sent_id]["roberta_subword_pos"], pos = True)

                                to_append = (x, y, z, \
                                             x_sent, y_sent, z_sent, \
                                             x_position, y_position, z_position, \
                                             x_sent_pos, y_sent_pos, z_sent_pos, \
                                             xy, yz, xz, 1) # 1 means MATRES
                                #print(to_append)
                                train_set_MATRES.append(to_append)

        else:
            for (eiid1, eiid2) in eiid_pair_to_label[fname].keys():
                xy = eiid_pair_to_label[fname][(eiid1, eiid2)]

                x = my_dict["eiid_dict"][eiid1]["eID"]
                y = my_dict["eiid_dict"][eiid2]["eID"]

                x_sent_id = my_dict["event_dict"][x]["sent_id"]
                y_sent_id = my_dict["event_dict"][y]["sent_id"]

                x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
                y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])

                x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                y_position = my_dict["event_dict"][y]["roberta_subword_id"]

                x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)

                to_append = x, y, x, \
                            x_sent, y_sent, x_sent, \
                            x_position, y_position, x_position, \
                            x_sent_pos, y_sent_pos, x_sent_pos, \
                            xy, xy, xy, 1
                if file_name in onlyfiles_AQ:
                    valid_set_MATRES.append(to_append)
                elif file_name in onlyfiles_PL:
                    test_set_MATRES.append(to_append)

    elapsed = format_time(time.time() - t0)
    print("MATRES Preprocessing took {:}".format(elapsed)) 
    
if debugging:
    if dataset == "MATRES":
        train_set_MATRES = train_set_MATRES[0:100]
        test_set_MATRES = train_set_MATRES
        valid_set_MATRES = train_set_MATRES
    elif dataset == "HiEve":
        train_set_HIEVE = train_set_HIEVE[0:100]
        test_set_HIEVE = train_set_HIEVE
        valid_set_HIEVE = train_set_HIEVE
    
# ==============================================================
#      Use DataLoader to convert to Pytorch acceptable form
# ==============================================================
if dataset == "MATRES":
    num_classes = 4
    train_dataloader_MATRES = DataLoader(EventDataset(train_set_MATRES), batch_size=batch_size, shuffle = True)
    valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
    test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True) 
    print("length of train_dataloader_MATRES:", len(train_dataloader_MATRES))
    print("length of train_dataloader_MATRES:", len(train_dataloader_MATRES))
    print("length of train_dataloader_MATRES:", len(train_dataloader_MATRES))
elif dataset == "HiEve":
    num_classes = 4
    train_dataloader_HIEVE = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
    valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle = True)    
    test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True)  
elif dataset == "Joint":
    num_classes = 8
    train_set_HIEVE.extend(train_set_MATRES)
    train_dataloader = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
    valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
    test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True)
    valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle = True)    
    test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True)
else:
    raise ValueError("Currently not supporting this dataset! -_-'")

MLP_size = 512
if finetune:
    hidden_size = 1024
    model = roberta_mlp(hidden_size, num_classes, MLP_size, dataset, add_loss)
else:
    hidden_size = 256
    model = BiLSTM_MLP(768, hidden_size, MLP_size, 1, num_classes, add_loss)
model.to(cuda)
model.zero_grad()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("# of parameters:", count_parameters(model))

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
if dataset == "MATRES":
    total_steps = len(train_dataloader_MATRES) * epochs

model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters

if dataset == "MATRES":
    matres_exp = exp(cuda, model, epochs, learning_rate, train_dataloader_MATRES, valid_dataloader_MATRES, test_dataloader_MATRES, None, None, finetune, dataset, MATRES_best_PATH, None, load_model_path, model_name)
    matres_exp.train()
    matres_exp.evaluate(test = True)
elif dataset == "HiEve":
    hieve_exp = exp(cuda, model, epochs, learning_rate, train_dataloader_HIEVE, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, None, HiEve_best_PATH, load_model_path, model_name)
    hieve_exp.train()
    hieve_exp.evaluate(test = True)
elif dataset == "Joint":
    joint_exp = exp(cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, MATRES_best_PATH, HiEve_best_PATH, load_model_path, model_name)
    joint_exp.train()
    joint_exp.evaluate(test = True, eval_data = "MATRES")
    joint_exp.evaluate(test = True, eval_data = "HiEve")
else:
    raise ValueError("Currently not supporting this dataset! -_-'")