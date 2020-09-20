import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from transformers import RobertaModel
from all_loss_aug import transitivity_loss_H_, transitivity_loss_T_, cross_category_loss_

HierPC = 1802.0
HierCP = 1846.0
HierCo = 758.0
HierNo = 63755.0 
HierTo = HierPC + HierCP + HierCo + HierNo # total number of event pairs
hier_weights = [0.25*HierTo/HierPC, 0.25*HierTo/HierCP, 0.25*HierTo/HierCo, 0.25*HierTo/HierNo]
temp_weights = [0.25*818.0/412.0, 0.25*818.0/263.0, 0.25*818.0/30.0, 0.25*818.0/113.0]

# roberta + MLP
class roberta_mlp(nn.Module):
    def __init__(self, hidden_size, num_classes, MLP_size, dataset, add_loss, Sub = True, Mul = True, freq = None):
        super(roberta_mlp, self).__init__()
        self.dataset = dataset
        self.Sub = Sub
        self.Mul = Mul
        self.add_loss = add_loss
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model = RobertaModel.from_pretrained('roberta-large')
        self.hier_class_weights = torch.FloatTensor(hier_weights).cuda()
        self.temp_class_weights = torch.FloatTensor(temp_weights).cuda()
        self.HiEve_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights)
        self.MATRES_anno_loss = nn.CrossEntropyLoss(weight=self.temp_class_weights)
        self.transitivity_loss_H = transitivity_loss_H_()
        self.transitivity_loss_T = transitivity_loss_T_()
        self.cross_category_loss = cross_category_loss_()
        if freq is not None:
            self.fc1 = nn.Linear(hidden_size*4+1, MLP_size)  
        else:
            if self.Sub is None and self.Mul is None:
                self.fc1 = nn.Linear(hidden_size*2, MLP_size)
                self.fc2 = nn.Linear(MLP_size, num_classes)
            elif self.Sub is not None and self.Mul is not None:
                self.fc1 = nn.Linear(hidden_size*4, MLP_size*2)
                self.fc2 = nn.Linear(MLP_size*2, num_classes)
            else:
                self.fc1 = nn.Linear(hidden_size*3, int(MLP_size*1.5))
                self.fc2 = nn.Linear(int(MLP_size*1.5), num_classes)
        self.relu = nn.LeakyReLU(0.2, True)
        
    def forward(self, x_sent, y_sent, z_sent, x_position, y_position, z_position, xy, yz, xz, flag, loss_out = None):
        batch_size = x_position.size(0)
        
        output_x = self.model(x_sent)[0]
        output_y = self.model(y_sent)[0]
        output_z = self.model(z_sent)[0]
        
        output_A = torch.cat([output_x[i, x_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        output_B = torch.cat([output_y[i, y_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        output_C = torch.cat([output_z[i, z_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        
        if self.Sub is None and self.Mul is None:
            alpha_representation = torch.cat((output_A, output_B), 1)
            beta_representation = torch.cat((output_B, output_C), 1)
            gamma_representation = torch.cat((output_A, output_C), 1)
        elif self.Sub is not None and self.Mul is not None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC, mulAC), 1)
        elif self.Sub is not None and self.Mul is None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC), 1)
        else:
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, mulAC), 1)
            
        alpha_logits = self.fc2(self.relu(self.fc1(alpha_representation)))
        beta_logits = self.fc2(self.relu(self.fc1(beta_representation)))
        gamma_logits = self.fc2(self.relu(self.fc1(gamma_representation)))
        if loss_out is None:
            return alpha_logits, beta_logits, gamma_logits
        else:
            loss = 0.0
            if self.dataset == "MATRES":
                #print("calculating annotation loss")
                loss += self.MATRES_anno_loss(alpha_logits, xy) + self.MATRES_anno_loss(beta_logits, yz) + self.MATRES_anno_loss(gamma_logits, xz)
                if self.add_loss:
                    #print("adding transitivity loss")
                    loss += self.transitivity_loss_T(alpha_logits, beta_logits, gamma_logits).sum()

            elif self.dataset == "HiEve":
                #print("calculating annotation loss")
                loss += self.HiEve_anno_loss(alpha_logits, xy) + self.HiEve_anno_loss(beta_logits, yz) + self.HiEve_anno_loss(gamma_logits, xz)
                if self.add_loss:
                    #print("adding transitivity loss")
                    loss += self.transitivity_loss_H(alpha_logits, beta_logits, gamma_logits).sum()

            else:
                for i in range(0, batch_size):
                    if flag[i] == 1:
                        loss += self.MATRES_anno_loss(alpha_logits[i][4:], xy[i]) + self.MATRES_anno_loss(beta_logits[i][4:], yz[i]) + self.MATRES_anno_loss(gamma_logits[i][4:], xz[i])
                    elif flag[i] == 0:
                        loss += self.HiEve_anno_loss(alpha_logits[i][0:4], xy[i]) + self.HiEve_anno_loss(beta_logits[i][0:4], yz[i]) + self.HiEve_anno_loss(gamma_logits[i][0:4], xz[i])
                    if self.add_loss:
                        loss += self.transitivity_loss_T(alpha_logits[4:], beta_logits[4:], gamma_logits[4:]).sum()
                        loss += self.transitivity_loss_H(alpha_logits[0:4], beta_logits[0:4], gamma_logits[0:4]).sum()
                        loss += self.cross_category_loss(alpha_logits, beta_logits, gamma_logits).sum()

            return alpha_logits, beta_logits, gamma_logits, loss
        
        
# BiLSTM + MLP
class BiLSTM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, MLP_size, num_layers, num_classes, add_loss = True, freq = None, Sub = None, Mul = None, dataset = "HiEve"):
        super(BiLSTM_MLP, self).__init__()
        self.dataset = dataset
        self.add_loss = add_loss
        self.Sub = Sub
        self.Mul = Mul
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.num_classes = num_classes # 8: 4 for Hier, 4 for Temp
        self.hier_class_weights = torch.FloatTensor(hier_weights).cuda()
        self.temp_class_weights = torch.FloatTensor(temp_weights).cuda()
        if dataset == "HiEve":
            self.loss = nn.CrossEntropyLoss(weight=self.hier_class_weights)
        elif dataset == "MATRES":
            self.loss = nn.CrossEntropyLoss(weight=self.temp_class_weights)
        self.transitivity_loss_H = transitivity_loss_H_()
        self.transitivity_loss_T = transitivity_loss_T_()
        if freq is not None:
            self.fc1 = nn.Linear(hidden_size*4+1, MLP_size)  
        else:
            if self.Sub is None and self.Mul is None:
                self.fc1 = nn.Linear(hidden_size*4, MLP_size)  # 4: 2 for bidirection, 2 for concatenation
                self.fc2 = nn.Linear(MLP_size, num_classes)
            elif self.Sub is not None and self.Mul is not None:
                self.fc1 = nn.Linear(hidden_size*8, MLP_size*2)
                self.fc2 = nn.Linear(MLP_size*2, num_classes)
            else:
                self.fc1 = nn.Linear(hidden_size*6, int(MLP_size*1.5))
                self.fc2 = nn.Linear(int(MLP_size*1.5), num_classes)
        self.relu = nn.LeakyReLU(0.2, True)
    
    def forward(self, A, B, C, A_pos, B_pos, C_pos, freqAB = None, freqBC = None, freqAC = None, xy = None, yz = None, xz = None, flag = None, loss_out = None):
        batch_size = A_pos.size(0)
        BiLSTM_output_A, _ = self.lstm(A) # size: [batch_size, 78, 256]
        BiLSTM_output_B, _ = self.lstm(B)
        BiLSTM_output_C, _ = self.lstm(C)
        output_A = torch.cat([BiLSTM_output_A[i, A_pos[i].long(), :].unsqueeze(0) for i in range(batch_size)], 0)
        output_B = torch.cat([BiLSTM_output_B[i, B_pos[i].long(), :].unsqueeze(0) for i in range(batch_size)], 0)
        output_C = torch.cat([BiLSTM_output_C[i, C_pos[i].long(), :].unsqueeze(0) for i in range(batch_size)], 0)
        
        if self.Sub is None and self.Mul is None:
            alpha_representation = torch.cat((output_A, output_B), 1)
            beta_representation = torch.cat((output_B, output_C), 1)
            gamma_representation = torch.cat((output_A, output_C), 1)
        #elif freqAB is not None and self.Sub is None and self.Mul is None:
        #    alpha_representation = torch.cat((output_A, output_B, freqAB.view(-1, 1)), 1)
        #    beta_representation = torch.cat((output_B, output_C, freqBC.view(-1, 1)), 1)
        #    gamma_representation = torch.cat((output_A, output_C, freqAC.view(-1, 1)), 1)
        elif self.Sub is not None and self.Mul is not None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC, mulAC), 1)
        elif self.Sub is not None and self.Mul is None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC), 1)
        else:
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, mulAC), 1)
        alpha_logits = self.fc2(self.relu(self.fc1(alpha_representation)))
        beta_logits = self.fc2(self.relu(self.fc1(beta_representation)))
        gamma_logits = self.fc2(self.relu(self.fc1(gamma_representation)))
        if loss_out is None:
            return alpha_logits, beta_logits, gamma_logits
        else:
            loss = self.loss(alpha_logits, xy) + self.loss(beta_logits, yz) + self.loss(gamma_logits, xz)
            if self.add_loss:
                #print("adding transitivity loss")
                if self.dataset == "MATRES":
                    loss += self.transitivity_loss_T(alpha_logits, beta_logits, gamma_logits).sum()
                elif self.dataset == "HiEve":
                    loss += self.transitivity_loss_H(alpha_logits, beta_logits, gamma_logits).sum()
                #else: # Joint
            return alpha_logits, beta_logits, gamma_logits, loss