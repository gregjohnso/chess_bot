import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class model(nn.Module):

    def __init__(self, gpu_ids, opt=None):
        super(model, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        ksize = 8
        dstep = 2
        

        self.main = nn.Sequential(
            
            #board in is 8x8
            nn.Conv2d(13, 256, ksize, 1, 4),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            #board in is 8x8
            nn.Conv2d(256, 256, ksize, 1, 4),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            #board in is 8x8
            nn.Conv2d(256, 512, ksize, 1, 4),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            #board in is 8x8
            nn.Conv2d(512, 1024, ksize, 2, 1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            
            #board in is 4x4
            nn.Conv2d(1024, 2048, 4, 2, 1),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
            
            #board in is 2x2
            nn.Conv2d(2048, 2048, 2, 2, 1),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),)

        #three classes out, white, black, tie
        self.classOut = nn.Sequential(nn.Linear(2048, 3))

        if gpu_ids is not None:
            self.main = self.main.cuda(gpu_ids[0])
            self.classOut = self.classOut.cuda(gpu_ids[0])
        
    def forward(self, x):
        gpu_ids = self.gpu_ids
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 2048 )

        pred = nn.parallel.data_parallel(self.classOut, x, gpu_ids)

        return pred        

class BoardDataSet(Dataset):
    def __init__(self, board_list):
        """
        Creates a dataset out of chess_bot.Board objects.
        Returns 2 variables, state, which is every board state for every game (last vector is whose turn it is), and winner, which is whether that board state was associated with a loss (0), win (1), or a tie (2)
        """
        winner = [board.who_won() for board in board_list]
        game_len = [len(board) for board in board_list]
        
        winner_list = list()
        
        for w, l in zip(winner, game_len):
            winner_list.append(np.ones(l) * w)
            
        all_winners = np.hstack(winner_list)
            
        all_states = list()
        
        for board in board_list:
            states, turns = board.vec_history()
            
            states = np.concatenate([np.expand_dims(state, axis=0) for state in states])
            turns_array = np.zeros([len(turns), 1, 8, 8])
            for i,turn in zip(range(0, len(turns)), turns): turns_array[i] = turn
                
            all_states.append(np.concatenate([states, turns_array], axis=1))
            

#         all_turns = np.concatenate(all_turns)
        all_states = np.concatenate(all_states)
        
        self.state = torch.Tensor(all_states) 
#         self.turn = torch.Tensor(all_turns.astype(float))
        self.winner = torch.LongTensor(all_winners)
        
        
    def __len__(self):
        return len(self.winner)

    def __getitem__(self, idx):
        item = {'x': self.state[idx], 'y': self.winner[idx]}
        
        return item

class TorchBot(object):
    def __init__(self, play_mode='explore', learning_rate=1E-4, batch_size = 32, gpu_ids = None):
        
        self.model = model(gpu_ids)
        self.model.apply(weights_init)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        self.play_mode = play_mode

        self.criterion = nn.CrossEntropyLoss()
        
        self.batch_size = batch_size
        
    def choose_move(self, board):
        ###make dataprovider
        
        self.model.train(False)
        
        states, turns = board.vec_next_moves()
        
        #### munge data to make it be digestable ####
        states = np.concatenate([np.expand_dims(state, axis=0) for state in states])
        
        turns_array = np.zeros([len(turns), 1, 8, 8])
        for i,turn in zip(range(0, len(turns)), turns): turns_array[i] = turn
        states = np.concatenate([states, turns_array], axis=1)
           
        x = Variable(torch.Tensor(states), volatile=True)
            
        target_pred = self.model(x)

        
        pdb.set_trace()
        
        target_pred = target_pred.data.cpu().numpy()
        
        _, indices = torch.max(target_pred, 1)

        acc = (indices == target).double().mean().data[0]

        x.volatile = False
        self.model.train(True)

        errors = (
            pred_loss,
            acc,)
        return errors
        
    def train_from_board(self, boards):
        
#         if gpu_ids is not None:
#             gpu_id = opt['gpu_ids'][0]

        dataset = BoardDataSet(boards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last = True)

        losses = list()
        start = time.time()
        
        for datum in dataloader:
            
            
            
            x = Variable(datum['x'])
            y = Variable(datum['y'])
            
            if self.model.gpu_ids is not None:
                gpu_id = self.model.gpu_ids[0]
                x = x.cuda(gpu_id)
                y = y.cuda(gpu_id)
            
            self.optimizer.zero_grad()

            ## train the classifier
            y_hat = self.model(x)

            
            pred_loss = self.criterion(y_hat, y)
            pred_loss.backward()
            pred_loss = pred_loss.data[0]

            self.optimizer.step()

            losses.append(pred_loss)
            
        end = time.time()
        t = end-start

        print('loss: ' + str(np.mean(losses)) + ', time: ' + str(t))
        
    def load(self, save_path):
        gpu_id = self.model.gpu_ids[0]

        checkpoint = torch.load(save_path)

        self.model.load_state_dict(checkpoint['model'])
        self.model.cuda(gpu_id)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.state = set_gpu_recursive(self.optimizer.state, gpu_id)


    def save(self, save_path):
        #         for saving and loading see:
        #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.model.gpu_ids[0]

        self.model = self.model.cpu()
        self.optimizer.state = set_gpu_recursive(self.optimizer.state, -1)

        checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}

        torch.save(checkpoint, save_path)

        self.model = self.model.cuda(gpu_id)
        self.optimizer.state = set_gpu_recursive(self.optimizer.state, gpu_id)    