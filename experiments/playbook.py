# TODO
# add module check points for observing environemnts
# sample the trainng data 
# change to passing batched
# wrong passign + shuffle the passed batch after configuring the batch size to send

import os 
import glob 
import json 
from typing import List, Literal 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np 

device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)

max_grid_size = 30
min_grid_size = 1
total_colors = 10

class GridEncoder(nn.Module):
    def __init__(self, grid_size: int = min_grid_size, output_size: int = max_grid_size):
        super(GridEncoder, self).__init__()
        self.encoder = nn.Conv2d(in_channels=grid_size, out_channels=output_size, kernel_size=2, stride=2, padding=1, padding_mode='reflect', bias=True).to(device)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
        self.latent = nn.Linear(256 * 30, 128).to(device)
        self.classifier = nn.Linear(128, output_size * output_size * 10).to(device)

    def forward(self, inputs: torch.Tensor):
        assert len(inputs.shape) == 3, f"Incorrect input shape: {inputs.shape}"
        assert inputs.shape[1] == inputs.shape[2], f"1 batch of input shape must be n x n size"
        # removed the activation function after passing encoder
        # example input size = [1, 29, 29]
        x = self.encoder(inputs).to(device) # [30, 15, 15]
        x_pooler = torch.flatten(self.pool(x), start_dim=1).to(device)  # [batch_size, 256] = [30, 256]
        output = F.relu(self.latent(x_pooler.flatten().unsqueeze(0))).to(device)  # Output shape [batch_size, 128] = [256, 128]
        output = self.classifier(output).unsqueeze(0).to(device) # [1, 1, 9000]
        size = output.shape[-1] // 10
        outputs = output.reshape(1, size, 10) # [1, 900, 10]
        return outputs.sigmoid()

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

glob_path = f'{os.getcwd()}/ARC-AGI/data/training/*.json'

class DataBasket:
    def __init__(self):
        data = []
        for item in glob.glob(glob_path):
            with open(item, 'r') as file:
                data += [json.load(file)]
                
        self._train = [j for i in data for j in i['train']]
        self._test = [j for i in data for j in i['test']]

    def transform(self, x_arrs: torch.Tensor):
        x_height_pad = (max_grid_size - x_arrs.shape[1]) // 2
        x_width_pad = (max_grid_size - x_arrs.shape[2]) // 2

        # Check if there's an odd number of rows/columns to pad
        x_height_pad_extra = (max_grid_size - x_arrs.shape[1]) % 2
        x_width_pad_extra = (max_grid_size - x_arrs.shape[2]) % 2

        # Apply padding: (left, right, top, bottom)
        padding = (x_width_pad, x_width_pad + x_width_pad_extra, x_height_pad, x_height_pad + x_height_pad_extra)
        x = F.pad(x_arrs, padding)
        
        assert x.shape[1] == x.shape[-1], f"Error padding x training data: x shape is supposed to be '[1, 30, 30]' but received {x.shape}"
        
        return x 
    
    def transform_predictor(self, y_arrs: torch.Tensor):
        assert len(y_arrs.shape) == 2, f"Expected 2 size shape for y predictor, received array y of shape: {y_arrs.shape}"
        y_indices = [j.item() for i in y_arrs.cpu().numpy() for j in i]
        y = np.zeros((len(y_indices), total_colors), dtype=np.int32)
        
        assert len(y_indices) == y.shape[0], f'Unmatched size for y indices: {np.array(y_indices).shape} and y zeros: {y.shape}'
        
        for row, value in enumerate(y_indices):
            y[row][value] = 1
            
        y_height_pad = abs(max_grid_size * max_grid_size - len(y_arrs.flatten()))
        y_pad_top = y_height_pad // 2
        y_pad_bottom = y_height_pad - y_pad_top
        
        y = torch.tensor(y, dtype=torch.float, requires_grad=False, device=device).unsqueeze(0)
        y = F.pad(y, (0, 0, y_pad_top, y_pad_bottom))
        
        assert list(y.shape) == [1, 900, 10], f"Error with padding y prediction variable: shape {y.shape}"
        return y 
    
    def fetch_data(self, data_type: Literal['train', 'test']):
        if data_type == 'train':
            data = self._train
        else:
            data = self._test
            
        print(f"Number of {data_type} data: {len(data)}")
        for item in data:
            x_arrs, y_arrs = item['input'], item['output']
            
            x_arrs = torch.tensor(x_arrs, dtype=torch.float, device=device).unsqueeze(0)
            y_arrs = torch.tensor(y_arrs, requires_grad=False, device=device)
            
            #assert x_arrs.shape[1] == y_arrs.shape[0], f"Expected same n.o of rows for x and y training data: x - {x_arrs.shape} and y - {y_arrs.shape}"
            
            x = self.transform(x_arrs)
            y = self.transform_predictor(y_arrs)
            
            yield x, y
    
def train(basket: DataBasket, model, optimizer, scheduler, criterion, max_epoch: int):
    model.train()
    total_train_loss = []
    for ep in range(max_epoch):
        train_loss = []
        for x, y in basket.fetch_data('train'):
            print(f'Training x: {x.shape} and y: {y.shape}')
            optimizer.zero_grad()
            output = model(x.to(device)) # output shape [30, 31, 15]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += [loss.item()]
            
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        print(f"EPOCH: {ep} \t Avg. Loss: {np.array(train_loss).mean()} \t LR = {current_lr}")
        total_train_loss += [np.array(train_loss).mean()]
    return np.mean(total_train_loss)

def test_model(basket: DataBasket, model, data_type: Literal['train', 'test']):
    model.eval()
    test_metric = []
    test_grid_metric = []
    bool_grid_metric = []
    for input_x, true_y in basket.fetch_data(data_type):
        with torch.no_grad():
            pred = model(input_x)
        
        grid_points = [point for point, i in enumerate(true_y.squeeze()) if i.sum() != 0]
        grid_correct = [1 if pred.squeeze()[point].argmax().item() == true_y.squeeze()[point].argmax().item() else 0 for point in grid_points]
        correct = [1 if i.argmax().item() == true_y.squeeze()[idx].argmax().item() else 0 for idx, i in enumerate(pred.squeeze())]
        
        score = np.sum(correct) / len(correct)
        grid_score = np.sum(grid_correct) / len(grid_correct)
        
        print(f"Total correct: {score * 100:.4f} %  \t Predicted grid points: {grid_score * 100:.4f} % ")
        
        test_metric += [score]
        test_grid_metric += [grid_score]
        bool_grid_metric.append(1 if grid_score > 0.70 else 0)
        
    print(f"Average: {np.mean(test_metric)} \t Average grid scored: {np.mean(test_grid_metric)}")

def main():
    data = DataBasket()
    model = GridEncoder()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    max_epoch = 20
    total_loss = train(basket=data, model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, max_epoch=max_epoch)
    save_checkpoint(model, optimizer, max_epoch, total_loss)
    data.test_model(basket=data, model=model, data_type='test')
    data.test_model(basket=data, model=model, data_type='train')

