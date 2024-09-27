
from altair import X
import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim 

from datasets import Dataset
from grid_util import GridBasket
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

color_pallet = 9
max_grid_size = 30 
max_flatten_size = 900 # 900 max_grid_size * max_grid_size
default_hidden_dim = max_flatten_size // 2

learning_rate = 0.01

class Head(nn.Module):
    def __init__(self, input_dim: int = max_grid_size, output_dim: int = max_flatten_size):
        super(Head, self).__init__()
        self.xemb = nn.Embedding(num_embeddings=max_grid_size, embedding_dim=max_grid_size, device=device) # 30, 900
        self.yemb = nn.Embedding(num_embeddings=max_grid_size, embedding_dim=max_grid_size, device=device) # 30, 900
        self.fc = nn.Linear(in_features=output_dim*output_dim, out_features=output_dim, bias=True, device=device)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.to(device)
        
    def forward(self, input_tensor, x_coord, y_coord):
        assert len(x_coord.shape) == 3 and len(y_coord.shape) == 3
        
        x_emb = self.xemb(x_coord).squeeze(1)
        y_emb = self.yemb(y_coord).squeeze(1)
        #conditioner = torch.matmul(x_emb, y_emb.T).to(device)
        #print('Conditioner shape: ', conditioner.shape)
        x_emb = x_emb.expand(x_emb.shape[0], 30, 30).reshape(x_emb.shape[0], 1, 900) # [batch_size, 1, 900]
        y_emb = y_emb.expand(y_emb.shape[0], 30, 30).reshape(y_emb.shape[0], 1, 900) # [batch_size, 1, 900]
        print(f"Shape of x_emb after expansion: {x_emb.shape}")
        print(f"Shape of input tensor: {input_tensor.shape}")
        x = input_tensor.transpose(1, 2) # [batch_size, 1, 900]
        print(f"New input tensors shape: {x.shape}") # [500, 900, 1]
        x_comb = torch.matmul(x, x_emb).to(device) # [batch_index, 900, 900]
        y_comb = torch.matmul(x, y_emb).to(device)
        print(f"Shape of x comb and y comb {x_comb.shape}")
        #inputs = torch.concat([x_comb, y_comb], dim=1) # [batch_size, 1800, 900]
        #inputs = torch.mean(inputs, dim=-1).unsqueeze(0).reshape(x_coord.shape[0], 900, 900) # [batch_size, 900, 900] --- [batch_size, 1800] -> [batch_size, 900
        inputs = x_comb + y_comb # [batch_index, 900, 900]
        print(f"Shape of inputs after concatting with the grid tensors: {inputs.shape}")
        output = self.relu(self.fc(inputs.reshape(inputs.shape[0], 900 * 900).unsqueeze(0))) # [900]
        output = output.permute(1, 0, 2)
        return output 
    
class CustomTrainer:
    def __init__(self, **kwargs):
        self.model = Head()
        self.lr = kwargs['lr']
        self.criterion = nn.MSELoss()
    
    def train_centre(self, train_data: GridBasket, max_epoch: int, batch_size: int, stopper: int = 1):
        self.model.train()
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        
        train_loss = []
        for epoch in range(max_epoch):
            loss_per_epoch = []
            for batch_index, data in enumerate(data_loader):
                if batch_index == stopper:
                    break 
                
                X, Y, x_shape, y_shape = data['input'].to(device), data['output'].to(device), data['x_shape'], data['y_shape']
                self.model.optimizer.zero_grad()
                output = self.model(X, x_shape, y_shape)
                print(f"Received output size of {output.shape}")
                loss = self.criterion(output, Y)
                loss.backward()
                self.model.optimizer.step()
                if batch_index == 0:
                    print(f"Batch iter: {batch_index} \t Training Loss: {loss.item()}")
                loss_per_epoch += [loss.item()]
                
            train_loss += [np.mean(loss_per_epoch)]
            print(f"EPOCH: {epoch} after running {batch_index} batches >>> TOTAL LOSS: {train_loss[-1]}")
    
    def test_centre(self, test_data: GridBasket, end: int = 5):
        self.model.eval()
        
        for index, data in enumerate(test_data):
            if index == end:
                return 
            
            with torch.no_grad():
                inputs, true_y, x_grid, y_grid = data['input'].to(device), data['output'].to(device), data['x_shape'].to(device), data['y_shape'].to(device)
                prediction = self.model(inputs, x_grid, y_grid)
                x_coord = x_grid.argmax()
                y_coord = y_grid.argmax()
                grid_size = x_coord * y_coord 
                true_grid = true_y[:grid_size]
                pred_grid = prediction[:grid_size]
                green = torch.where(pred_grid == true_grid, 1, 0)
                correct_score = green.sum() / len(green)
                metrics += [correct_score.item()]
                print(f"Pred grid:\n{pred_grid}\nTrue grid:\n{true_grid}\ncorrect (%): {correct_score * 100:.5f}%\n")

torch.mps.empty_cache() 
data = GridBasket()
railway = CustomTrainer(lr=learning_rate)
railway.train_centre(data.train_data, max_epoch=2, batch_size=1)
railway.test_centre(data.test_data)