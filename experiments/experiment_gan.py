# trial_2 - idea is that instead of improving a classification model by building a net that can recreate the predicting pattern instead of classifying the correct ones
# TODO: 
# - use eval tools instead of manually evaluating the built model
# - reconsider why do i need to import DataLoader when I have already built a data iterator customed to my preferences

import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import numpy as np 
from datasets import Dataset

directory = os.getcwd()
if directory.split('/')[-1] == 'arcagi':
    print('In expected directory (arcagi)')
    checkpoint_path = f"{directory}/output/checkpoint"
else: 
    raise OSError(f"Directory error expected to be in containing directory 'arcagi' but running in directory: {directory}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

color_pallet = 9
max_grid_size = 30 # maybe: add 1 to include index sizes of range(0, 30). If 30 then the only available indices are in range(0, 29)
max_flatten_size = 900 # 900 max_grid_size * max_grid_size
default_hidden_dim = max_flatten_size // 2

class Head(nn.Module):
    def __init__(self, input_dim: int = max_flatten_size, hidden_dim: int = default_hidden_dim, learning_rate: float = 0.001):
        super(Head, self).__init__() 
        self.xemb = nn.Embedding(num_embeddings=max_grid_size, embedding_dim=max_grid_size, device=device) # 30, 900
        self.yemb = nn.Embedding(num_embeddings=max_grid_size, embedding_dim=max_grid_size, device=device) # 30, 900
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True, device=device)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x_grid_vec: torch.Tensor, y_grid_vec: torch.Tensor, inputs: torch.Tensor):
        # NOTE: below is only for if you are not passing batches
        # assert inputs.shape[0] == max_flatten_size and len(inputs.shape) == 1, f"Expected flattened array but received {inputs.shape}"
        assert x_grid_vec.shape[-1] == y_grid_vec.shape[-1] == max_grid_size, \
            f"Expected grid size vector to be of {max_grid_size} shape but received \nx: {x_grid_vec.shape}\ny:{y_grid_vec.shape}"
        assert len(inputs.shape) == 2 and inputs.shape[-1] == max_flatten_size, f"Expected X input shape [batch_size, 900] but received: {inputs.shape}"
        
        x_emb = self.xemb(x_grid_vec) # [1, 30, 30], with batch: [batch_size, 1, 30, 30]
        y_emb = self.yemb(y_grid_vec) # [1, 30, 30], with batch: [batch_size, 1, 30, 30]
        
        grid_embed = torch.sum(x_emb * y_emb, dim = 0) # [30, 30], with batch: [1, 30, 30]
        embedding = grid_embed.flatten(1) # with/out shape is [1, 900] 
        embedding = embedding.expand(inputs.shape[0], -1) # [batch_size, 1, 900]
        
        assert embedding.shape == inputs.shape, f"Expected embedding shape to be the same as inputs shape but received: embedding: {embedding.shape} and inputs: {inputs.shape}"
        
        #output = attention(inputs, embedding) # [900, 900]
        output = (embedding * inputs).tanh() # [batch_size, 900]
        output = output.unsqueeze(1) # [batch_size, 1, 900]
        hidden_layer = self.fc1(output) # [batch_size, 1, 450]
        return hidden_layer.squeeze(1)

class Tail(nn.Module):
    def __init__(self, input_dim: int = default_hidden_dim, learning_rate: float = 0.001, output_dim: int = max_flatten_size):
        super(Tail, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, device=device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, hidden: torch.Tensor):
        assert len(hidden.shape) == 3 and hidden.shape[-1] == default_hidden_dim, f"Expected hidden shape [batch_size, 450] but received {hidden.shape}"
        return self.fc(hidden) # [batch_size, 1, 900]

class DataPipe(IterableDataset):
    @staticmethod
    def configure_pad(input_dim: int):
        # input dim is the flattened shape of the vector
        
        # TODO: Instead of padding in the middle, can insert it from the beginning since the vec is flattened anyway
        dim = max_flatten_size - input_dim 
        
        if dim % 2 == 1:
            # odd
            dim = dim - 1

        top_padding = dim // 2 # 445
        bot_padding = top_padding + input_dim if input_dim % 2 == 0 else top_padding + input_dim + 1
        output = [top_padding, bot_padding]
        
        assert np.sum(output) == max_flatten_size, f"Expected padding tuple sum to be {max_flatten_size} but received {np.sum(output)}"
        return output
    
    @staticmethod
    def preprocess_pad_left(inputs: torch.Tensor):
        # this pads the inputs from its flattened dimensions (left). Replaces remaining with 0s after flattened_dim st. it returns array shape [1, 900]
        flatten_input = inputs.flatten()
        flatten_dim = flatten_input.shape[0]
        
        assert flatten_dim <= max_flatten_size, f"Received unusual dimensions for the flattened input: {flatten_input.shape}\nfor array: {flatten_input}"
        
        padded = F.pad(flatten_input, (0, max_flatten_size - flatten_dim))
        padded = padded.unsqueeze(0)
        
        assert tuple(padded.shape) == tuple((1, max_flatten_size)), f"Incorrect size for padded vec: {padded.shape}"
        return padded
    
    @staticmethod
    def preprocess(inputs: torch.Tensor):
        assert len(inputs.shape) == 2, f"Only accept grid n x m but received input of shape: {inputs.shape}"
        
        flatten_input = inputs.flatten()
        flatten_dim = flatten_input.shape[0]
        pad_tuple = DataPipe.configure_pad(flatten_dim)
        pad_tuple[0] -= flatten_dim
        padded = F.pad(flatten_input, pad_tuple, 'constant')
        padded._dtype = torch.int64
        
        assert inputs.shape[0] * inputs.shape[-1] == flatten_input.shape[0], f"Flatten shape must equal {inputs.shape[0] * inputs.shape[-1]} but received flatten input of shape: {flatten_input.shape}"
        assert padded.shape[0] == max_flatten_size, f"Incorrect padded shape : {padded.shape}"
        return padded.unsqueeze(0) # [1, 900]
    
    @staticmethod
    def vectorise_grid(grid_size: int):
        grid_size = grid_size - 1
        idx_vec = torch.zeros(max_grid_size, dtype=torch.int, device=device)
        idx_vec[grid_size] = 1
        vec = idx_vec.unsqueeze(0)
        
        assert idx_vec.sum() == 1, f"Expected sum of vector to be 1 but received {idx_vec.sum()}\n Output: {idx_vec}"
        assert list(vec.shape) == [1, 30], f"Expected shape to be [1, 30] but received {vec.shape}"
        return vec 
    
    def __init__(self, dataset: list):
        """
        Args:
            inputs: A tensor containing the input data (e.g., shape [N, M]).
            outputs: A tensor containing the corresponding labels or outputs.
        """
        # convert to tensor arrays 
        self.data = [{keys: torch.tensor(vals, dtype=torch.float, device=device, requires_grad=True) for keys, vals in i.items()} for i in dataset]

        assert set(self.data[0].keys()) == set(['input', 'output']), f'Expected keys to be [input, output] but received: {self.data[0].keys()}'
        assert isinstance(self.data[0]['input'], torch.Tensor) and isinstance(self.data[0]['output'], torch.Tensor), f"Incorrect dtype for input data: {self.data[0]} where values of {type(self.data[0]['input'])}"
        
    def __iter__(self):
        # returns features 
        # x_shape [1, 30], y_shape [1, 30], input (padded) [1, 900], output (padded) [1, 900]

        for items in self.data:
            if 'x_shape' not in items or 'y_shape' not in items:
                # NOTE: transformations to the items are saved in self.data 
                items['x_shape'] = DataPipe.vectorise_grid(items['input'].shape[0]) # [1, 30]
                items['y_shape'] = DataPipe.vectorise_grid(items['input'].shape[-1]) # [1, 30]
                items['input'] = DataPipe.preprocess_pad_left(items['input']) # [1, 900]
                items['output'] = DataPipe.preprocess_pad_left(items['output']) # [1, 900]

            assert set(items.keys()) == set(['x_shape', 'y_shape', 'input', 'output']), f'Incorrect keys in preprocessed data. Listed keys in the data: {list(items.keys())}'
            assert tuple(items['input'].shape) == tuple(items['output'].shape), f"Expected input and output are of the same shape but received input shape {items['input'].shape} and output shape: {items['output'].shape}"
            
            yield items

class GridBasket:
    """
    Example of usage:
        grids = GridBasket()
        for i in grids.train_data:
            print(i)
            break
    """
    def __init__(self):
        import glob 
        import json 
        
        glob_path = f'{directory}/ARC-AGI/data/training/*.json'

        data = []
        for item in glob.glob(glob_path):
            with open(item, 'r') as file:
                data += [json.load(file)]
                
        self.train_data = DataPipe([j for i in data for j in i['train']])
        self.test_data = DataPipe([j for i in data for j in i['test']])

class CandyShop:
    @staticmethod
    def create_z_data(x: torch.Tensor):
        assert len(x.shape) == 2, f'Expected num of dimensions of input x to be [batch_size, 900] but received {x.shape}'
        
        output_size = x.shape[-1]
        cat_num = color_pallet + 1
        
        freq = [torch.bincount(row_data.long(), minlength=cat_num).cpu().numpy() for row_data in x]
        proba = [i / i.sum() for i in freq] 
        mock_data = torch.tensor(np.stack([np.random.choice(range(0, cat_num), (1, output_size), p=i) for i in proba]), dtype=torch.float, device=device) # [10, 1, 900]
        mock_data = mock_data.permute(1, 0, 2).squeeze(0) # [batch_size, 900]

        assert tuple(x.shape) == tuple(mock_data.shape), f"Input x shape ({x.shape}) must be identical to mock_data shape: {mock_data.shape}"
        assert tuple(mock_data.shape) == tuple(x.shape), f"Expected mock data shape to be identical to input's shape (shape: {x.shape}) but received : {mock_data.shape}"
        return mock_data.unsqueeze(1)
    
    def __init__(self):
        self.encoder = Head()
        self.decoder = Tail()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def run(self, X, x_vec, y_vec, labels = None):
        # X & labels shape: [batch_size, 900]; x_vec & y_vec shape: [batch_size, 30]
        encoder_output = self.encoder(x_vec, y_vec, X) # [batch_size, 450]
        encoder_output = encoder_output.unsqueeze(1) # [batch_size, 1, 450]
        decoder_output = self.decoder(encoder_output) # [batch_size, 1, 900]
        #print(f"Hidden shape: {encoder_output.shape} \tDecoder output at running trainer: {decoder_output.shape}\t target label shape: {labels.shape}")
        return self.criterion(decoder_output, labels), encoder_output if labels is not None else decoder_output
    
    def train(self, train_data: DataPipe, **kwargs: dict) -> None:
        assert all(key in kwargs for key in ['max_epochs', 'batch_size']), f"Must pass ['max_epochs', 'batch_size', 'schedular'] inside trainer: {kwargs}"
        
        self.encoder.train()
        self.decoder.train()

        train_loss = [] #real_y_labels = torch.ones_like(y, dtype=torch.int, requires_grad=False, device=device)
        data_loader = DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=False)
        #fake_y_labels = torch.zeros((kwargs['batch_size'], 1, max_flatten_size), dtype=torch.int, requires_grad=False, device=device)
        
        for epoch in range(kwargs['max_epochs']):
            epoch_decoder_loss = []
            epoch_encoder_loss = []
            
            for batch_index, data in enumerate(data_loader):
                X, real_y_labels, x_vec, y_vec = data['input'].to(device), data['output'].to(device), data['x_shape'].to(device), data['y_shape'].to(device) # [batch_size, 900], [batch_size, 900], [10, 1, 30], [10, 1, 30] where 10 is the batch size
                real_y_labels._requires_grad = False 
                
                # zero grad of the optimizer for decoder only
                self.encoder.optimizer.zero_grad()
                self.decoder.optimizer.zero_grad()
                
                X = X.permute(1, 0, 2).squeeze(0) # [batch_size, 900]
                real_y_labels = real_y_labels.permute(1, 0, 2).squeeze(0) # [batch_size, 900]
                
                fake_encoder_input = CandyShop.create_z_data(X) # z creation with shape: [batch_size, 1, 900]
                fake_y_labels = CandyShop.create_z_data(real_y_labels) # [batch_size, 1, 900]

                real_decoder_loss, real_encoder_output = self.run(X, x_vec, y_vec, real_y_labels.unsqueeze(1).float()) # insert REAL dataset into encoder and decoder 
                fake_decoder_loss, _ = self.run(fake_encoder_input.squeeze(1), x_vec, y_vec, fake_y_labels.float()) # insert FAKE dataset into encoder and decoder

                # real vs fake decoder loss 
                decoder_loss = max(real_decoder_loss, fake_decoder_loss)
                decoder_loss.backward()
                self.decoder.optimizer.step()
                
                # inserting fake encoder output into decoder 
                fake_encode_X = torch.randn_like(real_encoder_output, dtype=torch.float, device=device)
                fake_output = self.decoder(fake_encode_X)
                fake_encoder_loss = self.criterion(fake_output, fake_y_labels)
                
                fake_encoder_loss.backward()
                self.encoder.optimizer.step()
                
                if batch_index % 3 == 0 or batch_index == 0:
                    print(f"\tEPOCH {epoch} [[Batch:  {batch_index}]] \tDecoder loss: {decoder_loss:<.10} \tEncoder loss: {fake_encoder_loss}")
                
                epoch_decoder_loss += [decoder_loss.item()]
                epoch_encoder_loss += [fake_encoder_loss.item()]
                
            train_loss += [{'d_loss': np.mean(epoch_decoder_loss), 'e_loss': np.mean(epoch_encoder_loss)}]
            #print(f">>> Mean loss\t Decoder: {train_loss[-1]['d_loss']}\t Encoder: {train_loss[-1]['e_loss']}")

        print(f"{'>'*20} ... Training Loss Summary ... \n{'>'*20} D_Loss: {[i['d_loss'] for i in train_loss]} \n{'>'*20} E_Loss: {[i['e_loss'] for i in train_loss]}")
        
    def _uncook_testing_data(self, x_shape: torch.Tensor, y_shape: torch.Tensor, padded_test_pred: torch.Tensor, padded_true_pred: torch.Tensor):
        metrics = []
        data_size = padded_test_pred.shape[0]
        
        print(f"Total data for model to eval is {data_size}")
        for index in range(data_size):
            x = x_shape[index].argmax()
            y = y_shape[index].argmax()
            flatten_dim = x * y
            unpad_pred_grid_ = padded_test_pred[index].sigmoid()[0:flatten_dim] # [flatten_dim]
            unpad_true_grid_ = padded_true_pred[index][0:flatten_dim] # [flatten_dim]
            unpad_pred_grid = torch.where(unpad_pred_grid_ >= unpad_pred_grid_.mean(), 1.0, 0)
            unpad_true_grid = torch.where(unpad_true_grid_ != 0, 1.0, 0)
            pred_grid = unpad_pred_grid.reshape((x, y))
            true_grid = unpad_true_grid.reshape((x, y))
            green = torch.where(unpad_pred_grid == unpad_true_grid, 1, 0)
            correct_score = green.sum() / len(green)
            metrics += [correct_score.item()]
            print(f"Pred grid:\n{pred_grid}\nTrue grid:\n{true_grid}\ncorrect (%): {correct_score * 100:.5f}%\n")
            
        print(f">>> AVG score over {data_size} tests: {np.mean(metrics)}")
        
    def test(self, test_data: list):
        # note that dataset puts all tensors into list...
        testing_data = Dataset.from_list(test_data)
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            inputs = torch.tensor(np.array(testing_data['input']), dtype=torch.float, device=device).squeeze(1) # [batch_size, 900]
            x_shape = torch.tensor(np.array(testing_data['x_shape']), dtype=torch.int, device=device) # [batch_size, 1, 30]
            y_shape = torch.tensor(np.array(testing_data['y_shape']), dtype=torch.int, device=device) # [batch_size, 1, 30]
            true_output = torch.tensor(np.array(testing_data['output']), dtype=torch.float, device=device).squeeze(1) # [batch_size, 900]
            
            encoded_output = self.encoder(x_shape, y_shape, inputs) # [batch_size, 450]
            decoded_output = self.decoder(encoded_output.unsqueeze(1)) # [batch_size, 1, 900]
            decoded_output = decoded_output.squeeze(1) # [batch_size, 900]
        
        assert tuple(decoded_output.shape) == tuple(true_output.shape), f"Expected identical shape for model's output shape:{decoded_output.shape} and true shape: {true_output.shape}"
        return self._uncook_testing_data(x_shape, y_shape, decoded_output, true_output)
        
def run_trial():
    max_epochs = 10
    batch_size = 500 # higher the batch size, the better the results
    data = GridBasket()
    model = CandyShop()
    print('training.....')
    model.train(data.train_data, max_epochs=max_epochs, batch_size=batch_size)
    print('\ntesting...')
    model.test(list(data.test_data)[:10])
    print('\n\n thanks for playing with me!')

run_trial()