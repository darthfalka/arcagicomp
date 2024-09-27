import torch         
import torch.nn.functional as F  
import numpy as np   
from torch.utils.data import IterableDataset  
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
    def __init__(self):
        import os 
        import glob 
        import json 
        
        directory = os.getcwd()
        if directory.split('/')[-1] == 'arcagi':
            print('In expected directory (arcagi)')
            self.checkpoint_path = f"{directory}/output/checkpoint"
        else: 
            raise OSError(f"Directory error expected to be in containing directory 'arcagi' but running in directory: {directory}")

        glob_path = f'{directory}/ARC-AGI/data/training/*.json'

        data = []
        for item in glob.glob(glob_path):
            with open(item, 'r') as file:
                data += [json.load(file)]
                
        self.train_data = DataPipe([j for i in data for j in i['train']])
        self.test_data = DataPipe([j for i in data for j in i['test']])

        self.color_pallet = color_pallet
        self.max_grid_size = max_grid_size # maybe: add 1 to include index sizes of range(0, 30). If 30 then the only available indices are in range(0, 29)
        self.max_flatten_size = max_flatten_size # 900 max_grid_size * max_grid_size
        self.default_hidden_dim = default_hidden_dim

if __name__ == '__main__':
    data = GridBasket()

    for i in data.train_data:
        print(f"""
            Input shape:      {i['input'].shape}
            Output shape:     {i['output'].shape}
            x_shape:          {i['x_shape'].shape}
            y_shape:          {i['y_shape'].shape}
            """)
        break
    