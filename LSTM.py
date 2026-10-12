import torch
import torch.nn as nn
from data_loader import *
from utils.YParams import YParams
import matplotlib.pyplot as plt
import yaml

# Build the DataLoaders.
#   params: Config and command line parameters.
#   batch_size: The batch size requested for the data loader.
#   return: The Datasets and DataLoaders for the training,
#       validation, and test data.
def build_DataLoaders(params):
    trainDL, trainDS = get_data_loader(params, split="train", shuffle=True)
    validDL, validDS = get_data_loader(params, split="valid", shuffle=True)
    testDL, testDS = get_data_loader(params, split="test", shuffle=True)
    return (trainDS, validDS, testDS, trainDL, validDL, testDL)

# Print a summary of the Dataset and DataLoader information.
#   config: The name of the dataset configuration being used for prediction.
#   trainDS: The dataset for the training split.
#   validDS: The dataset for the validation split.
#   testDS: The dataset for the test split.
#   trainDL: The DataLoader for the training split.
#   ValidDL: The DataLoader for the validation split.
#   testDL: The DataLoader for the test split.
def print_summary(config, trainDS, validDS, testDS, trainDL, validDL, testDL):
    print("\n========== DATASET SPLIT INFORMATION ==========")
    print(f"Config:               {config}")
    print(f"Task:                 {params.get("task")}")
    print(f"Training:")
    print(f"  Samples:            {len(trainDS)}")
    print(f"  Batch size:         {trainDL.batch_size}")
    print(f"  Number of batches:  {len(trainDL)}")
    print(f"Validation:")
    print(f"  Samples:            {len(validDS)}")
    print(f"  Batch size:         {validDL.batch_size}")
    print(f"  Number of batches:  {len(validDL)}")
    print(f"Testing:")
    print(f"  Samples:            {len(testDS)}")
    print(f"  Batch size:         {testDL.batch_size}")
    print(f"  Number of batches:  {len(testDL)}")
    print("=================================================\n")


class Model(nn.Module):

    # Initialize the model.
    #   hidden_size: Number of LSTM cells.
    #   input_size: Size of the input data (19 sequence events, 300 static events).
    #   dropout_rate: Dropout rate of the last fully connected layer.
    def __init__(self, hidden_size: int, input_size:int=319, dropout_rate: float=0.0):

        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=1, 
            bias=True,
            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    # Forward pass.
    #   seq: Tensor containing the sequence events.
    #   static: Tensor containing the static events.
    #   return: Tensor containing the network predictions.
    def forward(self, seq: torch.Tensor,  static: torch.Tensor) -> torch.Tensor:
        input = seq
        static_repeat= static.unsqueeze(1).repeat(1, seq.shape[1], 1)
        input = torch.cat((seq, static_repeat), -1)

        output, (h_n, c_n) = self.lstm(input)

        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred


# Train and evaluate the model for a specified number of epochs.
#   model: The LSTM model.
#   optimizer: A PyTorch optimizer class.
#   trainDL: The DataLoader for the training data.
#   validDL: The DataLoader for the validation data.
#   loss_func: The loss function.
#   n_epochs: The number of epochs to train for.
#   return: model, train_losses (an array of the training losses
#       from each epoch), valid_losses (an array of the validation
#       losses from each epoch)
def train(model, optimizer, trainDL, validDL, loss_func, n_epochs):

    t_losses, v_losses = [], []
    tLoss, tCount = 0.0, 0
    vLoss, vCount = 0.0, 0
    for i in range(n_epochs):

        model.train()                                       # Set model to train mode

        for x in trainDL:
            seq_input = x["input"]
            static_input = x["static"]
            target = x["output"]

            if target.dim() > 1:
                target = torch.nan_to_num(target, nan=0.0)

            optimizer.zero_grad()                           # Clear gradients
            p = model(seq_input, static_input)              # Get model prediction
            loss = loss_func(p, target.unsqueeze(1))        # Calculate loss
            loss.backward()                                 # Calculate gradients
            optimizer.step()                                # Update weights
            tLoss += loss.item()
            tCount = tCount+1

        tLoss = tLoss/tCount                                # Calculate training loss

        model.eval()                                        # Set model to evaluate mode
        with torch.no_grad():
            for j in validDL:
                seq_input = j["input"]
                static_input = j["static"]
                target = j["output"]

                if target.dim() > 1:
                    target = torch.nan_to_num(target, nan=0.0)

                p = model(seq_input, static_input)          # Get model prediction
                loss = loss_func(p, target.unsqueeze(1))    # Calculate loss
                vLoss += loss.item()
                vCount = vCount+1
            vLoss = vLoss/vCount                            # Calculate validation loss

        t_losses.append(tLoss)
        v_losses.append(vLoss)

        print(f'Epoch [{i+1}/{n_epochs}], tLoss: {tLoss:.4f}, vLoss: {vLoss:.4f}')
        
    return model, t_losses, v_losses

# Test the model.
#   model: The LSTM model.
#   loader: The DataLoader for the test data.
#   return: Torch Tensors containing the targets and
#       model predictions.
def test_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for x in loader:
            seq_input = x["input"]
            static_input = x["static"]
            target = x["output"]

            if target.dim() > 1:
                target = torch.nan_to_num(target, nan=0.0)

            p = model(seq_input, static_input)              # Get model prediction
            targets.append(target)                          # Collect targets
            preds.append(p)                                 # Collect predictions

    return torch.cat(targets), torch.cat(preds)

# Plot the loss and accuracy from testing.
#   task: A string denoting the task to be performed.
#   train_losses: An array containing the training losses from training.
#   valid_losses: An array containing the validation losses from training.
#   targets: A Torch Tensor containing the target test values.
#   preds: A Torch Tensor containing the model predictions from testing.
def plot_results(task_somsc, train_losses, valid_losses, targets, preds):

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16, 6))

    fig.suptitle('Training Results', fontsize=16)

    ax1.set_title('Loss', fontsize=14)
    ax1.plot(train_losses, label="Training Loss", color="#747575", linewidth=2, linestyle='-')
    ax1.plot(valid_losses, label="Validation Loss", color="#c70c1f", linewidth=2, linestyle='-')
    ax1.legend(fontsize='large')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    ax2.set_title('Accuracy', fontsize=14)
    ax2.plot(targets, label="Target Values", color="#e47d08", linewidth=2, linestyle='--', marker='o', markersize=6)
    ax2.plot(preds, label="Model Predictions", color="#1071B6", linewidth=2, linestyle='-', marker='o', markersize=6)
    ax2.legend(fontsize='large')
    ax2.set_xlabel("Sample", fontsize=12)
    if task_somsc:
        ax2.set_ylabel("Normalized Monthly Soil Organic Carbon", fontsize=12)
    else:
        ax2.set_ylabel("Normalized Annual Yield", fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.show()



# ========================================  MAIN  ======================================== #

FIGURE_DIR = "docs/figures"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="config.yaml", type=str)
    parser.add_argument("--config", default="model_yield", type=str)
    args = parser.parse_args()
    params = YParams(args.yaml_config, args.config)

    with open(args.yaml_config) as f:
        configs = yaml.safe_load(f)
    if args.config not in configs:
        raise KeyError(f"Config {args.config!r} not found in {args.yaml_config}")
    config = configs[args.config]

    task_somsc = False
    if params.get("task") == "SOMSC":
        task_somsc = True

hidden_size = 25
dropout_rate = 1.0
learning_rate = 1e-4        
n_epochs = 15

print("Building DataLoaders...")
trainDS, validDS, testDS, trainDL, validDL, testDL = build_DataLoaders(params)
print_summary(args.config, trainDS, validDS, testDS, trainDL, validDL, testDL)

print("Initializing model...")
model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

print("Training Model...")
modelT, train_losses, valid_losses = train(model, optimizer, trainDL, validDL, loss_func, n_epochs)

print("Testing Model...")
targets, preds = test_model(modelT, testDL)

plot_results(task_somsc, train_losses, valid_losses, targets, preds)