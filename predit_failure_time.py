import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pandas as pd
from sklearn.model_selection import KFold

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv('processed_data_yz.csv')
inputs = data.iloc[:, [2,4,6]].values # 2: y-amplitude, 4: z-amplitude, 6: growth rate
targets = data.iloc[:, -1].values

# Convert NumPy arrays to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
targets = torch.tensor(targets, dtype=torch.float32).to(device)

# Normalize the data
inputs_mean = inputs.mean(dim=0)
inputs_std = inputs.std(dim=0)
inputs = (inputs - inputs_mean) / inputs_std

# Create the MLP model, inheriting from the nn.Module class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.Sigmoid() # same as the logistic function
        self.output_activation = nn.Identity()
        # Random weight initialization
        init.kaiming_normal_(self.input_layer.weight)
        init.kaiming_normal_(self.hidden_layer.weight)

    def forward(self, x):
        x = self.input_layer(x) # apply the first linear transformation to the input x
        x = self.hidden_activation(x) # introduce non-linearity in order to learn complex patterns
        x = self.hidden_layer(x)
        return x

# Define the loss function and optimizer
learning_rate = 0.85 # determines the step size at which the optimizer updates the model's weights during training
num_epochs = 50000 # number of times the model will see the entire training dataset
hidden_layer_size = 25 # num of neurons in the hidden layer

model = MLP(inputs.size(1), hidden_layer_size, 1).to(device)  # Input size is the number of input features
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

# Perform k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True)
rmse_scores = []

for train_index, test_index in kf.split(inputs):
    # Split the data into training and test sets
    inputs_train, inputs_test = inputs[train_index], inputs[test_index]
    targets_train, targets_test = targets[train_index], targets[test_index]
    
    # Convert the data to PyTorch tensors
    inputs_train = inputs_train.clone().detach() #torch.tensor(inputs_train, dtype=torch.float32)
    targets_train = targets_train.clone().detach() # torch.tensor(targets_train, dtype=torch.float32)
    inputs_test = inputs_test.clone().detach() # torch.tensor(inputs_test, dtype=torch.float32)
    target_test = targets_test.clone().detach().reshape(-1, 1) # torch.tensor(targets_test, dtype=torch.float32)

    # print(inputs_train.shape, targets_train.shape, inputs_test.shape, targets_test.shape)

    # Define the loss function and optimizer
    criterion = nn.MSELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs_train).squeeze()
        loss = criterion(outputs.squeeze(), targets_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        if epoch % 1000 == 0:
            #print(outputs.shape, targets_train.shape)
            # Compute and print the gradient norm
            gradient_norm = 0.0
            for param in model.parameters():
                gradient = param.grad
                gradient_norm += gradient.norm(2).item() ** 2
            gradient_norm = gradient_norm ** 0.5
            print(f"Epoch:{epoch}, Loss: {round(loss.item(), 2)}, Gradient Norm: {round(gradient_norm,2)}")

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # helps prevent the exploding gradient problem in RNNs / LSTMs

        optimizer.step()
    
    # Test the model on the test set
    with torch.no_grad():
        outputs_test = model(inputs_test)
        rmse = torch.sqrt(criterion(outputs_test, targets_test))
        print("RMSE:", rmse.item())
        rmse_scores.append(rmse.item())

# Calculate the average RMSE score
avg_rmse = sum(rmse_scores) / k
print("Average RMSE:", avg_rmse)

