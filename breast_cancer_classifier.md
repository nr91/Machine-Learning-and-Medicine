Breast FNA Biopsy Classification using Neural Network
================
***NR***

Breast cancer is one of the most common malignancies and can present as a breast mass. One of the many methods of diagnosis is a fine needle aspiration (FNA) biopsy of the mass. Analysis of the cellular elements of the aspirate can help rule in or rule out a malignancy. The [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) contains features of cell nuclei obtained from images of fine needle aspirates of breast masses. We will apply a neural network model to classify the mass as benign or malignant based on the characteristics of the nuclei of cells found in the aspirate.

We will use PyTorch, a Python package, via Reticulate. Reticulate enables use of Python functions and modules within R as well as interaoperability between R and Python objects. In this example, I will perform all data processing and analysis in R and use PyTorch to create and train a model.

Import and Preprocess data
--------------------------

#### Libraries/packages

R packages

``` r
library(reticulate)
use_python(python_dir) # Insert your python location
library(DataExplorer)
library(dplyr)
library(ggplot2)
library(pROC)
library(corrplot)
```

Python libraries

``` python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
```


#### Import data from csv

``` r
data <- read.csv('data.csv', header = T)
head(data)
```


#### Missing values

``` r
plot_missing(data)
```

![](bc_nn_git_files/figure-markdown_github/missing-1.png)



#### Preprocess

``` r
# Remove unneeded features or features with missing values  
data <- data %>% select(-c(X, id))

# Rename diagnosis levels
levels(data$diagnosis) <- c('Benign','Malignant')
```



#### Class distribution

``` r
ggplot(data, aes(x = diagnosis)) +
  geom_bar(stat = 'count', fill = 'steelblue', width=0.6) +
  labs(x = 'Diagnosis', y = 'Count')
```

![](bc_nn_git_files/figure-markdown_github/classes-1.png) <br>

#### Heatmap of correlations

``` r
corrplot(cor(as.matrix(data %>% mutate(diagnosis = as.numeric(diagnosis)))), method = 'color')
```

![](bc_nn_git_files/figure-markdown_github/corrplot-1.png) <br>

Neural network model
--------------------

#### Train/Test split

Because the dataset is small, we will apply an 80%/20% split. We will convert all data frames to matrices to ensure a simple conversion to numpy arrays for use in Python. In addition, we will modify the train and test labels to convert them to zero-based ordering used in Python.

``` r
# Set seed
set.seed(1)

# Convert labels to numbers
data$diagnosis = as.numeric(data$diagnosis)

# 80% - 20% split
split = 0.2
n <- nrow(data)

# Shuffle data
shuffled <- data[sample(n),]
train_indices <- (round(split*n)+1):n
test_indices <- 1:round(split*n)

# Shuffled train and test set
train <- shuffled[train_indices,]
test <- shuffled[test_indices,]

# Separate features from labels
train_features <- train %>% select(-diagnosis)
train_labels <- train %>% select(diagnosis) 
test_features <- test %>% select(-diagnosis)
test_labels <- test %>% select(diagnosis)

# Convert data frames to matrices and subtract 1 from labels for zero-based ordering
train_features <- data.matrix(train_features) 
train_labels <- data.matrix(train_labels) - 1
test_features <- data.matrix(test_features) 
test_labels <- data.matrix(test_labels) - 1
```


#### Convert features and labels for use in PyTorch
We will convert the train and test datasets into tensors and input them into a dataloader

``` python
#### Python Code
# Convert numpy arrays to tensors
train_features = torch.from_numpy(r.train_features.astype('float32'))
train_labels = torch.from_numpy(r.train_labels.astype('float32'))
test_features = torch.from_numpy(r.test_features.astype('float32'))
test_labels = torch.from_numpy(r.test_labels.astype('float32'))  

# Load data into pytorch dataloader
trainset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(dataset=trainset, batch_size=20, shuffle=True)
testset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(dataset=testset, batch_size=20, shuffle=False)
```


#### Create model
We will use a simple feed-forward neural network with a hidden layer of 100 units followed by a hidden layer of 50 units and finally an single output layer. We will use relu activation functions for the hidden layers and a sigmoid activation for the output layer.

``` python
#### Python Code
# Set seed
torch.manual_seed(1)

# Set device type
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

# Network
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(30, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
model = net()
```


#### Network Parameters
We will use the Adam optimizer with a learning rate of 0.001. Using a scheduler, we will apply a learning rate decay of 10% for every 100 epochs. Our loss function will be binary crossentropy.

``` python
#### Python Code
# Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Loss fxn
criterion = nn.BCELoss()
```


#### Train model
We will train the model for 200 epochs

``` python
#### Python Code
epochs = 200
# List of losses
train_losses=[]
test_losses=[]
for epoch in range(epochs):
    # Loop through train data
    model.train()
    train_loss=0
    train_corrects=0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        preds = torch.round(outputs)
        train_corrects += preds.eq(labels).sum().item()
        optimizer.step()
    
    train_acc = train_corrects/len(train_loader.dataset)
    train_loss = train_loss/len(train_loader)
    train_losses.append(train_loss)
  
    # Loop through test data
    model.eval()
    test_loss=0
    test_corrects=0
    with torch.no_grad():
      for features, labels in test_loader:
          features, labels = features.to(device), labels.to(device)
          outputs = model(features)
          loss = criterion(outputs, labels)
          test_loss += loss.item()
          preds = torch.round(outputs)
          test_corrects += preds.eq(labels).sum().item()

      test_acc = test_corrects/len(test_loader.dataset)
      test_loss = test_loss/len(test_loader)
      test_losses.append(test_loss)

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}".format(epoch + 1, train_loss, train_acc, test_loss, test_acc))
```


#### Training results:

    ## Epoch 200: Loss: 0.085, Accuracy: 96.044%, Test Loss: 0.144, Test Accuracy: 94.737%

``` r
# Plot of train and test losses over the 200 epochs
# Obtain train and test losses
train_losses <- py$train_losses
test_losses <- py$test_losses
epochs <- 1:length(train_losses)
loss <- data.frame(train_losses, test_losses, epochs)

# Plot train and test loss
ggplot(loss, aes(epochs)) + 
  geom_smooth(aes(y=train_losses, color='Train Loss'), method='loess', se=F) + 
  geom_smooth(aes(y=test_losses, color='Test Loss'), method='loess', se=F) +
  theme(legend.title = element_blank()) +
  labs(x='Epochs', y='Loss')
```

![](bc_nn_git_files/figure-markdown_github/unnamed-chunk-2-1.png) <br>

Analysis of Results
-------------------

``` python
# Obtain predictions on test data
outputs = model(test_features)
preds = outputs.detach().numpy()
```

#### AUC

``` r
plot.roc(as.numeric(test_labels), as.numeric(py$preds), print.auc=T)
```

![](bc_nn_git_files/figure-markdown_github/auc-1.png)

With our neural net classifier, we obtain a classification accuracy of 94.7% and an AUC of 0.98. Not bad! In summary, the features of cell nuclei in a FNA biopsy can tell us whether a breast mass is likely to be benign or malignant with remarkable accuracy. Of course, determining these features on individual biopsies requires a great deal of time, money, and effort. A more practical approach would be to train a convolutional network on the fine needle aspirate images themselves with the rational that the model will "learn" the most important cell nuclei features for diagnosis.

<br>

#### Advice for Improvement?

If you have any specific feedback, please let me know. I am looking to further improve my programming skills and would appreciate any advice.
