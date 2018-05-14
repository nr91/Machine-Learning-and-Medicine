Entity Embeddings for interpretation of Clinical and Histopathological features in Dermatological Diseases
================
***NR***

In this repo, I will demonstrate how embeddings can be used to explore and visualize relationships between categorical variables. The idea of applying embeddings to categorical data was suggested by [Guo and Berkhahn](https://arxiv.org/abs/1604.06737). I was first introduced to the idea while watching the [Fast.ai](http://www.fast.ai/) course lectures, which I recommend to anyone interested in deep learning.

### What are entity embeddings?

Embeddings are representations of a piece of data projected to a n-dimensional space. In the case of entity embeddings, we are projecting a categorical vector to a lower dimension representation. Traditionally, categorical variables are encoded as independent variables using a method known as one-hot-encoding. Entity embeddings offer the advantage of being able to capture intricate relationships between the different categories within each feature.

### Dataset

We will generate entity embeddings for the features in the Dermatology Data Set in the UCI Machine Learning Repo. This dataset consists of 366 cases of erythemato-squamous diseases, which often share several similarities particularly on histopathology. There are 6 disease categories: psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris. The dataset looks at 34 features - 11 clinical features (i.e. age, presence of itching, location on body) and 22 histopathological features of tissue biopsy samples seen on microscopy. For more information on the dataset, check out <https://archive.ics.uci.edu/ml/datasets/dermatology>.

We will be using R and PyTorch via the [Reticulate R package](https://rstudio.github.io/reticulate/) for this project.

### Import useful libraries/packages

R packages:

``` r
library(reticulate)
use_python(python_dir) # Insert location to python
library(DataExplorer)
library(dplyr)
library(ggplot2)
library(ggrepel)
```

Python libraries:

``` python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
```

Load and preprocess the data
----------------------------

The data is stored in a csv file. Unfortunately, the names of the features are not in the file but are listed on the website. I have provided a modified csv with the appropriate column names.

``` r
# Load data, question marks are interpreted as missing values
data <- read.csv('dermatology.csv', header=T, na.strings = '?') 
head(data)
```

``` r
plot_missing(data)
```

![](catembed_Derm_files/figure-markdown_github/unnamed-chunk-2-1.png)

It appears **Age** has some missing variables - let's drop this feature. We will also drop **family.history** since the majority of the cases have no family history.

``` r
# Drop age and family history
data <- data %>% select(-c(Age, family.history))
```

<br> Now we are finished preprocessing our dataset. We will split the data into a train (80%) and test set (20%). We will also separate the features (x) from the label (y). We have 6 label classes (1 through 6), but because python uses zero-based-numbering, we will subtract 1 from the label vectors. We also obtain a list of the number of unique levels for each feature which we will use for creation of the embeddings.

``` r
# Set seed
set.seed(123)

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
train_x <- train %>% select(-Disease)
train_y <- train %>% select(Disease)
test_x <- test %>% select(-Disease)
test_y <- test %>% select(Disease)

# List of number of unique categories for each feature
cats <- data %>% select(-Disease) %>% lapply(as.factor) %>% lapply(nlevels) %>% as.integer

# Convert dataframe to matrices with zero-based ordering
train_x <- data.matrix(train_x) 
train_y <- data.matrix(train_y) - 1
test_x <- data.matrix(test_x) 
test_y <- data.matrix(test_y) - 1
```

<br>

Create a neural network classifier
----------------------------------

Now we will create a neural network classifier consisting of 32 embeddings, one for each feature, followed by a fully connected layer and an output layer. Most of our features have 4 levels (0, 1, 2, and 3), so we will use an embedding size of 2 for each feature to keep it simple. Using PyTorch's ModuleList, we create an embedding for each feature. The embeddings are concatenated and serve as the input for the first fully connected layer. This is followed by a softmax layer consisting of 6 outputs corresponding to the 6 possible conditions.

We will use Relu activation functions for each fully connected layer and softmax activation for the final output layer. We also use dropout on the penultimate layer to control overfitting. Our loss will be calculated using crossentropy. For training, we will use the SGD optimizer with a learning rate of 0.01.

``` python
#### Python code
# Set seed
torch.manual_seed(321)
# Set embedding size
embedding_size = 2
# Set device type
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
class model(nn.Module):
  def __init__(self,cats):
    super(model, self).__init__()
    # Create embedding for each feature
    self.embs = nn.ModuleList([nn.Embedding(k, embedding_size) for k in cats])
    self.fc1 = nn.Linear(64, 12)
    self.fc2 = nn.Linear(12, 6)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.2)
  def forward(self, x):
    # Input each feature into its respective embedding
    x = [embed(x[:,i]) for i, embed in enumerate(self.embs)]
    # Concatenate all embeddings
    x = torch.cat(x,-1)
    x = self.relu(self.fc1(x))
    x = self.drop(x)
    x = self.fc2(x)
    return x  
model = model(r.cats).to(device)  
# Loss function
loss_func = nn.CrossEntropyLoss()  
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

We will now proceed to load the data into a tensor dataset used by pytorch

``` python
#### Python Code 
# Convert numpy arrays to tensors
x_train = torch.from_numpy(r.train_x.astype('int64'))
y_train = torch.from_numpy(r.train_y.astype('int64')).view(-1,)
x_test = torch.from_numpy(r.test_x.astype('int64'))
y_test = torch.from_numpy(r.test_y.astype('int64')).view(-1,)  
# Load data into pytorch dataloader
trainset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True)
```

Train the model
---------------

We will train our classifier for 750 epochs

``` python
#### Python code
epochs=750
train_losses=[]
test_losses=[]
for epoch in range(epochs):
  train_loss=0
  running_corrects=0
  # Training loop
  model.train()
  for features, labels in train_loader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(features)
    loss = loss_func(outputs, labels)
    loss.backward()
    train_loss += loss.item()
    preds = torch.max(outputs, 1)[1]
    running_corrects += preds.eq(labels).sum().item()
    optimizer.step()
  train_acc = running_corrects/len(train_loader.dataset)
  train_loss = train_loss/len(train_loader)
  train_losses.append(train_loss)
  
  # Evaluation loop
  model.eval()
  with torch.no_grad():
    features, labels = x_test, y_test
    features, labels = features.to(device), labels.to(device)
    outputs = model(features)
    loss = loss_func(outputs, labels)
    test_loss=loss.item()
    test_losses.append(test_loss)
    preds = torch.max(outputs, 1)[1]
    corrects = preds.eq(labels).sum().item()
    test_acc = corrects/len(x_test)
  print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}".format(epoch + 1, train_loss, train_acc, test_loss, test_acc))
```

#### Training Results:

    ## Epoch 750: Loss: 0.074, Accuracy: 98.294%, Test Loss: 0.039, Test Accuracy: 98.630%

![](catembed_Derm_files/figure-markdown_github/metrics-1.png)

Extract and visualize trained embeddings
----------------------------------------

``` python
#### Python code
# obtain embedding weights
emb_weights = [embedding.weight.detach().numpy() for embedding in model.embs]
```

Currently, we have the embedding weights for each level of each feature. For visualization purposes, we will focus on features with a category of 3 (severe) and discard the other weights. We will incorporate the weights and feature labels into an R data frame, embed\_data, and plot the results.

``` r
head(embed_data)
```

    ##         Dim_1      Dim_2             Labels
    ## 1 -0.62775022 -1.1986362           erythema
    ## 2  0.07530504 -0.1837743            scaling
    ## 3 -0.61251748  0.3172307   definite.borders
    ## 4  0.14504303  0.4625579            itching
    ## 5  0.98426247  0.5630311 koebner.phenomenon
    ## 6  0.11390265  0.5921493  polygonal.papules

<br>

### Entity embeddings for clinical and histopathological features:

![](catembed_Derm_files/figure-markdown_github/plots%20of%20features-1.png)![](catembed_Derm_files/figure-markdown_github/plots%20of%20features-2.png)

The figures above are plots of the trained embedding weights for each clinical and pathological feature. Here, we get a glimpse of the complex associations between each of the features. Traditional methods of incorporating categorical data such as one-hot-encoding would not be able to capture such intricacies in the data. Here are some interesting points:

#### Clinical Features

-   Scaling, itching, and knee/elbow involvement are shown in close proximity to one another. These features are highly associated with psoriasis. Follicular papules, a relatively non-specific finding but also found in psoriasis, is nearby.
-   Oral mucosal involvement is a feature seen in Lichen planus but not with any of the other conditions in the dataset. It is understandable then that it is somewhat further away from the other points, which tend to be more associated with psoriasis and chronic dermatitis.
-   Polygonal papules is another feature that is highly specific to lichen planus. These papules tend to be itchy, hence its close proximity to itching.
-   Interestingly, erythema is seen quite far away from the other points. This may be due to the fact that it is a non-specific finding and thus does not have a strong tie to any of the other clinical features.

#### Histopathological Features

-   Several psoriatic features are in close proximity with one another. These include parakeratosis, suprapapillary dermis thinning, spongiform pustule, as well as disappearance of the granular layer.
-   Munro's micro abscesses are very strongly associated with psoriasis but not with any of the other skin conditions in this dataset. While it is close to several major psoriasis features (parakeratosis, suprapapillary dermis thinning, and spongiform pustule), its location in the rightmost region of the plot leaves is somewhat isolated from the rest of the features.
-   Lichen planus is notable for having a saw-tooth-appearance of the rete ridges along with a band-like inflammatory infiltrate that is primarily lymphocytic. All 3 of these features are shown in close proximity to one another.
-   Fibrosis of the papillary dermis is a feature associated with lichen planus. On the same plane is melanin incontinence. Patients with lichen planus can develop excess hyper pigmentation in the affected areas.

<br>

#### Advice for Improvement?

If you have any specific feedback, please let me know. I am looking to further improve my programming skills and would appreciate any advice.
