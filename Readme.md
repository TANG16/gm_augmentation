
# Data augmentation using generative model
The code in this repository is related to the unpublished paper by Axelsen et al with the title 'Data Augmentation Based on Generative Model'.

The key function for generating synthetic data which can be used for augmentation is `naive_bayes`.
This if composed as a class, which needs to be trained on training data before it can either predict or generate.
```
Xtrain = [N x D]
Ytrain = [N x 1]
nb = naive_bayes(Xtrain,Ytrain);

Xtest = [Ntest x D]
[Ypred,posterior_prob] = nb.predict(Xtest)

gN = 1 x 1
[Xsynth,Ysynth] = nb.generate(gN)
```
## Run example
Calling the function `main` with proper inputes, e.g. 
```
ds = 'ionosphere';
gN = 100;
main(ds,gN);
```
will download and save the Ionosphere data set from the [UCI database](https://archive.ics.uci.edu/ml/index.php) and run a learning curve comparison of a naïve Bayes model, a logistic regression model, and a logistic regression model where the training set is augmented with `gN` synthetic samples. All trained on the dataset `ds`.
Calling `main` assumes or creates the following file structure:
```
/ [root]
├── code
│   ├── main.m
│   ├── naive_bayes.m
│   └── utils 
│       ├── download_dataset.m 
│       ├── stratifiedLC.m
│       └── UCIaugment.m
├── data
│   ├── [download location of datafiles from UCI]
|   └── result_data
|       └── mat-files from running the algorithm
└── results
    └── [learning curve plots]
```
If `main` is called several times using the same data set but different number of generated samples (`gN`) the learning curves will be shown in the same plot and the plot is printed to the `/results/` folder.
