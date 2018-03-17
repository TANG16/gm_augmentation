
# Data augmentation using generative model
The code in this repository is related to the paper by Axelsen et al with the title 'Data Augmentation Using Generative Model' which is submitted to IEEE TPAMI.

The key function for generating synthetic data which can be used for augmentation is `naive_bayes`.
This if composed as a class, which needs to be trained on training data before it can either predict or generate.

Calling the function `main` with proper inputes, e.g. 
````
ds = ionosphere;
gN = 100;
main(ds,gN);
```
will download and save the Ionosphere data set from the [UCI database](https://archive.ics.uci.edu/ml/index.php) and run a learning curve comparison of a naïve Bayes model, a logistic regression model, and a logistic regression model where the training set is augmented with `gN` synthetic samples. All trained on the dataset `ds`.
Calling `main` assumes following the file structure of the repository:
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
│   └── [download location of datafiles from UCI]
└── results
    ├── [learning curve plots]
    └── result_data
        └── mat-files from running the algorithm
```
If the function is called several times using the same data set but different number of generated samples (`gN`) the learning curves will be shown in the same plot and the plot is printed to the `/results/` folder.
