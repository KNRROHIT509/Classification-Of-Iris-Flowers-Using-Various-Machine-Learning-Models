# Classification-Of-Iris-Flowers-Using-Various-Machine-Learning-Models

A machine learning project has a number of well known steps:
* Define Problem.
* Prepare Data.
* Evaluate Algorithms.
* Improve Results.
* Present Results.

The best way to really come to terms with a new platform or tool is to work through a machine learning project end-to-end and cover the key steps. Namely, from loading data, summarizing data, evaluating algorithms and making some predictions.If you can do that, you have a template that you can use on dataset after dataset. You can fill in the gaps such as further data preparation and improving result tasks later, once you have more confidence.The best small project to start with on a new tool is the classification of iris flowers (e.g. the iris dataset).

Overview of what we are going to cover:
1. Installing the Python and SciPy platform.
2. Loading the dataset.
3. Summarizing the dataset.
4. Visualizing the dataset.
5. Evaluating some algorithms.
6. Making some predictions.

**1. Install Python and Libraries**
There are 5 key libraries that you will need to install. Below is a list of the Python SciPy libraries required for this tutorial:
• scipy
• numpy
• matplotlib
• pandas
• sklearn

**2. Load The Data**
The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.In this step we are going to load the iris data from CSV file URL.

**3. Summarize the Dataset**
* Dimensions of the dataset.
* Peek at the data itself.
* Statistical summary of all attributes.
* Breakdown of the data by the class variable.

**4. Data Visualization**
We are going to look at two types of plots:
* Univariate plots to better understand each attribute.
* Multivariate plots to better understand the relationships between attributes

**5. Evaluate Some Algorithms**
Now it is time to create some models of the data and estimate their accuracy on unseen data.
   **5.1 Create a train test Dataset**
     (We need to know that the model we created is good.)
     Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.
That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.
     We will split the loaded dataset into two, 80% of which we will use to train, evaluate and select among our models, and 20% that we will hold back as a test dataset.
    **5.2 Test Harness**
    We will use stratified 10-fold cross validation to estimate model accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. Stratified means that each fold or split of the dataset will aim to have the same distribution of example by class as exist in the whole training dataset.
   **5.3 Build Models**
    We don’t know which algorithms would be good on this problem or what configurations to use.We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.
    
Let’s test 6 different algorithms:
• Logistic Regression (LR)
• K-Nearest Neighbors (KNN).
• Classification and Regression Trees (CART).
• Gaussian Naive Bayes (NB).
• Support Vector Machines (SVM).
   **5.4 Select Best Model**
   We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.Running the example above, we get the following raw results:
                                            * LR: 0.960897 (0.052113)
                                            * KNN: 0.957191 (0.043263)
                                            * CART: 0.957191 (0.043263)
                                            * NB: 0.948858 (0.056322)
                                            * SVM: 0.983974 (0.032083)
   In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%.
We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (via 10 fold-cross validation).A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.
