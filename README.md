# Stock Prediction using Multitask learning
A two-phase setup that uses fastdtw to bucket correlated stocks and a multitask learning framework that tries to simultaneously predict future values for correlated stocks.

## Project Abstract: 
Financial forecasting plays a crucial role in investment decision-making and portfolio diversification. A given stock’s future value can be predicted based on its history on the assumption that there will be repetition of trends. However, stocks are not only affected by their history but also are affected by market trends and competitors. 

In this light, we build a multitask learning model that simultaneously tries to predict the future values of stocks whilst sharing domain specific knowledge among them. However, consideration of all stocks at once is impractical and introduces noise in the network. So we build a stock selector that finds the dynamic correlation between the stocks and selects the most correlated stocks as input to our multitask learning model.

It was observed that the multitask learning model had an improvement of around 3 - 4 % of relative accuracy when compared to the baseline models for each of the corresponding correlated stocks in similar settings.

Publication: https://link.springer.com/chapter/10.1007/978-981-15-9774-9_38

# Code Execution: 
## Dataset Analysis:
  * \Dataset Analysis
    Loads stock histories of various companies and provides two most correlated stocks
* Requirement:
    * Python3.6+
    * Pandas
    * Numpy
    * Matplotlib
    * Fastdtw   (installed using “pip install fastdtw”)
* Execution:
   Python Dataset_analysis.py
   Or view open in Jupyter notebook Dataset_analysis.ipynb



## Accuracy prediction for given Date:
* \StockPredictor : Predicts the accuracy given company and date
  * Requirements:
  * Python3.6+
  * Tensorflow 1.x
  * Numpy
  * Pandas
  * Sklearn
* Execution:
  * python api.py or open with jupyter notebook apy.ipynb


## Model Training:
\ModelTrain
Retrain the model and testing accuracies for baseline and multitask models.
WARNING: Do not run using normal tensorflow(CPU) as the model is a sequential model and requires GPU power. Running using CPU only can cause substantial amounts of heating.
* Requirements :
  * Python3.6+
  * CudaNN
  * Tensorflow 1.x GPU set up
  * Numpy
  * Pandas
  * Sklearn
* Execution:
	* Baseline models:
    * \ ModelTrain\ BaselineModels
    * python gmBaseline.py
    * python toyotaBaseline.py
    * Or view using Jupyter notebook
    * gmBaseline.ipynb
    * toyotaBaseline.py
* Multitask model:
  * \ ModelTrain\ MultitaskLearningModel
  * python mlt.py
  * Or view using Jupyter notebook mlt.ipynb

# UI:
  ## \ UI
Accessing the project functionalities through the UI requires setting up the UI project on Pycharm as a separate project.
To keep dependencies trackable, it is recommended to create a separate environment with following dependencies:
1) flask
2) wtforms
3) flask-bootstrap
4) fastdtw
5) sklearn

Note: After setting up the environment, the paths of the dataset and static folder should be noted and changes must be made in the code accordingly.

To run the code, "flask run" command is executed. This initiates the server and runs the web application on the localhost.

To access the web application, navigate to "localhost:5000/" on the browser.

Note: Mozilla Firefox is the suggested browser. Google chrome might mess up the graph plotting functionality.  

General motors and Toyota output comparison:

![image](https://user-images.githubusercontent.com/31768980/146443412-c4d952bc-0a2f-4b0d-a3df-db0d714a1f1c.png)

![image](https://user-images.githubusercontent.com/31768980/146443707-88968df5-87f1-489a-9ad2-4e70ba4d08ff.png)


