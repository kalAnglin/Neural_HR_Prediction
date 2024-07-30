# neural-network-challenge-2

A data analysis assignment for Columbia University's AI bootcamp.

I worked with ChatGPt4o and Odele Pax to complete this assignment.

## Project Overview
As a business intelligence data analysts, I’ve been asked to create a neural network for HR to use to predict whether employees are likely to leave the company. HR also believes some employees may be better suited for other departments and is requested the model predict the department that best fits each employee.

## Methods
A neural network with 3 hidden layers was used before creating separate branches to predict the likelihood of an employee leaving the company and the department that best fits each employee. The solution/code is written in Python and uses the TensorFlow library to build and train the neural network model and along with OneHotEncoder & StandardScaler for data preprocessing The model is trained on the data read-in from an edx attrition dataset. 

For the dept_output_layer, the softmax activation function was chosen because it is appropriate for multi-class classification problems. The softmax function converts the raw output scores into probabilities that sum to 1, allowing the model to make a probabilistic prediction for each class, and the class with the highest probability is chosen as the predicted class.

For the atr_output_layer, the sigmoid activation function was chosen because it is appropriate for binary classification problems. The sigmoid function outputs a probability value between 0 and 1, which can be interpreted as the likelihood of the positive class. A threshold (usually 0.5) is then used to decide the class label.The model's accuracy is evaluated using the test data.

## Recommendations
Accuracy is not the best metric to use on this data given the imbalance identified by the value counts for Department and Attrition features. The model accuracy is measuring the proportion of correctly predicated instances out of total instances. A higher probability for one class may result in a high accuracy score by simply predicting the majority class, even though the model may not be correctly identifying the minority class.

The model score can be improved through the following ways:
Data Preprocessing - Adjusting the class weights or using resampling techniques to address the imbalance in the minority class. We could also creating or transforming features to provide a better way to capture the underlying patterns in the data.
Model Architecture - Changing the model architecture by adding more layers, increasing the number of neurons in the layers, or using a different activation function.
Hyperparameter Tuning - Changing the hyperparameters such as the learning rate, batch size, optimizer, loss function, and number of epochs.