
# Titanic Survival Prediction
## Project Description
This machine learning project utilizes a supervised binary classification algorithm to predict whether a passenger would survive the Titanic accident. The model has been implemented in Python using the Scikit-Learn library, employing the Logistic Regression algorithm for predictions.

### Input Features
Gender, Adult Status, and Alone Status
Gender: Choose between 'Male' or 'Female'.
Adult Status: Indicate whether you are an adult or not.
Alone Status: Specify if you are traveling alone or not.
Additional Numeric Features
Age: Adjust the slider to indicate your age.
Number of Siblings & Spouse Aboard (Sibsp): Use the slider to specify the number.
Number of Parent & Children Aboard (Parch): Adjust the slider accordingly.
Ticket Fare: Set the slider for the ticket fare.
Social Class, Who You Are, and Embark Town
Social Class: Select your social class from 'First,' 'Second,' or 'Third.'
Who You Are: Specify if you are a 'Man,' 'Woman,' or 'Child.'
Embark Town: Choose the embarkation town from 'Southampton,' 'Cherbourg,' or 'Queenstown.'
### Model Prediction
Click the 'Predict' button to get the model's survival prediction based on the input features. The result will display your chance of survival as a percentage.

### Variable Dictionary
sex: Gender of a passenger (string).
age: Age of a passenger (float).
sibsb: Number of siblings and spouses aboard (integer).
parch: Number of parents and children aboard (integer).
fare: Ticket price for a passenger (float).
class: Social class of a passenger (string).
who: Indication if a passenger is a man, woman, or child (string).
adult_male: Boolean indicating if a male passenger is an adult or not.
embark_town: Embarkation port (string).
alone: Boolean indicating if a passenger was alone or not.
