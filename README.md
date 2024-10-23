# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values.
3. Classify the training data and the test data. 
4. Calculate the accuracy score, confusion matrix and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohanachandran.J.B
RegisterNumber:  212221080049
*/
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![Screenshot 2024-10-09 085525](https://github.com/user-attachments/assets/2c74cc06-f936-4f2c-b2e5-950da5778245)
![Screenshot 2024-10-09 085615](https://github.com/user-attachments/assets/c0a4163e-32ca-468b-8995-be16f0774ecb)
![Screenshot 2024-10-09 085640](https://github.com/user-attachments/assets/43a4b9ed-7f75-4af4-8183-394eef9635f9)

![Screenshot 2024-10-09 085654](https://github.com/user-attachments/assets/7451af6e-5b23-47cf-85ce-c72133413519)
![Screenshot 2024-10-09 085710](https://github.com/user-attachments/assets/f9cb8511-f24d-4dd4-aa72-6af9395dd207)
![Screenshot 2024-10-09 085726](https://github.com/user-attachments/assets/c00779e5-364a-4b59-beb9-af7dfbedd4ef)
![Screenshot 2024-10-09 085746](https://github.com/user-attachments/assets/11bfd70b-36f2-4f95-831d-13bb5bd1b192)
![Screenshot 2024-10-09 085829](https://github.com/user-attachments/assets/6cf13b97-ea87-46e0-9f1e-715031cbf0e9)
![Screenshot 2024-10-09 085907](https://github.com/user-attachments/assets/19b2d19c-b4ca-4eaa-bce7-f13d25a2a8b9)
![Screenshot 2024-10-09 085950](https://github.com/user-attachments/assets/d6c0ae5f-77be-4bf1-b6ba-0e173ca2facc)
![Screenshot 2024-10-09 090052](https://github.com/user-attachments/assets/3fd15e87-b394-44d2-96ca-1bc8a7ded678)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
