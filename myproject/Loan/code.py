import pandas as pd
data = pd.read_csv(r'C:\Users\MOHANKUMAR\Downloads\Loan\Loan\myproject\Loan\train_data.csv')
columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.drop('Loan_ID',axis=1)

data = data.dropna(subset=columns)
data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')

X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
RandomForestClassifier()
rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }

rs_rf=RandomizedSearchCV(RandomForestClassifier(),param_distributions=rf_grid,cv=5,n_iter=20,verbose=True)

rs_rf.fit(X,y)
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

rf = RandomForestClassifier(n_estimators=270,min_samples_split=5,min_samples_leaf=5,max_depth=5)

rf.fit(X,y)

import joblib
joblib.dump(rf,'loan_status_predict')
model = joblib.load('loan_status_predict')
import pandas as pd


def make_predictions(gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area):
    # Create a dictionary from the provided values
    data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    }
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Preprocess the input data if needed
    # ...
    
    # Make predictions using the model
    predictions = model.predict(df)
    #a = int(predictions)
    b = np.array(predictions)

    print("".join(map(str, b)))
    return b


    
    # Return the predictions
    

 