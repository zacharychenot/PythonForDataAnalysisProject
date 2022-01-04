# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:19:06 2021

@author: HP
"""
# Import de toutes les librairies
from flask import Flask
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler # Import for standard scaling of the data
from sklearn.model_selection import train_test_split # Import train_test_split function

# Transformation du dataset en dataframe
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df_prep = df.copy()

# we remove the following columns because we only use height and weight
columns_to_remove = ['Age', 'Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
if set(columns_to_remove).issubset(df_prep):
    df_prep = df_prep.drop(columns_to_remove, axis=1)

X = df_prep.drop(columns=["NObeyesdad"])



# Target variable
y = df_prep['NObeyesdad'] 
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Creation of the model 
model = RandomForestClassifier(random_state=2020)
clf = model.fit(X_train, y_train)



app = Flask(__name__)
@app.route('/Height=/<int:Height>/Weight=/<int:Weight>')
def default(Height, Weight):
    
    data = [[Height, Weight]]
  
# Create the pandas DataFrame 
    df2 = pd.DataFrame(data, columns = ['Height', 'Weight'])
    y_pred = clf.predict(df2)
    predictionstr = str(y_pred) 
    predictionstr = predictionstr[2:-2]
    predictionstr = predictionstr.replace("_", " ")
    print(predictionstr)
   
    
    
    return "We predicted that your weight range is : "  + predictionstr


if __name__ == '__main__':
    app.run(host='localhost')
    
