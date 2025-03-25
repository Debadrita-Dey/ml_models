import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
sonar_data = pd.read_csv('C:\\Users\\DELL\\Desktop\\ml\\sonar data.csv',header=None)
print(sonar_data.head())
print(sonar_data.info())
print(sonar_data.shape)
print(sonar_data[60].value_counts())
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
#standardize features
scaler=StandardScaler()
x_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model1 = LogisticRegression(C=0.1)
model1.fit(x_train, Y_train)
y_predict=model1.predict(X_test)
test_data_accuracy = accuracy_score( Y_test,y_predict)
print("Accuracy on test data model1", test_data_accuracy*100)
#%%

from sklearn.preprocessing import MinMaxScaler
scaler2=MinMaxScaler()
X__train=scaler.fit_transform(X_train)
X__test=scaler.transform(X_test)
model2 = LogisticRegression(C=0.01)
model2.fit(X_train, Y_train)
y_predict_=model2.predict(X_test)
accuracy = accuracy_score( Y_test,y_predict_)
print("Accuracy of model 2",accuracy*100)
#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline 
model3=Pipeline([("Poly_features",PolynomialFeatures(degree=2)),
                 ("scaler",StandardScaler()),
                 ("log reg",LogisticRegression(C=1))])
model3.fit(X_train,Y_train)
y_predictt=model3.predict(X_test)
accuracyy=accuracy_score(Y_test,y_predictt)
print("Accuracy  of model 3", accuracyy*100)
#%%
from sklearn.model_selection import cross_val_score
cross_val_accuracy = cross_val_score(model3, X, Y, cv=5, scoring='accuracy')  
print("Cross-validation accuracy of model 3 :", cross_val_accuracy.mean() * 100)
#%%
#SVM
from sklearn.svm import SVC
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
y_train_scaled=scaler.transform(X_test)

model4= SVC(kernel='rbf',C=1,gamma="scale")
model4.fit(x_train_scaled,Y_train)
y_pred=model4.predict(X_test)
accuracy_svc=accuracy_score(Y_test,y_pred)
print("accuracy of svc",accuracy_svc*100)
#%%
from sklearn.model_selection import cross_val_score
cross_val_accuracyy = cross_val_score(model4, X_train, Y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy of model 4:",cross_val_accuracyy.mean()*100)
#%%
# working with model4 svc
import numpy as np


input_data = (0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)

input_data = np.asarray(input_data).reshape(1, -1)


input_data_scaled = scaler.transform(input_data)


prediction_model1 = model1.predict(input_data_scaled)
print('The object is a Rock' if prediction_model1[0] == 'R' else 'The object is a Mine (Logistic Regression)')

prediction_model4 = model4.predict(input_data_scaled)
print('The object is a Rock' if prediction_model4[0] == 'R' else 'The object is a Mine (SVM)')


#%%
from sklearn.metrics import confusion_matrix

# Confusion matrix for Logistic Regression
y_pred_model1 = model1.predict(X_test)
cm_model1 = confusion_matrix(Y_test, y_pred_model1)
print("Confusion Matrix for Logistic Regression:\n", cm_model1)

# Confusion matrix for SVM
y_pred_model4 = model4.predict(X_test)
cm_model4 = confusion_matrix(Y_test, y_pred_model4)
print("Confusion Matrix for SVM:\n", cm_model4)
#%%
from	sklearn.ensemble	import	AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
ada_clf=AdaBoostClassifier(RandomForestClassifier(random_state=42))
ada_clf.fit(X_train,Y_train)
y__=ada_clf.predict(X_test)
accuracy_ada=accuracy_score(Y_test,y__)
print("accuracy using ada boost",accuracy_ada*100)
#%%
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
voting_clf=VotingClassifier(estimators=[
 ('lr',	LogisticRegression()),
 ('rf',	RandomForestClassifier(random_state=42)),
 ('svc',DecisionTreeClassifier(max_depth=10))])
voting_clf.fit(X_train,Y_train)
y_predicted_val_vot=voting_clf.predict(X_test)
r=accuracy_score(Y_test,y_predicted_val_vot)
print('accuracy of voting regressor:',r*100)
#%%
from	sklearn.ensemble	import	StackingClassifier
stacking_clf=StackingClassifier(
 estimators=[
 ('lr',	LogisticRegression(random_state=42)),
 ('rf',	RandomForestClassifier(random_state=42)),
 ('svc',SVC(probability=True,	random_state=42))
 ],
final_estimator=RandomForestClassifier(random_state=43),
 cv=5)
stacking_clf.fit(x_train_scaled,Y_train)
y_predicted_val__=stacking_clf.predict(X_test)
r1=accuracy_score(Y_test,y_predicted_val__)
print('accuracy of stacking regressor:',r1*100)
# %%
