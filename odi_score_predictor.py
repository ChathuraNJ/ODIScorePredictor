import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib as jb
import os
dataset='\odi.csv'
path=os.path.abspath(os.getcwd())+dataset
print(path)
def accuracy(y_pred,y_test,thresh):
	correct=0
	for i in range(len(y_pred)):
		if abs(y_test[i]-y_pred[i])<=thresh:
			correct+=1
	return (correct/len(y_pred))*100
def preprocess_test_data(list_dic,test_data):
	l=[]
	for i in range(len(list_dic)):
		l.append(list_dic[i][test_data[i]])
	#xx=X[:,0:5]
	#x=np.hstack((xx,np.array(l)))
	#x=sc.fit_transform(x)
	td=np.array([l+test_data[5:]])
	return sc.fit_transform(td)
df=pd.read_csv(path)
#print(df.head())
l=df.columns
#Batsman statistics
rdf1=df[['Batsman','Runs']]
df1=rdf1.groupby(['Batsman']).mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df1.index,df1.Runs)
ax.set(title = "Average Score of the Batsman",
       xlabel = "Player",
       ylabel = "Average Score")
plt.setp(ax.get_xticklabels(), rotation = 45)
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.show()
#Bowler Statistics
bdf1=df[['Bowler','Wickets']]
df2=bdf1.groupby(['Bowler']).mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df2.index,df2.Wickets)
ax.set(title = "Average Wickets of the Bowler",
       xlabel = "Player",
       ylabel = "Average Wickets")
plt.setp(ax.get_xticklabels(), rotation = 45)
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.show()

#print(df.isnull().any())
#Selecting the inputs as Venue,Batting Team,Bowling Team,Batsman at Strike,Bowler,Runs till that ball,Wickets up untill that ball,
#Overs,Runs for the last 5 Overs,Wickets for the last 5 Overs
X=df.iloc[:,2:12].values
#print(X)
#Output as Total
y=df.iloc[:,14:15].values
sc=StandardScaler()
le=LabelEncoder()
list_dic=[]
for i in range(5):
	X[:,i]=le.fit_transform(X[:,i])
	d=dict(zip(le.classes_, le.transform(le.classes_)))
	list_dic.append(d)
#print(list_dic)
X=sc.fit_transform(X)
#print(X,y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=123)
dc=DecisionTreeRegressor()
dc.fit(X_train,y_train)
filename='DTR_model.sav'
jb.dump(dc, filename)
loaded_model = jb.load(filename)
y_pred=loaded_model.predict(X_test)
#The R^2 score of the test data
print("The R^2Score of Decision Tree Regressor",r2_score(y_pred,y_test))
#The accuracy with +/- 5 runs variation
print("The Custom accuracy of Decision Tree Regressor",accuracy(y_pred,y_test,5))
#Prediction of Score for test data 
test_data=['The Rose Bowl','England','Pakistan','IR Bell','Shoaib Akhtar',11,1,2.4,11,1]
td=preprocess_test_data(list_dic,test_data)
print("The Score of the Innings for the given test data is",loaded_model.predict(td)[0])



	


