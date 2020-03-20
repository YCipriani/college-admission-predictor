import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

# predictor to determine whether candidates would get admitted to a prestigious university
# 2 possible outcomes: Admitted (1) or Rejected (0)
# 3 independent variables: GMAT score, GPA, and Years of work experience

# In this project, we are creating the dataframe from scratch
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

candidates_df = pd.DataFrame(candidates, columns=['gmat','gpa','work_experience','admitted'])
candidates_df.index.name = 'index'
#print(candidates_df)

X = candidates_df[['gmat','gpa','work_experience']]
y = candidates_df['admitted']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
predictions=logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
#plt.show()
#print(confusion_matrix)
print(X_test)
print(predictions)
print('Accuracy: ' + str(metrics.accuracy_score(y_test, predictions)*100) + '%\n')

# new set of data that we want to predict the admitted column for each new candidate
new_candidates = {'gmat': [590,740,680,610,710],
                  'gpa': [2,3.7,3.3,2.3,3],
                  'work_experience': [3,4,6,1,5]
                  }
df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])
y_pred=logistic_regression.predict(df2)
print(df2)
print(y_pred)
