#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer #Dataset
import pandas as pd


# In[54]:


#Load Dataset
dataset = pd.read_csv("data.csv")
y = dataset['class_label']
dataset.drop('class_label', inplace=True, axis=1)
X = dataset


# In[87]:


#Load User Specified Model List
with open('models.txt') as f:
    models = f.read()
models_list = models.split(',')
models_list = [x.strip(' ').lower() for x in models_list]
print(models_list)


# In[82]:


#Creating Models Dictionary
models = {}

for x in range(len(models_list)):
    if models_list[x] == 'lr':
        from sklearn.linear_model import LogisticRegression
        models['Log. Regression'] = LogisticRegression()

    if models_list[x] == 'svm':
        from sklearn.svm import LinearSVC
        models['SVM'] = LinearSVC()

    if models_list[x] == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        models['Random Forest'] = RandomForestClassifier()

    if models_list[x] == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        models['Decision Tree'] = DecisionTreeClassifier()

    if models_list[x] == 'nb':
        from sklearn.naive_bayes import GaussianNB
        models['Naive Bayes'] = GaussianNB()

    if models_list[x] == 'cnn':
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D
        from tensorflow.keras.layers import MaxPool1D
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

    # Sample CNN Structure: https://www.kaggle.com/krutarthhd/breast-cancer-detection-using-cnn-98-accuracy
        models['CNN'] = Sequential()
        models['CNN'].add(Conv1D(filters=16,kernel_size=2,activation='relu',input_shape=(X.shape[1],1)))
        models['CNN'].add(BatchNormalization())
        models['CNN'].add(Dropout(0.2))
        models['CNN'].add(Conv1D(32,2,activation='relu'))
        models['CNN'].add(BatchNormalization())
        models['CNN'].add(Dropout(0.2))
        models['CNN'].add(Flatten())
        models['CNN'].add(Dense(32,activation='relu'))
        models['CNN'].add(Dropout(0.2))
        models['CNN'].add(Dense(1,activation='sigmoid'))
        models['CNN'].compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])



def csv_generator(model_dict, dataset):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy, precision, recall = {}, {}, {}

    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)

    X_train_CNN = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1)) #Reshape Data to structure required by CNN.
    X_test_CNN = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1)) #Reshape Data to structure required by CNN.

    final_df = pd.DataFrame()
    compiled_df = pd.DataFrame(columns = ['pred_label','true_label','classifier', 'Instance ID']) #MARS-ready dataframe

    for key in models.keys():
        if key == "CNN" or key == 'customCNN1' or key == 'customCNN2' or key == 'customCNN3' or key == 'customCNN4' or key == 'customCNN5':
        # Fit the classifier model
            models[key].fit(X_train_CNN,y_train,epochs=35,verbose=0,validation_data=(X_test_CNN,y_test))
        # Prediction
            predictions = models[key].predict(X_test_CNN)
            predictions = (predictions > 0.5).astype(np.int)
            predictions = [item for sublist in predictions.tolist() for item in sublist]

            model_json = models[key].to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            models[key].save_weights("model.h5")
            print("Saved model to disk")

        elif key == 'customNN1' or key == 'customNN2' or key == 'customNN3' or key == 'customNN4' or key == 'customNN5':
        # Fit the classifier model
            models[key].fit(X_train,y_train,epochs=35,verbose=0,validation_data=(X_test,y_test))
        # Prediction
            predictions = models[key].predict(X_test)
            predictions = (predictions > 0.5).astype(np.int)
            predictions = [item for sublist in predictions.tolist() for item in sublist]

            #model_json = models[key].to_json()
            #with open("model.json", "w") as json_file:
                #json_file.write(model_json)
            # serialize weights to HDF5
            #models[key].save_weights("model.h5")
            #print("Saved model to disk")
        else:
        # Fit the classifier model
            models[key].fit(X_train, y_train)
        # Prediction
            predictions = models[key].predict(X_test)

        compiled_df_temp = pd.DataFrame()
        compiled_df_temp['pred_label'] = predictions
        compiled_df_temp['true_label'] = y_test.tolist()
        compiled_df_temp['classifier'] = key
        compiled_df_temp['Instance ID'] = compiled_df_temp.index
        compiled_df = compiled_df.append(compiled_df_temp)

        final_df['y_test'] = y_test.tolist()
        final_df['predictions'] = predictions
        predictions = pd.DataFrame(predictions, columns=['pred_labels'])

    # Calculate Accuracy, Precision and Recall Metrics
        accuracy[key] = accuracy_score(predictions, y_test)
        precision[key] = precision_score(predictions, y_test)
        recall[key] = recall_score(predictions, y_test)

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    compiled_df = compiled_df[['Instance ID','classifier','pred_label','true_label']]

    return df_model, compiled_df


# In[84]:


def main():
    df_model, compiled_df = csv_generator(models, dataset)
    print(df_model)
    print('Successful run. Compiled_DF_MARSREADY.csv  is now saved to the current working directory')
    compiled_df.to_csv('Compiled_DF_MARSREADY.csv', index = False) # .CSV file ready for use saved to local path.
main()


# In[ ]:
