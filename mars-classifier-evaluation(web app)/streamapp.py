# number crunching
import pandas as pd
import datetime as dt
import numpy as np
import io
# plotting
import matplotlib.pyplot as plt
import plotly.express as px
#import base64
from io import BytesIO
import sklearn

#import xlsxwriter
#import pyxlsb
#from pyxlsb import open_workbook as open_xlsb
# web app
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="  MARS Metric & MARS charts  ", page_icon="🧊", layout="wide", menu_items={
     })
st.markdown("<h1 style='text-align: center; color: black;'>Implementation of MARS metrics and MARS charts for evaluating classifier exclusivity</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    st.header("CSV Generator")
    st.write("If you have a raw training dataset, you can upload your dataset in the CSV Generator & select the classifiers you want to use. ")
    st.write("CSV Generator will run the selected classifiers on your dataset and generate the classifier output in the format required to get the MARS metric & MARS charts. You can download the classifers output and upload the file to get the MARS metric & chart.")
    ###############################################################################################
    #if user wants to use csv_generator

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
            if key == "CNN":
                models[key].fit(X_train_CNN,y_train,epochs=35,verbose=0,validation_data=(X_test_CNN,y_test))
            # Prediction
                predictions = models[key].predict(X_test_CNN)
                predictions = (predictions > 0.5).astype(np.int)
                predictions = [item for sublist in predictions.tolist() for item in sublist]
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

            compiled_df = compiled_df[['Instance ID','classifier','pred_label','true_label']]

        return compiled_df


    uploaded_file = st.file_uploader(
        label="          Upload Dataset Here          ",
        type="csv",
        accept_multiple_files=False,
        help='''Here you can inport your dataset which will run on .....classifiers and will generate the csv file in a format required to use mars charts.''')

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        y = dataset['class_label']
        dataset.drop('class_label', inplace=True, axis=1)
        X = dataset

        models_list = st.multiselect(
             'Select the classifiers you want to use for classification:',
             ['LogisticRegression', 'SVM', 'RandomForestClassifier','DecisionTreeClassifier', 'GaussianNaiveBayes'])


        models = {}
        for x in range(len(models_list)):
            if len(models_list)<3:
                st.write("Please select 3 or more classifiers")
            if models_list[x] == 'LogisticRegression':
                from sklearn.linear_model import LogisticRegression
                models['Log. Regression'] = LogisticRegression()

            if models_list[x] == 'SVM':
                from sklearn.svm import LinearSVC
                models['SVM'] = LinearSVC()

            if models_list[x] == 'RandomForestClassifier':
                from sklearn.ensemble import RandomForestClassifier
                models['Random Forest'] = RandomForestClassifier()

            if models_list[x] == 'DecisionTreeClassifier':
                from sklearn.tree import DecisionTreeClassifier
                models['Decision Tree'] = DecisionTreeClassifier()

            if models_list[x] == 'GaussianNaiveBayes':
                from sklearn.naive_bayes import GaussianNB
                models['Naive Bayes'] = GaussianNB()

            #if models_list[x] == 'CNN':
                #from tensorflow.keras.models import Sequential
                #from tensorflow.keras.layers import Conv1D
                #from tensorflow.keras.layers import MaxPool1D
                #from tensorflow.keras.layers import Flatten
                #from tensorflow.keras.layers import Dropout
                #from tensorflow.keras.layers import BatchNormalization
                #from tensorflow.keras.layers import Dense
                #from tensorflow.keras.optimizers import Adam

            # Sample CNN Structure: https://www.kaggle.com/krutarthhd/breast-cancer-detection-using-cnn-98-accuracy
                #models['CNN'] = Sequential()
                #models['CNN'].add(Conv1D(filters=16,kernel_size=2,activation='relu',input_shape=(X.shape[1],1)))
                #models['CNN'].add(BatchNormalization())
                #models['CNN'].add(Dropout(0.2))
                #models['CNN'].add(Conv1D(32,2,activation='relu'))
                #models['CNN'].add(BatchNormalization())
                #models['CNN'].add(Dropout(0.2))
                #models['CNN'].add(Flatten())
                #models['CNN'].add(Dense(32,activation='relu'))
                #models['CNN'].add(Dropout(0.2))
                #models['CNN'].add(Dense(1,activation='sigmoid'))
                #models['CNN'].compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
        #st.write(models)
        if st.button("Run"):
        #if st.button(‘Run’):
            compiled_df = csv_generator(models, dataset)
            st.write('Successful run!')
            st.write(compiled_df.head())

            st.write("  Download the classifier output file here. Same file can be used to get the MARS metric & MARS charts  ")
            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            Compiledcsv = convert_df(compiled_df)
            st.download_button(
                "Download Classifier_output.csv",
                Compiledcsv,
                "Classifier_output.csv",
                "text/csv",
                key='download-classifier_output'
                )
    #print('Successful run. Compiled_DF_MARSREADY.csv  is now saved to the current working directory')
    #compiled_df.to_csv('Compiled_DF_MARSREADY.csv', index = False)

###############################################################################################

with col2:
    #st.markdown("<h1 style='text-align: center; color: black;'>Implementation of MARS metrics and MARS charts for evaluating classifier exclusivity</h1>", unsafe_allow_html=True)
    #st.title("Implementation of MARS metrics and MARS charts for evaluating classifier exclusivity")
    st.header("Get MARS metric & MARS charts")
    st.write("If you already have your classifier outputs in the format displayed below. Upload your csv file to get the desired results.")
    st.info("Required Format: In column 1 of your csv provide Instance ID, in column 2 provide name of the classifier, in column 3 and column 4 provide predicted label & true label respectively as shown below.  ")

    sample = pd.read_csv('sample.csv')
    st.table(sample.iloc[0:4])
    st.write("  You can also download the sample file for reference.  ")
    @st.cache
    def convert_df(df):
       return df.to_csv().encode('utf-8')

    csv = convert_df(sample)
    st.download_button(
       "Download sample input data file",
       csv,
       "sample.csv",
       "text/csv",
       key='download-csv'
    )
    st.write("  You can find detailed user instructions below which shows how to interprete the results:  ")

    with open("documentation.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Full tutorial documentation is available here.",
                        data=PDFbyte,
                        file_name="documentation.pdf",
                        mime='application/octet-stream')
    #image = Image.open("D:\RA Project\MARS\.py files\streamlit-app/mars.png")
    #st.sidebar.image(image)
    #######################################################################################################################
    def shinethrough(df, k):
        total_classifiers = df['classifier'].nunique()
        k=(df.shape[0])/total_classifiers
        df=df.groupby('classifier').head(k)
        df = df.loc[(df['true_label'] == 1) & (df['pred_label'] == 1)] #To keep only TP entries and remove rest
        df.drop(columns = ['pred_label','true_label'])
        #calculate total number of total unique TP found by all classifiers
        total_unique_TP = len(pd.unique(df['Instance ID']))

        #No of unique classifiers
        un_class = df['classifier'].unique()
        un_class = sorted(un_class)
        data =[]
        column_names = ['x_axis', 'y_axis']

        #create pairs of classifiers to form a dataframe
        for i in range(len(un_class)):
            for j in range(len(un_class)):
                if i!=j:
                    new_ele = (un_class[i],un_class[j])
                    data.append(new_ele)

        #Dataframe df_all contains count of total unique True positives
        df_all = pd.DataFrame.from_records(
            data, columns=column_names)
        df_all['count'] = total_unique_TP

        ###############################################################################################################
        #df_unique_ID contains all the unique True positive reviews(ID)
        df_uni = pd.DataFrame.from_records(data, columns=column_names)
        names=df['classifier'].unique().tolist()
        df_unique_ID = pd.DataFrame()
        for name1 in names:
            df_1 = df.loc[df.classifier==name1]
            for name2 in names:
                if name1 != name2:
                    df_2 = df.loc[df.classifier==name2]
                    df_1 = df_1[~df_1['Instance ID'].isin(df_2['Instance ID'])]

            df_unique_ID=df_unique_ID.append(df_1)
        df_unique_ID = df_unique_ID.drop(columns = ['pred_label','true_label'])

        #df_uni contains count of unique ID's of classifier on y axis
        for name in names:
            df_1 = df_unique_ID.loc[df_unique_ID.classifier==name]
            df_uni.loc[df_uni['y_axis'] == name, 'count'] = len(df_1.index)

        ###############################################################################################################
        #create df_twin containing total unique TP found by classifier on x-axis and y-axis
        df_twin = pd.DataFrame.from_records(data, columns=column_names)
        names=df['classifier'].unique().tolist()
        for name1 in names:
            df_1 = df.loc[df.classifier==name1]
            for name2 in names:
                if name1 != name2:
                    df_2 = df.loc[df.classifier==name2]
                    df_two = df_1.append(df_2)
                    for name in names:
                        if name !=name1 and name !=name2:
                            df_3 = df.loc[df.classifier==name]
                            df_two = df_two[~df_two['Instance ID'].isin(df_3['Instance ID'])]
                            n = len(pd.unique(df_two['Instance ID']))
                            df_twin.loc[(df_twin['x_axis'] == name1) & (df_twin['y_axis'] == name2), 'common'] = n
        df_twin = df_twin.rename(columns={'common': 'count'})

        #shineThrough scores calculations
        def trim_fraction(text):
            text = str(text)
            sep='.'
            stripped = text.split(sep, 1)[0]
            return stripped

        df_shine_scores =df_uni.drop(columns = ['count'])
        df_shine_scores['shine_score1'] = df_uni['count'].apply(trim_fraction)
        df_shine_scores['shine_score2'] = total_unique_TP
        df_shine_scores['shine_score2'] = df_shine_scores['shine_score2'].apply(trim_fraction)
        df_shine_scores['shine_score3'] = df_twin['count'].apply(trim_fraction)
        df_shine_scores['shine_score4'] = total_unique_TP
        df_shine_scores['shine_score4'] = df_shine_scores['shine_score4'].apply(trim_fraction)

        df_shine_scores['shine_score'] = df_shine_scores['shine_score1'].astype(str)+'/'+df_shine_scores['shine_score2'].astype(str) +' | '+df_shine_scores['shine_score3'].astype(str)+'/'+df_shine_scores['shine_score4'].astype(str)
        df_shine_scores['shine_score'] = df_shine_scores['shine_score'].fillna(0)
        st.write("ShineThrough Scores(Count of Exclusive True Positives)")
        df_shine_scores = df_shine_scores.pivot(index="y_axis", columns="x_axis", values="shine_score").fillna(value = '-')
        df_shine_scores= df_shine_scores.style.set_properties(**{'text-align': 'center'})
        df_shine_scores = df_shine_scores.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        st.write(df_shine_scores)
        #print("\n")
        #print("\n")
        #shineThrough metric calculations
        def trim_fraction_(text):
            text = str(text)
            sep='.00'
            stripped = text.split(sep, 1)[0]
            return stripped

        df_shine_metric = df_uni.drop(columns = ['count'])
        df_shine_metric['shine_score1'] = df_uni['count']/total_unique_TP
        df_shine_metric['shine_score2'] = df_twin['count']/total_unique_TP
        df_shine_metric = df_shine_metric.round(2)
        df_shine_metric['shine_score1'] = df_shine_metric['shine_score1'].apply(trim_fraction_)
        df_shine_metric['shine_score2'] = df_shine_metric['shine_score2'].apply(trim_fraction_)
        df_shine_metric['shine_score'] = df_shine_metric['shine_score1'].astype(str)+' | '+df_shine_metric['shine_score2'].astype(str)
        st.write("ShineThrough Metric")
        df_shine_metric= df_shine_metric.pivot(index="y_axis", columns="x_axis", values="shine_score").fillna(value = '-')
        df_shine_metric= df_shine_metric.style.set_properties(**{'text-align': 'center'})
        df_shine_metric = df_shine_metric.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        st.write(df_shine_metric)

        #Plotting bubble chart
        #Creating blank dataframe for plotting chart in order
        data1=[]
        for i in range(len(un_class)):
            for j in range(len(un_class)-1, -1, -1):
                new_ele = (un_class[i],un_class[j])
                data1.append(new_ele)
                    #data.append(new_ele)

        df_blank = pd.DataFrame.from_records(
            data1, columns=column_names)
        df_blank['count'] =0

        df_twin = df_twin.sort_values(by =['count'],ascending=False)
        df_twin = df_blank.append(df_twin)
        #
        df_twin['size']=df_twin['count']*df_twin['count']
        df_twin['Color Significance']='Exclusive TP found by combined classifiers on x axis & y axis'
        df_uni['size']=df_uni['count']*df_uni['count']
        df_uni['Color Significance']='Exclusive TP found by individual classifier on y axis'
        df_final = df_twin.append(df_uni)

        fig=px.scatter(df_final, x="x_axis", y="y_axis",
                   color_discrete_sequence=px.colors.qualitative.Alphabet,
                   color = 'Color Significance',
                   size='size',opacity=0.6,size_max=70,
                   color_discrete_map={"Exclusive TP found by combined classifiers on x axis & y axis": 'orange',"Exclusive TP found by individual classifier on y axis":'yellow'},hover_data=['Color Significance'],
                  )

        fig.update_layout(
            width=800,
            height=600
        )
        fig.update_layout(
            xaxis={'side': 'top'},
            yaxis={'side': 'left'}
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1
        ))
        fig.update_layout(title_text='MARS ShineThrough Chart', title_x=0.05)
        #fig.show()
        st.plotly_chart(fig)
        return df_shine_metric


    #######################################################################################################################################

    def occlusion(df, k):
        df=df.groupby('classifier').head(k)
        df_tp = df.loc[(df['true_label'] == 1) & (df['pred_label'] == 1)]
        df_fn = df.loc[(df['true_label'] == 1) & (df['pred_label'] == 0)]

        #calculate total number of total unique TP found by all classifiers
        total_unique_TP = len(pd.unique(df_tp['Instance ID']))

        un_class = df['classifier'].unique()
        un_class = sorted(un_class)
        data =[]
        column_names = ['x_axis', 'y_axis']
        #create pairs of classifiers
        for i in range(len(un_class)):
            for j in range(len(un_class)):
                if i!=j:
                    new_ele = (un_class[i],un_class[j])
                    data.append(new_ele)

        #df_all contains total number of unique True positives
        df_all = pd.DataFrame.from_records(data, columns=column_names)
        df_all['count'] = total_unique_TP

        df_fn_twin = pd.DataFrame.from_records(data, columns=column_names)

        df_fn_unique1 = pd.DataFrame.from_records(data, columns=column_names)



        names=df['classifier'].unique().tolist()
        df_fn_unique = pd.DataFrame()
        for name1 in names:
            df_1 = df_fn.loc[df_fn.classifier==name1]
            df_1 = df_1[df_1['Instance ID'].isin(df_tp['Instance ID'])]
            df_fn_unique=df_fn_unique.append(df_1)

        ###############################################################################################################
        #Unique FN for that classifier relative to the TP of all the classifiers
        df_fn_unique = df_fn_unique.drop(columns = ['pred_label','true_label'])

        for name in names:
            df_1 = df_fn_unique.loc[df_fn_unique.classifier==name]
            df_fn_unique1.loc[df_fn_unique1['y_axis'] == name, 'count'] = len(df_1.index)

        for name1 in names:
            df_1 = df_fn.loc[df_fn.classifier==name1]
            df_1_tp = df_tp.loc[df_tp.classifier==name1]
            for name2 in names:
                if name1 != name2:
                    df_2 = df_fn.loc[df_fn.classifier==name2]
                    df_2_tp = df_tp.loc[df_tp.classifier==name2]
                    df_1 = df_1[~df_1['Instance ID'].isin(df_2_tp['Instance ID'])]
                    df_2 = df_2[~df_2['Instance ID'].isin(df_1_tp['Instance ID'])]

                    df_2 = df_2.append(df_1)
                    df_2 = df_2.drop_duplicates(subset='Instance ID', keep="last")
                    df_temp = pd.DataFrame()
                    for name3 in names:
                        if name3 != name1 and name3 != name2:
                            df_3 = df_tp.loc[df_tp.classifier==name3]
                            df_temp = df_temp.append(df_2[df_2['Instance ID'].isin(df_3['Instance ID'])])
                            df_temp = df_temp.drop_duplicates(subset='Instance ID', keep="last")
                    df_fn_twin.loc[(df_fn_twin['y_axis'] == name1) & (df_fn_twin['x_axis'] == name2), 'common'] = len(df_temp.index)

        df_fn_twin = df_fn_twin.rename(columns={'common': 'count'})

        ###############################################################################################################
        #created blank dataframe to arrange classifiers in order for plot
        data1=[]

        for i in range(len(un_class)):
            for j in range(len(un_class)-1, -1, -1):
                new_ele = (un_class[i],un_class[j])
                data1.append(new_ele)

        df_blank = pd.DataFrame.from_records(data1, columns=column_names)
        df_blank['count'] =0

        ###############################################################################################################
        #Occlusion scores calculations
        def trim_fraction(text):
            text = str(text)
            sep='.'
            stripped = text.split(sep, 1)[0]
            return stripped
        df_occ_scores =df_fn_unique1.drop(columns = ['count'])
        df_occ_scores['occ_score1'] = df_fn_unique1['count']
        df_occ_scores['occ_score2'] = total_unique_TP
        df_occ_scores['occ_score3'] = df_fn_twin['count']
        df_occ_scores['occ_score4'] = total_unique_TP

        df_occ_scores['occ_score1'] = df_occ_scores['occ_score1'].apply(trim_fraction)
        df_occ_scores['occ_score2'] = df_occ_scores['occ_score2'].apply(trim_fraction)
        df_occ_scores['occ_score3'] = df_occ_scores['occ_score3'].apply(trim_fraction)
        df_occ_scores['occ_score4'] = df_occ_scores['occ_score4'].apply(trim_fraction)

        df_occ_scores['occ_score'] = df_occ_scores['occ_score1'].astype(str)+'/'+df_occ_scores['occ_score2'].astype(str) +' | '+df_occ_scores['occ_score3'].astype(str)+'/'+df_occ_scores['occ_score4'].astype(str)
        df_occ_scores['occ_score'] = df_occ_scores['occ_score'].fillna(0)
        st.write("Occlusion Scores(Count of Exclusive False Negatives)")
        df_occ_scores = df_occ_scores.pivot(index="y_axis", columns="x_axis", values="occ_score").fillna(value = '-')
        df_occ_scores= df_occ_scores.style.set_properties(**{'text-align': 'center'})
        df_occ_scores = df_occ_scores.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        st.write(df_occ_scores)
        st.write("\n")
        ###############################################################################################################
        #Occlusion metric calculations
        def trim_fraction_(text):
            text = str(text)
            sep='.00'
            stripped = text.split(sep, 1)[0]
            return stripped
        df_occ_metric =df_fn_unique1.drop(columns = ['count'])
        df_occ_metric['occ_score1'] = df_fn_unique1['count']/total_unique_TP
        df_occ_metric['occ_score2'] = df_fn_twin['count']/total_unique_TP
        df_occ_metric = df_occ_metric.round(2)

        df_occ_metric['occ_score1'] = df_occ_metric['occ_score1'].apply(trim_fraction_)
        df_occ_metric['occ_score2'] = df_occ_metric['occ_score2'].apply(trim_fraction_)
        df_occ_metric['occ_score'] = df_occ_metric['occ_score1'].astype(str)+' | '+df_occ_metric['occ_score2'].astype(str)


        df_occ_metric =df_occ_metric.pivot(index="y_axis", columns="x_axis", values="occ_score").fillna(value = '-')
        st.write("Occlusion Metric")
        df_occ_metric= df_occ_metric.style.set_properties(**{'text-align': 'center'})
        df_occ_metric = df_occ_metric.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        st.write(df_occ_metric)
        st.write("\n")

        ###############################################################################################################

        df_fn_unique1 = df_fn_unique1.sort_values(by =['count'],ascending=False)
        df_fn_unique1 = df_blank.append(df_fn_unique1)
        df_fn_unique1['size']=df_fn_unique1['count']*df_fn_unique1['count']
        df_fn_unique1['Color Significance']='Exclusive FN missed by individual classifier on y axis'


        df_fn_twin['size']=df_fn_twin['count']*df_fn_twin['count']
        df_fn_twin['Color Significance']='Exclusive FN missed by combined classifiers on x axis & y axis'

        df_final = df_fn_unique1.append(df_fn_twin)

        fig=px.scatter(df_final, x="x_axis", y="y_axis",
                   color_discrete_sequence=px.colors.qualitative.Alphabet,
                   color = 'Color Significance',
                   size='size',
                   opacity=0.6,
                   size_max=70,
                   color_discrete_map={"Exclusive FN missed by combined classifiers on x axis & y axis": 'red',"Exclusive FN missed by individual classifier on y axis":'orange'},

                  )
        fig.update_traces(marker=dict(
                                          line=dict(width=0.2,
                                                    color='DarkSlateGrey')),
                             selector=dict(mode='markers'))
        fig.update_layout(
            width=800,
            height=600
        )
        fig.update_layout(
            xaxis={'side': 'top'},
            yaxis={'side': 'left'}
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1
        ))
        fig.update_layout(title_text='MARS Occlusion Chart', title_x=0.05)

        st.plotly_chart(fig)

        return df_occ_metric
    ######################################################################################################################################
    # %%

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden; }
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    # enable the end user to upload a csv file:
    st.write("_" * 10)
    # >>>>>>>>>>>>>>>>
    st.write("          Upload CSV File Here          ")


    uploaded_file = st.file_uploader(
        label="Upload the csv file containing the classification output for which you want Shinethrough & Occlusion charts",
        type="csv",
        accept_multiple_files=False,
        help='''Please upload a csv file in the format mentioned to get the desired results.''')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        total_classifiers = df['classifier'].nunique()
        k=(df.shape[0])/total_classifiers

        status = st.radio('Select the metric from below: ',
                      ('ShineThrough Metric', 'Occlusion Metric', 'Both'))

        #def to_excel(df):
            #output = BytesIO()
            #writer = pd.ExcelWriter(output, engine='xlsxwriter')
            #df.to_excel(writer, index=False, sheet_name='Sheet1')
            #workbook = writer.book
            #worksheet = writer.sheets['Sheet1']
            #format1 = workbook.add_format({'num_format': '0.00'})
            #worksheet.set_column('A:A', None, format1)
            #writer.save()
            #processed_data = output.getvalue()
            #return processed_data
        # compare status value
        if(status == 'ShineThrough Metric'):
            st.success("ShineThrough Metric & ShineThrough Chart")
            df = shinethrough(df, k)
            #df_xlsx = to_excel(df)
            #st.download_button(label='📥 Download Current Result',
                                    #data=df_xlsx ,
                                    #file_name= 'shinethrough_metric.xlsx')

        elif(status == 'Occlusion Metric'):
            st.success("Occlusion Metric & Occlusion Chart")
            df = occlusion(df, k)
            #df_xlsx = to_excel(df)
            #st.download_button(label='📥 Download Current Result',
                                    #data=df_xlsx ,
                                    #file_name= 'occlusion_metric.xlsx')
        else:
            st.success("ShineThrough & Occlusion Metric & Chart")
            df1 = shinethrough(df, k)
            #df_xlsx1 = to_excel(df1)
            #st.download_button(label='📥 Download SineThrough Result',
                                    #data=df_xlsx1 ,
                                    #file_name= 'shinethrough_metric.xlsx')

            df2 = occlusion(df, k)
            #df_xlsx2 = to_excel(df2)
            #st.download_button(label='📥 Download Occlusion Result',
                                    #data=df_xlsx2 ,
                                    #file_name= 'occlusion_metric.xlsx')

    st.write("_" * 10)

st.write("_" * 10)
#total_classifiers = df['classifier'].nunique()
#k=(df.shape[0])/total_classifiers
st.write("  For any queries/feedback, write to us at mars_classifier_evaluation@vt.edu.  ")
