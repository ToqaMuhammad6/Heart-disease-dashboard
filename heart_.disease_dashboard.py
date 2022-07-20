

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
#from feature_selector import FeatureSelector

df = pd.read_csv('heart.csv')
df.rename(columns={'Sex': 'Gender'}, inplace=True)
cat=np.array(df.select_dtypes('object').columns)
nums=df.select_dtypes('number')
nums=(nums-nums.mean())/nums.std()
df['Gender'].replace({'F':0,'M':1},inplace=True)
df['ChestPainType'].replace({'ASY':1,'ATA':2,'NAP':3,'TA':4},inplace=True)
df['RestingECG'].replace({'LVH':1,'Normal':2,'ST':3},inplace=True)
df['ExerciseAngina'].replace({'N':0,'Y':1},inplace=True)
df['ST_Slope'].replace({'Down':1,'Flat':2, 'Up':3},inplace=True)
TARGET = 'HeartDisease'
y = df[TARGET]
X = df.drop(columns=TARGET)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
test = pd.concat([X_test, y_test], axis=1)
#import plotly_express as px


# Add predicted probabilities
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
col_sorted_by_importance=rf.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':X.columns[col_sorted_by_importance],
    'imps':rf.feature_importances_[col_sorted_by_importance]
})
names_array=['accuracy','f1 Score','Recall Score','Precision Score']
y_predict_test=rf.predict(X_test)
acc_test=accuracy_score(y_test,y_predict_test)
f1_test=f1_score(y_test, y_predict_test)
Recall_test=recall_score(y_test, y_predict_test)
pre_test=precision_score(y_test, y_predict_test)
Test_performance=[acc_test,f1_test,Recall_test,pre_test]
Test_performance = pd.DataFrame(Test_performance,index=names_array) 

y_predict_train=rf.predict(X_train)
acc_train=accuracy_score(y_train,y_predict_train)
f1_train=f1_score(y_train,y_predict_train)
Recall_train=recall_score(y_train,y_predict_train)
pre_train=precision_score(y_train,y_predict_train)
Train_performance=[acc_train,f1_train,Recall_train,pre_train]
Train_performance = pd.DataFrame(Train_performance,index=names_array)

test['Probability'] = rf.predict_proba(X_test)[:,1]
test['Target'] = test[TARGET]
test[TARGET] = test[TARGET].map({0: 'No', 1: 'Yes'})
#arr1= pd.cut(test['Age'],bins=[0,25,50,80],labels=['young','adult','Elderly'])
#arr1=[test['Age'].max(),test['Age'].min(),test['Age'].mean()]
#dframe1 = pd.Series(arr1) 
# Helper functions for dropdowns and slider
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options
def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value
def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks
fig = px.bar(feat_imp, y='cols', x='imps', orientation="h",  color=["#ffffff", "#ffe6e6", "#ffcccc","#ffb3b3","#ff9999","#ff8080","#ff6666","#ff4d4d","#ff3333","#ff1a1a","#F54545"],color_discrete_map="identity", title='Features importance')
#fig.show()
fig1 = px.bar(Test_performance ,orientation='h',color=Test_performance.index,
              color_discrete_map={
                "Precision Score": "#214F8B",
                "Recall Score": "#F54545",
                "f1 Score": "#8C3D2E",
                "accuracy": "#FEB099"}, 
              title='Model preformance with testing set')
#fig1 = px.bar(Test_performance ,orientation='h',color=Test_performance.index, title='Model preformance with testing set')
#fig1.show()
#fig2 = px.bar(Train_performance,y=Train_performance.index ,orientation='h',title='Model preformance with training set')
fig2 = px.bar(Train_performance,y=Train_performance.index ,orientation='h',color=Test_performance.index,
             color_discrete_map={
                "Precision Score": "#214F8B",
                "Recall Score": "#F54545",
                "f1 Score": "#8C3D2E",
                "accuracy": "#FEB099"},
                 title='Model preformance with training set')
app = dash.Dash(__name__)
app.layout = html.Div([#html.H1("Heart disease predictions using randomforest model",style={'color': 'blue', 'fontSize': 30, 'margin-left': 150}),
    html.Div([
        html.H1("Heart Disease model using randomforest"),
        html.P("Summary of predicted probabilities for heart test dataset."),
        # html.Img(src=app.get_asset_url("left_pane.png")),
        html.Img(src="assets/download (1).png"),
        html.Label("ST_Slope", className='dropdown-labels'), 
        dcc.Dropdown(id='class-dropdown', className='dropdown', multi=True,
                     options=create_dropdown_options(test['ST_Slope'].map({1:'Down',2:'Flat', 3:'Up'})),
                     value=create_dropdown_value(test['ST_Slope'].map({1:'Down',2:'Flat', 3:'Up'}))),
        
        html.Br(),
        html.Label("Gender", className='dropdown-labels'), 
        dcc.Dropdown(id='gender-dropdown', className='dropdown', multi=True,
                     options=create_dropdown_options(test['Gender'].map({0:'Female',1:'Male'})),
                     value=create_dropdown_value(test['Gender'].map({0:'Female',1:'Male'}))),
        html.Br(),
        html.Label("ChestPainType", className='dropdown-labels'), 
        dcc.Dropdown(id='chestpain-dropdown', className='dropdown', multi=True,
                     options=create_dropdown_options(test['ChestPainType'].map({1:'ASY',2:'ATA',3:'NAP',4:'TA'})),
                     value=create_dropdown_value(test['ChestPainType'].map({1:'ASY',2:'ATA',3:'NAP',4:'TA'}))),
        #html.Br(),
        #html.Label("Cholesterol", className='dropdown-labels'),
        #dcc.Dropdown(id='Cholesterol-dropdown', className='dropdown', multi=True,
                     #options=create_dropdown_options(test_Copy['Cholesterol']),
                     #value=create_dropdown_value(test_Copy['Cholesterol'])),
        html.Button(id='update-button', children="Update", n_clicks=0),
        ], id='left-container', style={'color': 'blue', 'fontSize': 30, 'margin-top': -150}),
    html.Div([
        html.Div([
            dcc.Graph(id="histogram"),
            dcc.Graph(id="barplot",figure=fig)
        ], id='visualisation'),
        html.Div([
            dcc.Graph(id='table'),
            html.Div([
                html.Br(),
                html.Label("Heart Disease (Yes)", className='other-labels'), 
                daq.BooleanSwitch(id='target_toggle', className='toggle', color="#F54545", on=True),
                html.Br(),
                html.Label("Sort probability in ascending order", className='other-labels'),
                daq.BooleanSwitch(id='sort_toggle', className='toggle', color="#F54545", on=True),
                html.Br(),
                html.Label("Number of records", className='other-labels'), 
                dcc.Slider(id='n-slider', min=5, max=20, step=1, value=10, 
                           marks=create_slider_marks([5, 10, 15, 20])),
                html.Br()
            ], id='table-side'),
        ], id='data-extract'),
        html.Div([
         html.Div([
             dcc.Graph(id='barplot1',figure=fig2),
             dcc.Graph(id='barplot2',figure=fig1)
         ])], id='visualisation1'),
   ], id='right-container')
], id='container')

@app.callback(
    [Output(component_id='histogram', component_property='figure'),
     #Output(component_id='barplot', component_property='figure'),
     Output(component_id='table', component_property='figure')],
    [State(component_id='class-dropdown', component_property='value'),
     State(component_id='gender-dropdown', component_property='value'),
     State(component_id='chestpain-dropdown', component_property='value'),
     #State(component_id='Cholesterol-dropdown', component_property='value'),
     Input(component_id='update-button', component_property='n_clicks'),
     Input(component_id='target_toggle', component_property='on'),
     Input(component_id='sort_toggle', component_property='on'),
     Input(component_id='n-slider', component_property='value')]
)
def update_output(class_value, gender_value, chestpain_value, n_clicks, target, ascending, n):
    dff = test.copy()
    
    if n_clicks>0:
        if len(class_value)>0:
            dff = dff[dff['ST_Slope'].map({1:'Down',2:'Flat', 3:'Up'}).isin(class_value)]
        elif len(class_value)==0:
            raise dash.exceptions.PreventUpdate
        
        if len(gender_value)>0:
            dff = dff[dff['Gender'].map({0:'Female',1:'Male'}).isin(gender_value)]
        elif len(gender_value)==0:
            raise dash.exceptions.PreventUpdate
            
        if len(chestpain_value)>0:
            dff = dff[dff['ChestPainType'].map({1:'ASY',2:'ATA',3:'NAP',4:'TA'}).isin(chestpain_value)]
        elif len(chestpain_value)==0:
            raise dash.exceptions.PreventUpdate
      
    
    # Visual 1: Histogram
    histogram = px.histogram(dff, x='Probability', color=TARGET, opacity=0.6, marginal="box", 
                             color_discrete_sequence=['#F54545', '#214F8B'], nbins=30)
    histogram.update_layout(title_text=f'Distribution of probabilities by class (n={len(dff)})',
                            font_family='Tahoma', plot_bgcolor='rgba(255,242,204,100)')
                            # paper_bgcolor='rgba(0,0,0,0)',
                            
    histogram.update_yaxes(title_text="Count")

    # Visual 2: Barplot
    

    # Visual 3: Table
    if target==True:
         dff = dff[dff['Target']==1]
    else:
        dff = dff[dff['Target']==0]
          
    dff = dff.sort_values('Probability', ascending=ascending).head(n)
    
    columns = ['Age', 'Gender', 'RestingECG', TARGET, 'Probability']
    table = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color='#F54545', line_color='white',
                    font=dict(color='white', size=13), align='center'),
        cells=dict(values=[dff[c] for c in columns], format=["d", "", "", "", "", ".2%"],
                   fill_color=[['white', '#FFF2CC']*(len(dff)-1)], align='center'))
    ])
    table.update_layout(title_text=f'Sample records (n={len(dff)})', font_family='Tahoma')

    return histogram, table

if __name__ == '__main__':
    app.run_server(port=8800)