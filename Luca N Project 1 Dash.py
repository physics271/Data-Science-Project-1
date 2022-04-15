import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from urllib.request import urlopen
import json
import os

from dash import Dash, html, dcc, Input, Output

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

with open('./data_insights.txt', 'r') as f:
    data_insights = f.read().split('\n\nNEWTOPIC\n\n')

expenditure = pd.read_csv('./Cleaned Data/expenditure.csv', dtype={'FIPS':'str'})
diabetes_data = pd.read_csv('./Cleaned Data/diabetes.csv', dtype={'FIPS':'str'})
alcohol_data = pd.read_csv('./Cleaned Data/alcohol.csv', dtype={'FIPS':'str'})
life_data = pd.read_csv('./Cleaned Data/expectancy.csv', dtype={'FIPS':'str'})

limited_exp = expenditure[expenditure['Instruction Spending Per Student, 2018']<15000]

def make_map(df, variable, title=None, quant=0.01, tick_form=('','')):
    crange = tuple(df[variable].quantile([quant,1-quant]))
    map_fig = px.choropleth(df, geojson=counties, locations='FIPS', color=variable,
                            color_continuous_scale="Viridis", range_color=crange,
                            scope="usa", hover_name = 'PlaceName',
                            hover_data = {'FIPS':False}, title=title)
    
    map_fig.update_coloraxes(colorbar={'orientation':'h', 'thickness':10, 'y':-0.1, 'len':0.7,
                                      'title':{'text':''}, 'tickprefix':tick_form[0],'ticksuffix':tick_form[1]})
    
    map_fig.update_layout(margin=dict(l=0, r=0, t=33, b=20))
    
    return map_fig

def merged_scatter(dfs, variables, title=None, logit=False):
    df = pd.merge(left=dfs[0], right=dfs[1], left_on='FIPS', right_on='FIPS')

    if logit:
        new_col = f'Logistic {variables[1]}'
        df[new_col] = -np.log(100./df[variables[1]] - 1)
        fig = px.scatter(df, x=variables[0], y=new_col, trendline="ols", hover_name='PlaceName_x',
                    trendline_color_override="red",)
    else:
        fig = px.scatter(df, x=variables[0], y=variables[1], trendline="ols", hover_name='PlaceName_x',
                        trendline_color_override="red",)

    fig.update_layout(margin=dict(l=10, r=30, t=5, b=15))
    
    results = px.get_trendline_results(fig).iloc[0][0]

    stats_summary = results.summary(yname=f'{variables[1][:14]}...', slim=True)
    
    return fig, stats_summary.as_text()[:-234]

data_sources = {
    'Instruction Spending Per Student, 2018': expenditure,
    'Diabetes Percentage, 2019': diabetes_data,
    'Heavy Drinking Percentage, 2012': alcohol_data,
    'Life expectancy, 2014': life_data
}

map_sources = {
    key: make_map(val, key, title=key, tick_form = tick_form) for (key, val), tick_form in 
    zip(data_sources.items(), (('$',''), ('','%'), ('','%'), ('','')))
}

summaries = {key:val for key, val in 
    zip(('Diabetes Percentage, 2019', 
        'Heavy Drinking Percentage, 2012',
        'Life expectancy, 2014'), 
    data_insights)
}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("Compare Two Maps of the US", 
                style={'display':'flex', 'justify-content': 'center'}),
        html.Div([
            dcc.Graph(id='map-1', config = {'scrollZoom': False}, style={'width':'48%'}), 
            html.Div(style={'width':'1px', 'background-color':'#000'}),
            dcc.Graph(id='map-2', config = {'scrollZoom': False}, style={'width':'48%'})
            ], 
            style={'display':'flex', 'flex-direction':'row', 'margin':'auto', 'justify-content':'center'}),
        html.Div([
            html.Label([
                'Map 1',
                dcc.Dropdown(
                    id='map-1-dropdown', clearable=False, value='Instruction Spending Per Student, 2018', 
                    options=[{'label': c, 'value': c} for c in map_sources.keys()]
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Label([
                'Map 2',
                dcc.Dropdown(
                    id='map-2-dropdown', clearable=False, value='Diabetes Percentage, 2019', 
                    options=[{'label': c, 'value': c} for c in map_sources.keys()]
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'display':'flex', 'justify-content':'space-around'}),
    ]),
    html.H2("The Impact of Student Instruction Spending", 
        style={
        'padding-top':'2%', 
        'display':'flex', 
        'justify-content': 'center'
    }),
    dcc.Graph(id='scatter', style={'width':'90%', 'margin':'auto'}),
    html.Div([
        html.Div([
            html.Label([
                "y-Axis Data Source",
                dcc.Dropdown(
                    id='scatter-dropdown', clearable=False,
                    value='Diabetes Percentage, 2019', options=[
                        {'label': c, 'value': c}
                        for c in summaries.keys()
                    ]),
            ], style={'width':'100%', 'padding-bottom':'5px'}),
            html.Label([
                "Logistic y-Scale (For Percentage Values)",
                dcc.RadioItems(
                    id='logit-radio',
                    value=True, options=[
                        {'label':str(c), 'value':c}
                        for c in [True, False]],
                    inline=True)
            ])
        ]),
        html.Div(id='summary', 
             style={"white-space": "pre", 'font-family': 'Monaco', 'font-size':'1vw'})
    ], style={
        'display':'flex', 
        'flex-direction':'row', 
        'align-items':'start',
        'justify-content':'space-between',
        'width':'90%',
        'margin':'auto'
        }),
    html.H4("Data Insights", style={
        'display':'flex', 
        'justify-content': 'center',
        'padding-top':'3%'
    }),
    html.Div(id='data insights', style={
        'display':'flex', 
        'margin':'auto',
        'width':'70%',
        "white-space": "pre-wrap", })
],  )

@app.callback(
    [Output('scatter', 'figure'),
     Output('summary', 'children')],
    [Input("scatter-dropdown", "value"),
    Input('logit-radio', "value")]
)
def update_scatter(value, logit):
    source = data_sources[value]
    fig, summary= merged_scatter((limited_exp, source), ['Instruction Spending Per Student, 2018', value], logit=logit)
    return fig, summary

@app.callback(
    Output('map-1', 'figure'),
    [Input("map-1-dropdown", "value")]
)
def update_map(value):
    return map_sources[value]

@app.callback(
    Output('map-2', 'figure'),
    [Input("map-2-dropdown", "value")]
)
def update_map(value):
    return map_sources[value]

@app.callback(
    Output('data insights', 'children'),
    [Input("scatter-dropdown", "value")]
)
def update_summary(value):
    return summaries[value]

if __name__ == '__main__':
    app.run_server(debug=True)