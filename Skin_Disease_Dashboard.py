# IMPORT BASIC LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# PLOTY FOR INTERACTIVE GRAPH
import plotly.express as px
import plotly.graph_objects as go

# DASH FOR SETTING ALL COMPONENT OF THE DASHBOARD
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

# SETTING UP APP
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.config["suppress_callback_exceptions"] = True


# LOADING AND PREPROCESS DATA 
metadata = pd.read_csv('HAM10000_metadata.csv')
# change values for pathology to more meaningfull name
metadata = metadata.replace({
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
})



# CREATE FIGURE FOR THE GRAPH (using plotly and updating layout for style)

# datatable
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(metadata.columns[i] for i in [1,5,4,2,6,3]),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[metadata.image_id, metadata.sex, metadata.age, metadata.dx, metadata.localization, metadata.dx_type],
               fill_color='lavender',
               align='left'))
])
fig_table.update_layout(plot_bgcolor='#FFFFFF')

# pie chart for gender distribution
fig_pie = px.pie(metadata, names='sex')
fig_pie.update_layout(title_text='',title_x=0.5,plot_bgcolor='#192444',paper_bgcolor='white',font_color='black')

# histogram for age distribution
fig_age = px.histogram(metadata, x="age")
fig_age.update_layout(margin_pad=5,plot_bgcolor='#FFFFFF')

# bar plot for localization distribution
fig_localization = go.Figure(data=go.Bar(x=metadata['localization'].value_counts(),y=metadata['localization'].unique(),orientation='h'))
fig_localization.update_layout(width=500,plot_bgcolor='#FFFFFF')

# bar plot for pathology distribution
fig_patho = go.Figure(data=[go.Bar(name='male', y=metadata['dx'][metadata['sex'] == 'male'].value_counts(),x=metadata['dx'].unique()),
                     go.Bar(name='female',y=metadata['dx'][metadata['sex'] == 'female'].value_counts(),x=metadata['dx'].unique())])
fig_patho.update_layout(plot_bgcolor='#FFFFFF')


# Create option for table filter
age = [dbc.DropdownMenuItem('< 40'),dbc.DropdownMenuItem('< 60'), dbc.DropdownMenuItem('< 80')]
genre = [dbc.DropdownMenuItem('Male'),dbc.DropdownMenuItem('Female')]
local = [dbc.DropdownMenuItem(i) for i in metadata['localization'].unique()]
diag = [dbc.DropdownMenuItem(i) for i in metadata['dx'].unique()]

# create filter card
table_filtre = dbc.Card(
    [
        html.H5("Filtering data"),
        dbc.FormGroup(
            [
                dbc.Label("Age"),
                dcc.Dropdown(id="age_input",
                options=[{"label": 'All', "value": 'All'}]+[{"label": "< 40", "value": 40},{"label": "< 60", "value": 60},{"label": "< 80", "value": 80}],
                value='All'),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Sex"),
                dcc.Dropdown(id="genre_input",
                options=[{"label": 'All', "value": 'All'}]+[{"label": 'Male', "value": 'male'},{"label": 'Female', "value": 'female'}],
                value='All'),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("localization"),
                dcc.Dropdown(id="local_input",
                options=[{"label": 'All', "value": 'All'}]+[{"label": col, "value": col} for col in metadata['localization'].unique()],
                value='All'),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Pathology"),
                dcc.Dropdown(id="patho_input",
                options=[{"label": 'All', "value": 'All'}]+[{"label": col, "value": col} for col in metadata['dx'].unique()],
                value='All'),
            ]
        ),
    ],
    # setting style for the filter card
    style={'color':'#119DFF','backgroundColor':'#1F2C56'},
    body=True,
)


table = dash_table.DataTable(
    id='datatable-interactivity',
    columns=[
        {"name": i, "id": i, "deletable": True, "selectable": True} for i in list(metadata.columns[i] for i in [1,5,4,2,6,3])
    ],
    data=metadata.to_dict('records'),
    editable=False,
    filter_action="none",
    sort_action="native",
    sort_mode="multi",
    row_selectable="single",
    row_deletable=False,
    selected_rows=[],
    page_action="native",
    page_current= 0,
    page_size= 12,
),

# age range slider values
mark_values = {0:'0',5:'5',10:'10',15:'15',
               20:'20',25:'25',30:'30',35:'35',
               40:'40',45:'45',50:'50',55:'55',
               60:'60',65:'65',70:'70',75:'75',
               80:'80',85:'85'}


# CREATING THE FRAME OF APPLICATION USING BOOTSTRAP COMPONENT AND HTML
app.layout = dbc.Container(
    [
        # First element = header + main panel (with blue background)
        html.Div(
            [
                # Header
                dbc.Row(
                    [                      
                        dbc.Col(
                            [
                                html.H2("SKIN DISEASE DASHBOARD"),
                                html.P("This app is intended for visualizing in a smart fashion health data from 10 000 patients suffering of skin disease.")
                            ],style={'color':'white'}      
                        ),
                        
                        dbc.Col(
                            [
                                html.Img(
                                    src=app.get_asset_url("dash-new-logo.png"),
                                    style={'height':'3rem'}
                                )
                            ],md=3
                        ),
                    ],
                    style={'backgroundColor':'#192444','padding':'2%'},
                    justify='center',
                ),

                html.Hr(),

                # Main pannel
                dbc.Row(
                    [
                        # filter card
                        dbc.Col(table_filtre, md=3),
                        # data table
                        dbc.Col(html.Div(id='table'), md=5),
                        # image
                        dbc.Col(html.Div(id='image'),md=3),
                    ],
                    align='center',
                    justify='center',
                ),
            ],
            style={'backgroundColor':'#192444','color':'white','padding-bottom':'4%'}

        ),
        

        # Second element = graphs + footer (with grey background)
        html.Div(
            [
                html.Hr(),
                
                # first 2 graphs
                dbc.Row(
                    [                     
                        dbc.Col(children=[
                            html.H4('Gender distribution'),
                            dcc.Graph(
                            figure=fig_pie
                            ),
                        ],
                        width=5),

                        dbc.Col(children=[
                                    html.H4('Pathology distribution by gender and age'),
                                    dcc.Graph(
                                        id='example-graph-6',
                                        figure=fig_patho
                                    ),
                                    html.Div(
                                        [
                                            dcc.RangeSlider(id='the_age',
                                                min=0,
                                                max=85,
                                                value=[0,5],
                                                marks=mark_values,
                                                step=None),
                                                
                                        ],style={'padding-top':'2%'},),
                                    
                                    
                                    ],width=7,
                                    
                                ),                
                    ],
                    justify="center",
                ),

                html.Hr(),

                # second 2 graphs
                dbc.Row(
                    [
                        dbc.Col(children=[
                                    html.H4('Areas that are the most affected'),
                                    dcc.Graph(
                                        figure=fig_localization
                                    ),
                                ],width=5
                            ),

                        dbc.Col(children=[
                                    html.H4('Age distribution'),
                                    dcc.Graph(
                                        figure=fig_age
                                    ),
                                ],width=7
                            ),
                    ],justify='center',
                ),

                html.Hr(),
                
                # Footer
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(" RESSOURCES "),
                                            dbc.CardBody(
                                                [
                                                    html.P(" MNIST HAM 10 000 is a public data set that contain 10 000 images of skin disease patients as well as some medical metadata. This data set is intended to train model to perform classification and detect anomalies."),
                                                    html.A(dbc.Button('View Source'),href='https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000'),
                                                ]
                                            )
                                        ],color='#1F2C56',inverse=True,
                                    ),
                                    md=5,
                                ),

                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(" ABOUT US"),
                                            dbc.CardBody(
                                                [
                                                    html.P(" We are Bioinformatics student trying to apply data science and ML to solve medical problems. For this dashboard we are using those data to provide a functional application allowing to visualize rapidly patient statistics and images."),
                                                    html.A(dbc.Button('View Code'),href='https://github.com/diabamine'),
                                                ]
                                            )
                                        ],color='#1F2C56',inverse=True,
                                    ),
                                    md=5,
                                ),
                            ], justify='center',
                        )

                    ]
                ),

                html.Hr(),

            ],
            style={'backgroundColor':'#f2f5fa',
                    'padding-left':'5%',
                    'padding-right':'5%'}
        ),

    ],
    # general style of the page
    style={'backgroundColor':'#192444',
    'width':'100%',
    'font-family': '"Open Sans", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif',
    'font-weight': '300',
    'line-height': '1.6',
    'padding-left':'0',
    'padding-right':'0'
    },
    fluid=True,
)

# SETTING UP CALLBACK FOR INTERACTIVITY

# fisrt callback for updating table according to filter
@app.callback(
    Output("table", "children"),
    [
        Input("age_input", "value"),
        Input("genre_input", "value"),
        Input("local_input", "value"),
        Input("patho_input", "value"),
    ],
)
def update_table(age,genre,local,patho):
    # create masks for all options and merging them to retrieve filtered dataframe
    if age != 'All':
        mask_age = metadata['age'] < age
    else:
        # creating mask full of 1 for 'all' option
        mask_age = pd.Series([True]*metadata.shape[0])
    if genre != 'All':
        mask_genre = metadata['sex'] == genre
    else:
        mask_genre = pd.Series([True]*metadata.shape[0])
    if local != 'All':
        mask_local = metadata['localization'] == local
    else:
        mask_local = pd.Series([True]*metadata.shape[0])
    if patho != 'All':
        mask_patho = metadata['dx'] == patho
    else:
        mask_patho = pd.Series([True]*metadata.shape[0])
    
    final_mask = mask_age & mask_genre & mask_local & mask_patho
    filtered = metadata[final_mask]

    # create a datatable witth filtered data 
    updated_table = dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in list(filtered.columns[i] for i in [1,5,4,2,6,3])
        ],
        data=filtered.to_dict('records'),
        editable=False,
        filter_action="none",
        sort_action="native",
        sort_mode="multi",
        row_selectable="single",
        row_deletable=False,
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 12,
        # styling datatable 
        style_as_list_view=True,
        style_header={'backgroundColor': '#119DFF'},
        style_cell={
            'backgroundColor': '#1F2C56',
            'color': 'white'
        },
    ),

    return updated_table

# callback for pathology histogram
@app.callback(
    Output('example-graph-6','figure'),
    [Input('the_age','value')]
)

def update_graph(years_chosen):
    # mask for retrieving data in age range selected
    patho = metadata[(metadata['age']>=years_chosen[0])&(metadata['age']<=years_chosen[1])]

    # create histogram
    fig = go.Figure(data=[go.Bar(name='Male', y=patho['dx'][patho['sex'] == 'male'].value_counts(),x=patho['dx'].unique()),
                        go.Bar(name='Female',y=patho['dx'][patho['sex'] == 'female'].value_counts(),x=patho['dx'].unique())])

    fig.update_layout(
        title='', 
        xaxis=dict(title='Pathology'),
        yaxis=dict(title='Number of patient'),
        title_x=0.5,
        plot_bgcolor='#FFFFFF')

    return fig

# callback for showing image according to row selected 
@app.callback(
    Output("image", "children"),
    [
        Input("datatable-interactivity", "derived_virtual_data"),
        Input("datatable-interactivity", "derived_virtual_selected_rows"),
    ],
)

def update_image(rows,selected_rows):

    # create condition if no rows is selected 
    if selected_rows:
        # retriving image_id for rows selected
        paths=[x.get("image_id") for x in [rows[y] for y in selected_rows]]
        # finding the good path of image
        image_source = 'assets/HAM10000_images/'+ str(paths[0]) +'.jpg'
        # plot image
        img = plt.imread(image_source)
        fig = px.imshow(img, binary_string=True, binary_backend="jpg",)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(width=400, height=400, margin=dict(l=10, r=10, b=10, t=10),paper_bgcolor='#192444')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        card = dcc.Graph(
            figure=fig
        )
    else:
        card = html.Div([html.H5('Please select a patient to view image')],style={'color':'white'})
    
    return card



if __name__ == '__main__':
    app.run_server(debug=True)