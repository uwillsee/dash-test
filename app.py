import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import gunicorn

import numpy as np
import pandas as pd
from re import search

import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')


def classes_fromto(year1=1929, year2=2020):
    return px.bar(data[(data['DateAcquired'] >= year1) & (data['DateAcquired'] <= year2)]['Classification'].value_counts(), labels=dict(value='count', index='classes')).update_layout(showlegend=False, title=f'Classes distribution from {year1} to {year2}')


def countries_distribution(year=2020, group_smallest=True, group_method='mean'):
    temp = data[data['DateAcquired'] <= year]['Country'].value_counts()
    if group_smallest:
        other = temp[temp <= (np.mean(temp) if group_method ==
                              'mean' else np.median(temp))]
        temp.drop(other.index, inplace=True)
        temp = temp.append(pd.Series({'Other': np.sum(other)}))
    return px.bar(temp, labels=dict(value='count', index='countries')).update_layout(showlegend=False, title=f'Countries distribution in {year}')


def bar_with_animation():
    classes = ['Architecture', 'Design', 'Drawing', 'Illustrated Book', 'Painting', 'Photograph', 'Print', 'Sculpture']
    temp = pd.DataFrame(columns=['Year', 'Classification', 'Count'])
    for year in sorted(data['DateAcquired'].unique()):
        for class_ in classes:
            count = data[(data['DateAcquired'] == year) & (data['Classification'] == class_)].count()['Title']
            temp = temp.append({'Year': year, 'Classification': class_, 'Count': count}, ignore_index=True)
    temp['CountLog'] = temp[temp['Count'] != 0]['Count'].astype(float).apply(np.log)
    temp['Count'] = temp[temp['Count'] != 0]['Count'].astype(int)
    temp = temp.replace(np.nan, 0)
    fig = px.bar(
        temp,
        x='Classification',
        y='CountLog',
        animation_frame='Year',
        labels={'CountLog': 'Number of Artworks (log)', 'Count': 'Number of artworks'},
        color_discrete_sequence=['#3DCCC0'],
        range_y=[0.1, max(temp['CountLog'])],
        hover_data={'Year': False, 'Classification': False, 'Count': True, 'CountLog': False}
    )
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(
        title=dict(text='Classification of Acquired Artworks',
                   x=0.05, y=0.95
                   ),
        template='ggplot2'
    )
    return fig


def acquired_plot(total=True):

    if total:
        temp = data.groupby('DateAcquired').count().cumsum().reset_index()
    else:
        temp = data.groupby('DateAcquired').count().reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=temp['DateAcquired'],
            y=temp['Artist']
        )
    )

    fig.update_layout(
        title='Total number of art acquired' if total else 'Arts acquired',
        xaxis_title='Year',
        yaxis_title='Arts',
    )

    return fig

def sunburst(countries=None):
    if countries:
        temp = data[data['Country'].isin(countries)]
    else:
        temp = data.copy()
    fig = px.sunburst(temp, path=['Department', 'Classification'], color_discrete_sequence=['#4a4bc7'])
    fig.update_traces(
        go.Sunburst(hovertemplate='Number of artworks=%{value}'
    ))
    fig.update_layout(
        title=dict(text='Artworks Classification Arranged by Department', font=dict(color='black'))
    )
    return fig

def genders_chart():
    df = pd.pivot_table(data, index='DateAcquired', columns='Gender', values='Artist', aggfunc='count')
    df = df.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['DateAcquired'], y=df['Female'],
        fillcolor='rgb(111, 231, 219)',
        mode='lines',
        line=dict(width=0.5, color='rgb(111, 231, 219)'),
        stackgroup='one',
        name='Female',
        groupnorm='percent',  # sets the normalization for the sum of the stackgroup
        hovertemplate='Year=%{x}' + '<br>Percentage=%{y:.2f}' + '%'

    ))

    fig.add_trace(go.Scatter(
        x=df['DateAcquired'], y=df['Male'],
        fillcolor='rgb(74, 75, 199)',
        mode='lines',
        line=dict(width=0.5, color='rgb(74, 75, 199)'),
        stackgroup='one',
        name='Male',
        hovertemplate='Year: %{x}' + '<br>Percentage: %{y:.2f}' + '%'
    ))

    fig.update_layout(
        title=dict(text='Gender of Artists Over Time',
                   x=0.05, y=0.95
                   ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            font=dict(family="Arial", size=12),
            bordercolor="white",
            borderwidth=1,
            itemsizing='trace'
        ),
        xaxis_type='linear',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    fig.update_yaxes(title='Percentage of Artworks Acquired')
    fig.update_xaxes(title='Year Acquired')
    fig.update_layout(template="simple_white")

    return fig

def line_chart_nationalities():
    temp = data.groupby(['DateAcquired', 'Country'])['Title'].count().reset_index().rename({'Title': 'Count'}, axis=1)
    # Number of different countries per year
    temp2 = temp.groupby('DateAcquired')['Country'].count().reset_index()

    # Number of new countries by the years
    new_country = []
    new = []
    temp2['New'] = np.nan
    for year in temp['DateAcquired'].unique():
        country_year = temp[temp['DateAcquired'] == year]['Country'].values
        new_year = [country for country in country_year if country not in new_country]
        for n in new_year: new_country.append(n)
        new.append(len(new_country))
    temp2['New'] = new
    temp2.rename({'Country': 'Different nationalities by year', 'New': 'Different nationalities until that year'},
                 axis=1, inplace=True)

    fig = px.line(temp2, x='DateAcquired',
                  y=['Different nationalities by year', 'Different nationalities until that year'],
                  hover_data={'DateAcquired': False, 'variable': False, 'value': True},
                  color_discrete_sequence=['#9acfbf','#3eceaf'],
                  labels={'DateAcquired': 'Year Artworks were Acquired', 'variable': '', 'value': 'Number of Nationalities'}
                  )

    fig.update_layout(
        title=dict(text='Diversity of Artists\' Origins Over Time',
                   x=0.05, y=0.95
                   ),
        template='ggplot2',
        hovermode="x",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            font=dict(family="Arial", size=12),
            bordercolor="white",
            borderwidth=1,
            itemsizing='trace'
        )
    )
    fig.update_traces(hovertemplate='%{y}<extra></extra>',
                      line=dict(width=3))

    return fig

def map_with_animation():
    # 2 options (comment the one you don't want)
    ### With growth
    temp = data.groupby(['DateAcquired', 'Country'])['Title'].count().unstack().replace(np.nan, 0).cumsum()
    temp = pd.DataFrame(temp.stack()).reset_index().rename({0: 'Count'}, axis=1)

    # logarithm
    temp['Countlog'] = np.log(temp['Count'])

    fig = px.choropleth(temp, locations='Country', locationmode='country names',
                        color='Countlog',
                        color_continuous_scale=['#e8e6ff', '#dcd9ff', '#b2b0ff', '#8889e0', '#6063b6', '#4a4bc7', '#33357f'],
                        animation_frame='DateAcquired',
                        range_color=(0, temp['Countlog'].max()),
                        hover_name='Country',
                        hover_data={'Country':False,'Count':True,'Countlog':False,'DateAcquired':False},
                        labels={'DateAcquired': 'Year', 'Countlog': 'Acquired<br>Artworks (log)','Count': 'Acquired Artworks'}
                        #projection="natural earth"
                        )
    fig.update_layout(
        title=dict(text='Artworks\' Nationality Evolution',
                                           x=0.1,y=0.95,
                   font=dict(color='black')
            )
    )
    return fig


def statistics(countries = None):
    if countries:
        temp = data[data['Country'].isin(countries)]
    else:
        temp = data.copy()
    unique_artsits = temp['Artist'].unique().shape[0]
    unique_artworks = temp['Title'].unique().shape[0]
    gender_counts = temp['Gender'].value_counts(normalize = True) * 100
    try:
        male = str(round(gender_counts['Male'], 2)) + '%'
    except:
        male = '0.0%'
    try:
        female = str(round(gender_counts['Female'], 2)) + '%'
    except:
        female = '0.0%'


    return unique_artsits, unique_artworks, male, female


def unique_countries():
    # select countries with paintings works
    classification_paint = ['Drawing', 'Painting', 'Work on Paper']
    data_withpaint = data[data.Classification.isin(classification_paint)]

    return data_withpaint['Country'].sort_values().unique()

def filter_technique(technique):
    if search('and', technique):
        return 'Mixed'
    elif search('mixed', technique.lower()):
        return 'Mixed'
    elif search('oil', technique.lower()):
        return 'Oil'
    elif search('charcoal', technique.lower()):
        return 'Charcoal'
    elif search('acrylic', technique.lower()):
        return 'Acrylic'
    elif search('tempera', technique.lower()):
        return 'Tempera'
    elif search('pencil', technique.lower()):
        return 'Pencil'
    elif search('ink', technique.lower()):
        return 'Ink'
    elif search('watercolour', technique.lower()):
        return 'Watercolour'
    elif search('watercolor', technique.lower()):
        return 'Watercolour'
    elif search('crayon', technique.lower()):
        return 'Crayon'
    elif search('gouache', technique.lower()):
        return 'Gouache'
    elif search('paint', technique.lower()):
        return 'Paint'
    elif search('dye', technique.lower()):
        return 'Paint'
    elif search('pigment', technique.lower()):
        return 'Paint'
    elif search('pastel', technique.lower()):
        return 'Pastel'

    else:
        return 'Other'


def donut_chart(countries = None):
    if countries:
        temp = data[data['Country'].isin(countries)]
    else:
        temp = data.copy()
    data_drawings = temp[(data.Classification == 'Drawing') | (
        data.Classification == 'Painting') | (data.Classification == 'Work on Paper')]
    data_drawings['Technique'] = data_drawings['Medium'].apply(
        filter_technique)
    data_drawings = data_drawings[data_drawings['Medium']
                                  != 'Not known'].reset_index()

    # Create donut graph for different techniques
    donut_chart = go.Figure()
    sum_counts = data_drawings['Technique'].value_counts().sort_index()
    # Colour palette for graph
    colors = ['#516CCC',  # Acrylic

              '#7799E0',  # Charcoal

              '#6084B6',  # Crayoin

              '#82BCFA',  # goauche

              '#3DCCC0',  # ink

              '#00ae91',  # mixed

              '#00C2B2',  # oil

              '#63E6E3',  # other

              '#8DD8EB',  # paint

              '#4A4BC7',  # pastel

              '#81D7CF',  # pencil

              '#7087CF',  # tempera

              '#7DADE6']  # watecolour

    donut_chart.add_trace(go.Pie(values=sum_counts,
                                 labels=sum_counts.index,
                                 hole=0.88,
                                 hovertemplate="%{label}<br>Number of artworks=%{value}<extra></extra>",
                                 marker=dict(colors=colors, line=dict(color='#ffffff', width=1))
                                 ))
    donut_chart.update_layout(
        title=dict(text='Most Popular Painting Techniques',font=dict(color='black')),
        showlegend=True,
        annotations=[dict(text='Techniques<br>used<br>by artists', x=0.5, y=0.5, font_size=20, showarrow=False)])

    return donut_chart


app = dash.Dash(__name__,
                title="MoMA on Tour", suppress_callback_exceptions=True)

app.layout = dcc.Loading(
    html.Div(
        children=[
            # title
            html.Div(
                children=[
                    html.Img(
                        src=app.get_asset_url("moma-logo.png"),
                        style={'width': '80%', 'margin-top': '10px'}
                    ),
                ],
            style={'width': '25%'}
            ),
            html.Div(
                children=[
                    html.H5('The Museum of Modern Art (MoMA) acquired its first artworks in 1929.'
                            ' Today, the Museum’s evolving collection contains almost 200,000 works'
                            ' from around the world spanning the last 150 years. In this dashboard, '
                            'you can go on tour with the MoMA museum by getting insights into which '
                            'artworks it acquired over the years and by which artists. Next, you can '
                            'see MoMA per country by checking which country the art pieces come from. '
                            'The art collections include an ever-expanding range of visual expression, '
                            'including painting, sculpture, photography, architecture, design, and '
                            'film art. Travel through time and space with MoMA and enjoy the tour...'),
                    ],
                    className='card',
                    style={"height": "25%","width":"70%"},
            ),
            #1st row
            html.Div(
                children=[
                #1st column
                    dcc.Graph(
                        figure=line_chart_nationalities(),
                        className='card',
                        style={"width": "50%"},
                    ),
                #2nd column
                    html.Div(
                        children=[
                            dcc.Loading([
                                    dcc.Graph(figure=map_with_animation(),
                                              id='main-choropleth')],
                                    type='default', color='black', id="map-loading"
                            )
                        ],
                        className='card',
                        style={'width': '50%'}
                    ),
                ],
                className='container'
            ),

        #2nd row
        html.Div(
            children=[

                    dcc.Graph(
                        figure=genders_chart(),
                        className='card',
                        style={
                            "width": "40%"
                        }
                    ),
                    dcc.Graph(
                        figure=bar_with_animation(),
                        className='card',
                        style={
                            "width": "55%"
                        }
                    ),
            ],
            className='container'
        ),

        #3rd row
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H2("Choose countries"),
                                        dcc.Dropdown(
                                            options=[{'label': v, 'value': v}
                                                     for v in unique_countries()],
                                            multi=True,
                                            id='countries-dropdown'),
                                    ],
                                    className='first card',
                                    style={'width': '40%', 'float': 'right'}
                                ),
                html.Div(children=[
                    html.H2('Artists'),
                    html.H3('123', id='unique-artists')
                ],
                    className='card small'
                ),
                html.Div(
                    children=[
                        html.H2('Artworks'),
                        html.H3('123', id='unique-artworks')
                    ],
                    className='card small'
                ),
                html.Div(
                    children=[
                        html.H2('Gender'),
                        html.H3('Male:'),
                        html.H3('123%', id='male-count'),
                        html.H3(' Female:'),
                        html.H3('123%', id='female-count')
                    ],
                    className=' card small'
                ),
            ],
            className='container'
        ),

        #4th row
        html.Div(
            children=[
                dcc.Graph(
                    figure=sunburst(),
                    className='card',
                    style={'width': '50%'},
                    id='sunburst'
                ),
                dcc.Graph(
                    figure=donut_chart(),
                    className='card',
                    style={'width': '50%'},
                    id='donut'
                )

            ],
            className='container'
        ),



        html.Footer([
            html.Label(["Anastasiia Tagiltseva, m20200041 | Beatriz Pereira, m20200674 | Nadine Aldesouky, m20200568 | Svitlana Vasylyeva, m20200617| "
                    "Source: ", html.A("MoMA",
                                          href="https://github.com/MuseumofModernArt/collection", target="_blank")])

    ],
        className="footer"
    ),
    ]),
    type = 'cube',  color='white')

server = app.server

@app.callback(
    Output('unique-artists', 'children'),
    Output('unique-artworks', 'children'),
    Output('male-count', 'children'),
    Output('female-count', 'children'),
    Output('sunburst', 'figure'),
    Output('donut', 'figure'),
    Input('countries-dropdown', 'value')
)
def update_stats(value):
    stats = statistics(value)
    return stats[0], stats[1], stats[2], stats[3], sunburst(value), donut_chart(value)


if __name__ == '__main__':
    app.run_server(debug=True)
