import os
import dash
from click import style
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

current_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(current_dir, '../3_varhato_elettartam.csv'))

noempty_data = data.dropna()
data['Year'] = data['Year'].astype(int)

data_2015 = data[data['Year'] == 2015]
population_min = data_2015['Population'].min()
population_max = data_2015['Population'].max()
gdp_min = data_2015['GDP'].min()
gdp_max = data_2015['GDP'].max()
countries = data['Country'].unique()
years=data['Year'].unique()
filtered_data_reg = data.dropna(subset=['GDP', 'Life expectancy '])

keszito_adatai_markdown = '''
### Készítő adatai
- **Név:** Duli Bálint Adrián
- **Neptun kód:** MDI509
- **E-mail:** balint.duli@gmail.com
- **Tanulmányok:** Budapesti Gazdasági Egyetem Pénzügyi És Számviteli Kar
- **Szak:** Gazdaságinformatikus szak, Üzleti Adatelemző specializáció
'''

projekt_adatai_markdown = '''
### Projekt adatai
- **Cél:** A projekt célja a különböző országokban és években mért várható élettartam alakulásának elemzése és ábrázolása diagrammokon. A végső cél összefüggések keresése az egészségügyi adatok és a várható élettartam között, illetve konkluziók levonása. 

- **Megvalósítás módja:** A projektet Python nyelven készítettem, a Dash keretrendszer segítségével. A felhasználók interaktív módon tekinthetik és jeleníthetik meg az adatokat és grafikonokat.

- **Adathalmaz információk:**
    - **Fájl név:** 3_varhato_elettartam.csv
    - **Adatbázis oszlopai:**
        - `Country`: Az ország neve.
        - `Year`: Az év, amelyben az adatokat gyűjtötték.
        - `Status`: Az ország fejlettségi státusza (Fejlett vagy Fejlődő).
        - `Life expectancy`: Várható élettartam az adott évben.
        - `Adult Mortality`: Felnőtt halálozási arány (1000 főre vetítve).
        - `Infant deaths`: Csecsemőhalálozási arány.
        - `Alcohol`: Egy főre jutó alkoholfogyasztás.
        - `Percentage expenditure`: Egészségügyre fordított kiadások az adott országban.
        - `Hepatitis B`: Hepatitis B átoltottsági arány.
        - `Measles`: Kanyaró megbetegedések száma.
        - `BMI`: Az ország átlagos testtömeg-indexe.
        - `Under-five deaths`: Ötéves kor alatt bekövetkezett halálozások.
        - `Polio`: Polio átoltottsági arány.
        - `Total expenditure`: Teljes egészségügyi kiadás százalékban.
        - `Diphtheria`: Diftéria átoltottsági arány.
        - `HIV/AIDS`: HIV/AIDS elterjedtség.
        - `GDP`: Az ország bruttó hazai terméke.
        - `Population`: Népesség nagysága.
        - `Thinness 1-19 years`: 1-19 éves korosztály vékonysági aránya.
        - `Thinness 5-9 years`: 5-9 éves korosztály vékonysági aránya.
        - `Income composition of resources`: Jövedelem és erőforrások összetétele (0-1 közötti index).
        - `Schooling`: Iskolai végzettségi átlag.

'''

app.layout = html.Div([
    html.H1('Várható élettartam vizsgálata', style={'textAlign': 'center','marginBottom': '40px'}),

    html.Hr(),

    html.Div([
        html.H4("Egy választott ország egy főre jutó GDP-jének alakulása évenként", style={'textAlign': 'left'}),
    ], style={'padding': '0 20px', 'marginBottom': '20px'}),

    html.Div([
        html.Div([
            html.P("Válassz országot:", style={'marginRight': '10px', 'marginTop': '5px'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in countries],
                value=countries[0],
                clearable=False,
                style={'width': '250px'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),  # Flexbox stílus

        dcc.Graph(id='gdp-graph')
    ], style={'padding': '20px'}),

    html.Hr(),
    html.Div([
        html.H4("Szűrés a 2015-ös népesség és GDP adatok alapján", style={'textAlign': 'left'}),
    ], style={'padding': '0 20px', 'marginBottom': '20px'}),

    html.Div(id='sliders-container', children=[
        html.Div([
            html.Label("Népesség tartomány:"),
            dcc.RangeSlider(
                id='population-slider',
                min=population_min,
                max=population_max,
                step=(population_max - population_min) / 10,
                value=[population_min, population_max],
                marks={int(i): str(int(i)) for i in
                       range(int(population_min), int(population_max) + 1, int((population_max - population_min) / 10))},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'marginBottom': '30px'}),

        html.Div([
            html.Label("GDP tartomány:"),
            dcc.RangeSlider(
                id='gdp-slider',
                min=gdp_min,
                max=gdp_max,
                step=(gdp_max - gdp_min) / 10,
                value=[gdp_min, gdp_max],
                marks={int(i): str(int(i)) for i in
                       range(int(gdp_min), int(gdp_max) + 1, int((gdp_max - gdp_min) / 10))},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'marginBottom': '30px'}),

        html.Div(id='output-country-list')
    ], style={'padding': '20px'}),

    html.Hr(),

    html.Div([
        html.H4("Várható élettartam alakulása évenként a választott országokban", style={'textAlign': 'left'}),
        html.P("Válassz egy vagy több országot!"),
html.Div([
        dcc.Dropdown(
            id='country-dropdown-multiple',
            options=[{'label': country, 'value': country} for country in countries],
            multi=True,
            value=[countries[0]],
            clearable=False,
            style={'width': '500px'}
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),

    dcc.Graph(id='life-expectancy-graph')
    ], style={'padding': '20px', 'marginBottom': '20px'}),

    html.Hr(),

    html.Div([
        html.H4("Gyakorisági diagramm egy választott évre és változóra nézve", style={'textAlign': 'left'}),
        html.P("Válassz évet és változót!"),
        html.Div([
            html.Div(
                dcc.RangeSlider(
                    id='year-slider',
                    min=data['Year'].min(),
                    max=data['Year'].max(),
                    step=1,
                    value=[data['Year'].min(), data['Year'].max()],
                    marks={year: str(year) for year in range(data['Year'].min(), data['Year'].max() + 1, 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                style={'width': '50%', 'marginRight':'40px'}
            ),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[
                    {'label': 'Felnőtt halálozás', 'value': 'Adult Mortality'},
                    {'label': 'Csecsemőhalálozás', 'value': 'infant deaths'},
                    {'label': 'Alkoholfogyasztás', 'value': 'Alcohol'},
                    {'label': 'Iskolai végzettség', 'value': 'Schooling'},
                    {'label': 'Várható élettartam', 'value': 'Life expectancy '},
                    {'label': 'Átlagos BMI', 'value': ' BMI '},
                    {'label': 'Ötéves kor alatti halálozás', 'value': 'under-five deaths '},
                    {'label': 'Polio átoltottság', 'value': 'Polio'},
                    {'label': 'Diftéria átoltottság', 'value': 'Diphtheria '},
                    {'label': 'Hepatitis B átoltottság', 'value': 'Hepatitis B'},
                    {'label': 'Kanyaró megbetegedések', 'value': 'Measles '},
                    {'label': 'GDP', 'value': 'GDP'},
                    {'label': 'Népesség', 'value': 'Population'},
                    {'label': 'Jövedelem összetétele', 'value': 'Income composition of resources'},
                    {'label': 'Vékony 1-19 évesek', 'value': ' thinness  1-19 years'},
                    {'label': 'Vékony 5-9 évesek', 'value': ' thinness 5-9 years'},
                    {'label': 'Teljes egészségügyi kiadás', 'value': 'Total expenditure'},
                    {'label': 'Egészségügyre fordított kiadások', 'value': 'percentage expenditure'},
                    {'label': 'HIV/AIDS elterjedtség', 'value': ' HIV/AIDS'}
                ],
                value='Adult Mortality',
                clearable=False,
                style={'width': '350px'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'width': '100%'}),

        html.Div([
            html.P("Válasszd ki a gyakorisági osztályközök számát!"),
            dcc.Slider(
                id='bin-size-slider',
                min=1,
                max=100,
                step=1,
                value=10,
                marks={i: str(i) for i in range(1, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}

            )
        ], style={'width': '50%', 'marginTop': '20px', 'align': 'center'}),

        dcc.Graph(id='frequency-graph')
    ], style={'padding': '20px'}),

html.Hr(),

html.Div([
    html.H4("Egy választott változó évenkénti alakulása tematikus térképen", style={'textAlign': 'left'}),
    html.P("Válassz változót a térképhez:"),
    dcc.Dropdown(
        id='variable-dropdown-map',
        options=[
            {'label': 'Felnőtt halálozás', 'value': 'Adult Mortality'},
            {'label': 'Csecsemőhalálozás', 'value': 'infant deaths'},
            {'label': 'Alkoholfogyasztás', 'value': 'Alcohol'},
            {'label': 'Iskolai végzettség', 'value': 'Schooling'},
            {'label': 'Várható élettartam', 'value': 'Life expectancy '},
            {'label': 'Átlagos BMI', 'value': ' BMI '},
            {'label': 'Ötéves kor alatti halálozás', 'value': 'under-five deaths '},
            {'label': 'Polio átoltottság', 'value': 'Polio'},
            {'label': 'Diftéria átoltottság', 'value': 'Diphtheria '},
            {'label': 'Hepatitis B átoltottság', 'value': 'Hepatitis B'},
            {'label': 'Kanyaró megbetegedések', 'value': 'Measles '},
            {'label': 'GDP', 'value': 'GDP'},
            {'label': 'Népesség', 'value': 'Population'},
            {'label': 'Jövedelem összetétele', 'value': 'Income composition of resources'},
            {'label': 'Vékony 1-19 évesek', 'value': ' thinness  1-19 years'},
            {'label': 'Vékony 5-9 évesek', 'value': ' thinness 5-9 years'},
            {'label': 'Teljes egészségügyi kiadás', 'value': 'Total expenditure'},
            {'label': 'Egészségügyre fordított kiadások', 'value': 'percentage expenditure'},
            {'label': 'HIV/AIDS elterjedtség', 'value': ' HIV/AIDS'}
        ],
        value='Life expectancy ',
        clearable=False,
        style={'width': '400px'}
    ),
    dcc.Graph(id='choropleth-map')
], style={'padding': '20px'}),

    html.Hr(),

     html.Div([
         html.H4("Várható élettartam alakulása az egy főre jutó GDP szerint egy választott évben", style={'textAlign': 'left'}),
        html.P("Válassz évet a legördülő listából!"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in years],
            value=years[0],
            clearable=False,
            style={
            'height': '30px',
            'width': '200px',
            'margin': '10px 0 20px 0'
        }
        ),

         html.P("Válassz regressziós modellt:"),
         dcc.Dropdown(
             id='regression-model-dropdown',
             options=[
                 {'label': 'Lineáris', 'value': 'linear'},
                 {'label': 'Polinomiális', 'value': 'polynomial'}
             ],
             value='linear',
             style={
                 'height': '30px',
                 'width': '200px',
                 'margin': '10px 0 20px 0'
             }
         ),

         html.P("Polinomiális fok (csak polinomiális regresszióhoz):"),
         dcc.Input(
             id='polynomial-degree-input',
             type='number',
             value=2,
             min=2,
             max=10,
             style={'width': '100px'}
         ),


        dcc.Graph(id='life-expectancy-gdp-graph')
    ], style={'padding': '20px'}),

    html.Hr(),

    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Készítő adatai', children=[
                dcc.Markdown(keszito_adatai_markdown)
            ],style={'backgroundColor': '#d0d4be'}, selected_style={'backgroundColor': '#E9F0C9'}),
            dcc.Tab(label='Projekt adatai', children=[
                dcc.Markdown(projekt_adatai_markdown)
            ],style={'backgroundColor': '#d0d4be'}, selected_style={'backgroundColor': '#E9F0C9'})
        ])
    ], style={'marginTop': '40px', 'marginBotton': '20px'})
], style={'padding': '30px', 'backgroundColor': '#E9F0C9'})

# Callback 3.feladat
@app.callback(
    Output('gdp-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_gdp_graph(selected_country):
    filtered_data = data[data['Country'] == selected_country]
    if filtered_data.empty:
        fig = px.line(title=f'Nincs elérhető adat a kiválasztott országra: {selected_country}')
        fig.update_layout(
            xaxis_title='Évek',
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='rgba(255, 255, 255, 0.6)',
            xaxis=dict(gridcolor='black', zerolinecolor='black'),
            yaxis=dict(gridcolor='black', zerolinecolor='black')
        )
    else:
        fig = px.line(filtered_data, x='Year', y='GDP', title=f'{selected_country} egy főre jutó GDP-je évenként')
        fig.update_layout(
            xaxis_title='Évek',
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='rgba(255, 255, 255, 0.6)',
            xaxis=dict(gridcolor='black', zerolinecolor='black'),
            yaxis=dict(gridcolor='black', zerolinecolor='black')
        )

    return fig

# Callback 4.feladat - Csak 2015 adataival
@app.callback(
    Output('output-country-list', 'children'),
    Input('population-slider', 'value'),
    Input('gdp-slider', 'value')
)
def update_country_list(population_range, gdp_range):
    filtered_data = data_2015[
        (data_2015['Population'] >= population_range[0]) &
        (data_2015['Population'] <= population_range[1]) &
        (data_2015['GDP'] >= gdp_range[0]) &
        (data_2015['GDP'] <= gdp_range[1])
    ]

    country_list = filtered_data[['Country', 'Status']]

    if country_list.empty:
        return html.Div(
            "Nincs találat a megadott tartományban.",
            style={
                'padding': '20px',
                'backgroundColor': '#E9F0C9',
                'color': 'black',
                'textAlign': 'center',
                'border': '1px solid black',
                'borderRadius': '5px'
            }
        )

    country_elements = []
    row = []
    for index, row_data in country_list.iterrows():
        row.append(
            html.Div(
                [
                    html.Span(row_data['Country'], style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    html.Span(f"- {row_data['Status']}")
                ],
                style={'flex': '1', 'padding': '10px'}
            )
        )
        if len(row) == 4:
            country_elements.append(html.Div(row, style={'display': 'flex', 'justifyContent': 'space-between'}))
            row = []

    if row:
        country_elements.append(html.Div(row, style={'display': 'flex', 'justifyContent': 'space-between'}))

    return html.Div(country_elements)


# Callback 5. feladat
@app.callback(
    Output('life-expectancy-graph', 'figure'),
    Input('country-dropdown-multiple', 'value')
)
def update_life_expectancy_graph(selected_countries):
    if not selected_countries:
        fig = px.line(title='Várható élettartam alakulása', markers=True)
        fig.update_layout(
            xaxis_title='Évek',
            yaxis_title='Várható élettartam (év)',
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='rgba(255, 255, 255, 0.6)',
            xaxis=dict(gridcolor='black', zerolinecolor='black'),
            yaxis=dict(gridcolor='black', zerolinecolor='black')
        )
        return fig

    filtered_data = data[data['Country'].isin(selected_countries)]

    if filtered_data.empty:
        fig = px.line(title='Nincsenek elérhető adatok a kiválasztott országokhoz.', markers=True)
        fig.update_layout(
            xaxis_title='Évek',
            yaxis_title='Várható élettartam (év)',
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='rgba(255, 255, 255, 0.6)',
            xaxis=dict(gridcolor='black',zerolinecolor='black'),
            yaxis=dict(gridcolor='black',zerolinecolor='black')
        )
        return fig

    fig = px.line(
        filtered_data,
        x='Year',
        y='Life expectancy ',
        color='Country',
        title='Várható élettartam alakulása az országokban évek szerint',
        markers=True
    )

    fig.update_layout(
        xaxis_title='Évek',
        yaxis_title='Várható élettartam (év)',
        paper_bgcolor='#E9F0C9',
        plot_bgcolor='rgba(255, 255, 255, 0.6)',
        xaxis=dict(gridcolor='black',zerolinecolor='black'),
        yaxis=dict(gridcolor='black',zerolinecolor='black')
    )

    return fig

# Callback 6.feladat
@app.callback(
    Output('frequency-graph', 'figure'),
    Input('year-slider', 'value'),
    Input('variable-dropdown', 'value'),
    Input('bin-size-slider', 'value')
)
def update_frequency_graph(year_range, selected_variable, bin_size):
    filtered_data_2 = noempty_data[(noempty_data['Year'] >= year_range[0]) & (noempty_data['Year'] <= year_range[1])]
    if filtered_data_2.empty:
        fig = px.histogram(title='Nincs elérhető adat a kiválasztott időszakra.', markers=True)
        fig.update_layout(
            xaxis_title=selected_variable,
            yaxis_title='Gyakoriság',
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='rgba(255, 255, 255, 0.6)',
            xaxis=dict(gridcolor='black'),
            yaxis=dict(gridcolor='black')
        )
        return fig

    fig = px.histogram(filtered_data_2, x=selected_variable, title=f'{selected_variable} gyakoriság az évek alapján', nbins=bin_size)

    fig.update_layout(
        xaxis_title=selected_variable,
        yaxis_title='Gyakoriság',
        paper_bgcolor='#E9F0C9',
        plot_bgcolor='rgba(255, 255, 255, 0.6)',
        xaxis=dict(gridcolor='black'),
        yaxis=dict(gridcolor='black')
    )

    return fig

# Callback 7.feladat
@app.callback(
    Output('choropleth-map', 'figure'),
    Input('variable-dropdown-map', 'value')
)
def update_choropleth(selected_variable):
    filtered_data = data[data[selected_variable].notnull()]
    if filtered_data.empty:
        fig = px.choropleth(
            data_frame=data,
            locations='Country',
            locationmode='country names',
            color_discrete_sequence=["#636EFA"],
            hover_name='Country',
            title=f'Nincsenek elérhető adatok a kiválasztott változóhoz.',
        )

        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='natural earth',
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            paper_bgcolor='#E9F0C9',
            plot_bgcolor='#E9F0C9',
            height=500
        )

        return fig

    fig = px.choropleth(
        data_frame=filtered_data,
        locations='Country',
        locationmode='country names',
        color=selected_variable,
        hover_name='Country',
        animation_frame='Year',
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f'{selected_variable} tematikus térkép évek szerint'
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth',
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor='#E9F0C9',
        plot_bgcolor='#E9F0C9',
        height=500
    )

    return fig


# Callback 8.feladat
@app.callback(
    Output('life-expectancy-gdp-graph', 'figure'),
    [
        Input('year-dropdown', 'value'),
        Input('regression-model-dropdown', 'value'),
        Input('polynomial-degree-input', 'value')
    ]
)
def update_graph(selected_year, selected_model, polynomial_degree):
    filtered_data = filtered_data_reg[filtered_data_reg['Year'] == selected_year]

    x = filtered_data['GDP']
    y = filtered_data['Life expectancy ']

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=10, color='blue'),
        name=f'Year: {selected_year}'
    ))
    if selected_model == 'linear':
        # Lineáris regresszió
        x_reshaped = x.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y)
        y_pred = model.predict(x_reshaped)
        figure.add_trace(go.Scatter(
            x=x,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=2),
            name='Lineáris modell'
        ))

    elif selected_model == 'polynomial':
        if polynomial_degree is None or polynomial_degree < 1:
            polynomial_degree = 2
        coeffs = np.polyfit(x, y, deg=polynomial_degree)
        poly_eq = np.poly1d(coeffs)

        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred = poly_eq(x_pred)

        figure.add_trace(go.Scatter(
            x=x_pred,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'{polynomial_degree}. fokú polinomiális modell'
        ))

    figure.update_layout(
        title=f'Várható élettartam és GDP az évben {selected_year}',
        xaxis_title='GDP',
        yaxis_title='Várható élettartam',
        hovermode='closest',
        paper_bgcolor='#E9F0C9',
        plot_bgcolor='rgba(255, 255, 255, 0.6)',
        xaxis=dict(gridcolor='black', zerolinecolor='black'),
        yaxis=dict(gridcolor='black', zerolinecolor='black')
    )

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
