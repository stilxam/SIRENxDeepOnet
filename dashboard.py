import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from preprocessing import engineer_features, get_df
from pathlib import Path

app = dash.Dash(__name__)

# Initial load with defaults
def load_features(time_group="1min", rolling_window=5, num_files=1):
    df = get_df(Path("data"), num_files)
    features = engineer_features(df, interval=time_group, rolling_window=rolling_window)
    return features

# Get feature list for dropdown
features_df = load_features()
feature_options = [
    {"label": col, "value": col} for col in features_df.columns
]

def serve_layout():
    return html.Div([
        html.H1("Feature Engineering Dashboard"),
        html.Div([
            html.Label("Select Features to Plot:"),
            dcc.Dropdown(
                id="feature-dropdown",
                options=feature_options,
                value=["price_vwap"],
                multi=True
            ),
        ]),
        html.Div([
            html.Label("Time Interval (e.g. 1min, 5min, 1H):"),
            dcc.Input(id="interval-input", type="text", value="1min"),
            html.Label("Rolling Window Size:"),
            dcc.Input(id="rolling-window-input", type="number", value=5),
            html.Label("Number of Files to Load:"),
            dcc.Input(id="num-files-input", type="number", value=1),
            html.Button("Update", id="update-btn", n_clicks=0)
        ], style={"marginTop": 20, "marginBottom": 20}),
        dcc.Loading([
            dcc.Graph(id="feature-graph")
        ])
    ])

app.layout = serve_layout

@app.callback(
    Output("feature-graph", "figure"),
    Input("feature-dropdown", "value"),
    Input("interval-input", "value"),
    Input("rolling-window-input", "value"),
    Input("num-files-input", "value"),
    Input("update-btn", "n_clicks")
)
def update_graph(selected_features, interval, rolling_window, num_files, n_clicks):
    features = load_features(time_group=interval, rolling_window=int(rolling_window), num_files=int(num_files))
    data = []
    for feat in selected_features:
        if feat in features.columns:
            data.append(go.Scatter(x=features.index, y=features[feat], mode="lines", name=feat))
    fig = go.Figure(data=data)
    fig.update_layout(title="Selected Features Over Time", xaxis_title="Time", yaxis_title="Value")
    return fig

if __name__ == "__main__":
    app.run(debug=True)

