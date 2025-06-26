import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from custom_transforms.map_image import crop_map_image
from custom_transforms.particle_cloud_image import particles_to_image
from custom_transforms.scan_image import scan_to_image
from dash import Dash, Input, Output, callback, dcc, html
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def extract_position_and_orientation(x: dict) -> tuple[NDArray, NDArray]:
    position = np.array(
        [
            x["/amcl_pose/pose.pose.position.x"],
            x["/amcl_pose/pose.pose.position.y"],
        ],
    )
    orientation = Rotation.from_quat(
        (
            x["/amcl_pose/pose.pose.orientation.x"],
            x["/amcl_pose/pose.pose.orientation.y"],
            x["/amcl_pose/pose.pose.orientation.z"],
            x["/amcl_pose/pose.pose.orientation.w"],
        ),
    ).as_matrix()[:2, :2]
    return (position, orientation)


def compute_map_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    map_image = (
        np.array(x["/map"]["data"])
        .reshape(
            (-1, x["/map"]["info.width"]),
        )
        .astype(np.uint8)
    )
    position, orientation = extract_position_and_orientation(x)
    map_resolution = x["/map"]["info.resolution"]
    map_origin = np.array(
        [
            x["/map"]["info.origin.position.x"],
            x["/map"]["info.origin.position.y"],
        ],
    )

    return crop_map_image(
        map_image,
        robot_position=position,
        robot_orientation=orientation,
        map_resolution=map_resolution,
        map_origin=map_origin,
        width=image_width,
        height=image_height,
        width_meters=width_meters,
    )


def compute_scan_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    distances = np.array(x["/scan/ranges"])
    return scan_to_image(
        distances,
        angle_min=x["/scan/angle_min"],
        angle_increment=x["/scan/angle_increment"],
        width=image_width,
        height=image_height,
        width_meters=width_meters,
        hit_value=1,
        background_value=0,
    )


def compute_particle_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    particles = np.array(
        [
            [
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
                particle["weight"],
            ]
            for particle in x["/particle_cloud/particles"]
        ],
    )
    position, orientation = extract_position_and_orientation(x)
    return particles_to_image(
        particles,
        width=image_width,
        height=image_height,
        width_meters=width_meters,
        robot_position=position,
        robot_orientation=orientation,
    )


data_source = "recordings/rec_20250618_113817.processed.parquet"
data = (
    pl.read_parquet(data_source)
    .with_columns(
        pl.col("/map").struct.with_fields(
            pl.field("data").list.eval(pl.element() != 0),
        ),
    )
    .explode("measurements")
    .unnest("measurements")
    .unnest("value")
)
n_samples = len(data)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__,
    external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
)

app.layout = [
    html.H1(
        children="Inspecting Robot Localization Data",
        style={"textAlign": "center"},
    ),
    dcc.Input(
        id="sample_i",
        value=0,
        type="number",
        min=0,
        max=n_samples,
        step=1,
    ),
    dcc.Input(
        id="image_width",
        value=100,
        type="number",
        min=2,
        max=1000,
    ),
    dcc.Input(
        id="image_height",
        value=100,
        type="number",
        min=2,
        max=1000,
    ),
    dcc.Input(
        id="width_meters",
        value=15,
        type="number",
        min=0.1,
        max=100,
        step=0.1,
    ),
    html.Div(
        children=[
            html.Div(
                children=[dcc.Graph(id="full_map_image")],
                className="three columns",
            ),
            html.Div(
                children=[dcc.Graph(id="map_image")],
                className="three columns",
            ),
            html.Div(
                children=[dcc.Graph(id="scan_image")],
                className="three columns",
            ),
            html.Div(
                children=[dcc.Graph(id="particle_image")],
                className="three columns",
            ),
        ],
        className="row",
    ),
    dcc.Graph(id="position_error"),
]


@callback(
    Output("position_error", "figure"),
    Input("sample_i", "value"),
)
def update_line(
    sample_i: int,
) -> go.Figure:
    fig = px.line(data, x="time", y="position_error")
    fig.add_vline(
        x=data["time"][sample_i],
        line_width=3,
        line_dash="dash",
        line_color="green",
    )
    fig.update_layout(showlegend=False)
    return fig


@callback(
    Output("full_map_image", "figure"),
    Input("sample_i", "value"),
)
def update_full_map(
    sample_i: int,
) -> go.Figure:
    map_data = data.select(
        pl.col("/map").struct.field("data", "info.width", "info.resolution"),
    ).row(
        sample_i,
        named=True,
    )
    map_resolution = map_data["info.resolution"]
    amcl_position = data.select(
        pl.col("/amcl_pose/pose.pose.position.x"),
        pl.col("/amcl_pose/pose.pose.position.y"),
    ).row(sample_i)
    momo_position = data.select(
        pl.col("/momo/pose/pose.position.x"),
        pl.col("/momo/pose/pose.position.y"),
    ).row(sample_i)
    rotation = data.select(
        pl.col("/amcl_pose/pose.pose.orientation.x"),
        pl.col("/amcl_pose/pose.pose.orientation.y"),
        pl.col("/amcl_pose/pose.pose.orientation.z"),
        pl.col("/amcl_pose/pose.pose.orientation.w"),
    ).row(sample_i)
    rotation = Rotation.from_quat(rotation).as_matrix()[:2, :2]
    map_origin = data.select(
        pl.col("/map").struct.field(
            "info.origin.position.x",
            "info.origin.position.y",
        ),
    ).row(sample_i)
    fig = px.imshow(
        np.array(map_data["data"]).reshape((-1, map_data["info.width"])),
        origin="lower",
    ).update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.add_trace(
        go.Scatter(
            x=[(amcl_position[0] - map_origin[0]) / map_resolution],
            y=[(amcl_position[1] - map_origin[1]) / map_resolution],
            marker={"color": "blue", "size": 16},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[(momo_position[0] - map_origin[0]) / map_resolution],
            y=[(momo_position[1] - map_origin[1]) / map_resolution],
            marker={"color": "red", "size": 16},
        ),
    )
    # show_heading
    fig.add_trace(
        go.Scatter(
            x=[
                (amcl_position[0] - map_origin[0]) / map_resolution,
                (amcl_position[0] - map_origin[0]) / map_resolution
                + rotation[0, 0] / map_resolution,
            ],
            y=[
                (amcl_position[1] - map_origin[1]) / map_resolution,
                (amcl_position[1] - map_origin[1]) / map_resolution
                + rotation[1, 0] / map_resolution,
            ],
            mode="lines",
            line={"color": "blue", "width": 2},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[
                (momo_position[0] - map_origin[0]) / map_resolution,
                (momo_position[0] - map_origin[0]) / map_resolution
                + rotation[0, 0] / map_resolution,
            ],
            y=[
                (momo_position[1] - map_origin[1]) / map_resolution,
                (momo_position[1] - map_origin[1]) / map_resolution
                + rotation[1, 0] / map_resolution,
            ],
            mode="lines",
            line={"color": "red", "width": 2},
        ),
    )
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig


@callback(
    Output("map_image", "figure"),
    Input("sample_i", "value"),
    Input("image_width", "value"),
    Input("image_height", "value"),
    Input("width_meters", "value"),
)
def update_map(
    sample_i: int,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> go.Figure:
    fig = px.imshow(
        compute_map_image(
            data.row(sample_i, named=True),
            image_width=image_width,
            image_height=image_height,
            width_meters=width_meters,
        ),
        origin="lower",
    ).update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig


@callback(
    Output("scan_image", "figure"),
    Input("sample_i", "value"),
    Input("image_width", "value"),
    Input("image_height", "value"),
    Input("width_meters", "value"),
)
def update_scan(
    sample_i: int,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> go.Figure:
    fig = px.imshow(
        compute_scan_image(
            data.row(sample_i, named=True),
            image_width=image_width,
            image_height=image_height,
            width_meters=width_meters,
        ),
        origin="lower",
    ).update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig


@callback(
    Output("particle_image", "figure"),
    Input("sample_i", "value"),
    Input("image_width", "value"),
    Input("image_height", "value"),
    Input("width_meters", "value"),
)
def update_particles(
    sample_i: int,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> go.Figure:
    fig = px.imshow(
        compute_particle_image(
            data.row(sample_i, named=True),
            image_width=image_width,
            image_height=image_height,
            width_meters=width_meters,
        ),
        origin="lower",
    ).update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig


app.run(debug=True)
