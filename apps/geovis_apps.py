import os

import geopandas as gpd
import ipyleaflet
import ipywidgets as widgets
import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import shapely
from IPython.display import display
from ipywidgets import fixed, Layout, interactive_output
from matplotlib import pyplot as plt


def get_time_slider(df, timestamp_column='_timestamp'):
    def get_date_range():
        timestamps = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
        vmin, vmax = timestamps.min(), timestamps.max()
        return pd.Series(pd.date_range(vmin, vmax)).apply(pd.Timestamp.date)

    date_range = get_date_range()
    return widgets.SelectionRangeSlider(options=date_range.values, description='Date Range', continuous_update=False,
                                        index=(0, len(date_range) - 1), values=(0, len(date_range)),
                                        layout=Layout(width='500px'))


def get_nuts_shapes(shp_folder='nuts_data', simplify=False, tol=1e-3):
    def get_fns(directory, condition=lambda x: True):
        return list(filter(condition, [directory + '/' + fn for fn in os.listdir(directory)]))

    folders = get_fns(shp_folder, os.path.isdir)
    files = np.hstack([get_fns(folder, lambda f: f.endswith('.shp')) for folder in folders])
    geo_df = pd.concat(list(map(gpd.read_file, files)))
    if simplify:
        geo_df.geometry = geo_df.geometry.simplify(tol)
    return geo_df


def get_shapes_heatmap(data, nuts_ids_column, color_column, m, logarithmic: bool = False, cmap='viridis',
                       title_columns=('NUTS_NAME', 'num_persons')):
    def get_layer(shapes: gpd.GeoDataFrame, color):
        def get_event_handler(location, text):
            def layer_event_handler(event, **kwargs):
                if event == 'click':
                    m.add_layer(ipyleaflet.Popup(location=location, child=widgets.HTML(text)))

            return layer_event_handler

        style = {'color': color, 'fillColor': color, 'opacity': 0.5, 'weight': 1.9, 'dashArray': '2',
                 'fillOpacity': 0.2}
        hover_style = {'fillColor': 'blue', 'fillOpacity': 0.2}
        layer = ipyleaflet.GeoData(geo_dataframe=shapes, style=style, hover_style=hover_style)
        text = nuts_ids_column + '<br>' + '<br>'.join([str(shapes[k].values[0]) for k in title_columns])
        location = np.mean([shapes.geometry.centroid.y.values, shapes.geometry.centroid.x.values], 1)
        layer.on_click(get_event_handler(list(location), text))
        return layer

    def get_layer_group(shapes: gpd.GeoDataFrame, colors, group_name='', sorting_column='LEVL_CODE'):
        if sorting_column in shapes:
            shapes = shapes.sort_values(sorting_column)
        layers = [get_layer(shapes.iloc[[i]], color=color) for i, color in enumerate(colors)]
        return ipyleaflet.LayerGroup(layers=layers, name=group_name)

    def get_colors(values: pd.Series, logarithmic: bool = False, cmap='viridis'):
        values: pd.Series = values + 1e-5
        norm_class = matplotlib.colors.LogNorm if logarithmic else matplotlib.colors.Normalize
        norm = norm_class(vmin=values.min(), vmax=values.max())
        cm = matplotlib.cm.get_cmap(cmap)
        return values.apply(norm).apply(cm).apply(matplotlib.colors.to_hex)

    colors = get_colors(data[color_column], logarithmic=logarithmic, cmap=cmap)
    return get_layer_group(gpd.GeoDataFrame(data), colors=colors, group_name=nuts_ids_column)


def merge_df(data, nuts_shapes, nuts_ids_column, color_column, level='all'):
    def aggregate_all(df, levels=(0, 1, 2, 3), **kwargs):
        return pd.concat([aggregate_nuts_level(df, level=level, **kwargs) for level in levels])

    def aggregate_nuts_level(df, level, nuts_ids_column='NUTS_ID', aggregatable_columns=('num_persons',),
                             aggregation=np.sum):
        df = df.copy()
        df[nuts_ids_column] = df[nuts_ids_column].str.slice(0, level + 2)
        agg_df = aggregation(df.groupby(nuts_ids_column)[aggregatable_columns]).reset_index()
        return agg_df[agg_df[nuts_ids_column].apply(len) == level + 2]

    agg_data = data.groupby(nuts_ids_column).sum().reset_index()
    if level != 'all':
        agg_data = aggregate_nuts_level(agg_data, level=level, aggregatable_columns=[color_column],
                                        nuts_ids_column=nuts_ids_column)
    else:
        agg_data = aggregate_all(agg_data, aggregatable_columns=[color_column], nuts_ids_column=nuts_ids_column)

    merged_df = pd.merge(agg_data, nuts_shapes, left_on=nuts_ids_column, right_on='NUTS_ID')
    return merged_df.dropna(subset=['NUTS_ID', color_column])


def plot_cbar(name, vmin=0, vmax=1, logarithmic=False, n=100):
    fig, ax = plt.subplots(figsize=(.3, 14))
    norm = matplotlib.colors.LogNorm(vmin, vmax) if logarithmic else matplotlib.colors.Normalize(vmin, vmax)
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(name), norm=norm, orientation='vertical')
    return cbar


def plot_geo_data_shapes(data, nuts_shapes, date_range, nuts_ids_columns=('origin', 'destination'),
                         color_column='num_persons', logarithmic=False, cmap='viridis', timestamp_column='_timestamp',
                         level='all'):
    def get_geo_data(data, nuts_ids_column, m):
        merged_df = merge_df(data=data, nuts_shapes=nuts_shapes, nuts_ids_column=nuts_ids_column,
                             color_column=color_column,
                             level=level)
        return get_shapes_heatmap(merged_df, nuts_ids_column=nuts_ids_column, color_column=color_column,
                                  logarithmic=logarithmic, cmap=cmap, m=m)

    def date_filter(data, date_range):
        dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
        return data[(date_range[0] <= dates) & (dates <= date_range[1])]

    data = date_filter(data, date_range)
    m = ipyleaflet.Map(center=(51, 10), zoom=4)
    m.layout.height = '800px'
    for nuts_ids_column in nuts_ids_columns:
        layer = get_geo_data(data, nuts_ids_column, m)
        m.add_layer(layer)
    m.add_control(ipyleaflet.LayersControl())
    m.add_control(ipyleaflet.FullScreenControl())
    display(m)


def geo_vis_shapes_app(data):
    def get_cbar(name, logarithmic):
        color_vals = []
        for nuts_ids_column in nuts_ids_columns:
            color_vals.append(merge_df(data, nuts_shapes, 'origin', color_column)[color_column])
        vmin, vmax = max(np.min(color_vals), 1), np.max(color_vals)
        plot_cbar(name, vmin, vmax, logarithmic=logarithmic)

    nuts_ids_columns = ['origin', 'destination']
    color_column = 'num_persons'
    nuts_shapes = get_nuts_shapes(simplify=True, tol=1e-3)
    avail_levels = sorted(nuts_shapes['LEVL_CODE'].unique())

    level = widgets.Dropdown(options=['all', *avail_levels], description='NUTS levels')
    cmap = widgets.Dropdown(options=['viridis', 'inferno', 'magma', 'winter', 'cool'], description='colormap')
    logarithmic = widgets.Checkbox(description='logarithmic')
    time_slider = get_time_slider(data)
    controls = widgets.VBox([widgets.HBox([level, cmap, logarithmic]), time_slider])
    cbar = interactive_output(get_cbar, dict(name=cmap, logarithmic=logarithmic))

    geo_vis = interactive_output(plot_geo_data_shapes,
                                 dict(nuts_shapes=fixed(nuts_shapes), data=fixed(data), logarithmic=logarithmic,
                                      cmap=cmap, nuts_ids_columns=fixed(nuts_ids_columns), level=level,
                                      date_range=time_slider))

    geo_vis.layout.width = '90%'
    geo_vis_box = widgets.HBox([geo_vis, cbar])
    display(controls, geo_vis_box)


def get_marker_cluster(data, geom_column, title_columns=('text_translated', '_timestamp')):
    def wkb_hex_to_point(s):
        return list(shapely.wkb.loads(s, hex=True).coords)[0][::-1]

    def get_title(d):
        return '\n'.join([str(d[c]) for c in title_columns if d[c] not in (np.nan, None)])

    data = data.dropna(subset=[geom_column])
    locs = data[geom_column].apply(wkb_hex_to_point)
    dicts = data.to_dict(orient='rows')

    markers = [ipyleaflet.Marker(location=loc, title=get_title(d), draggable=False) for loc, d in zip(locs, dicts)]
    return ipyleaflet.MarkerCluster(markers=markers)


def plot_geo_data_cluster(data, geom_column, timestamp_column, date_range, title_columns):
    def date_filter(data, date_range):
        dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
        return data[(date_range[0] <= dates) & (dates <= date_range[1])]

    data = date_filter(data, date_range)
    m = ipyleaflet.Map(center=(51, 10), zoom=4)
    m.layout.height = '800px'
    m.add_layer(get_marker_cluster(data, geom_column, title_columns=title_columns))
    m.add_control(ipyleaflet.FullScreenControl())
    display(m)


def geo_vis_cluster_app(data, timestamp_column='_timestamp', geom_column='geom_tweet'):
    time_slider = get_time_slider(data)
    title_columns = widgets.SelectMultiple(options=sorted(data.columns), description='Information to show',
                                           value=['text_translated', '_timestamp'])
    title_columns_tip = widgets.HTML('Select multiple by dragging or ctrl + click <br> Deselect with ctrl + click')
    title_columns_controls = widgets.HBox([title_columns, title_columns_tip])

    geo_vis = interactive_output(plot_geo_data_cluster,
                                 dict(data=fixed(data), date_range=time_slider, geom_column=fixed(geom_column),
                                      timestamp_column=fixed(timestamp_column), title_columns=title_columns))

    geo_vis.layout.width = '90%'
    controls = widgets.Tab([time_slider, title_columns_controls])
    controls.set_title(0, 'Date Range')
    controls.set_title(1, 'Information to Show')
    display(controls, geo_vis)
