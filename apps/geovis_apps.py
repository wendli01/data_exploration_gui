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


def get_shapes_heatmap(data, nuts_ids_column, color_column, logarithmic: bool = False, cmap='viridis',
                       info_columns=('NUTS_ID', 'NUTS_NAME', 'num_persons'), info_widget_html=None,
                       vmin=0, vmax=1, time_hist=None, full_data=None, date_limits=None, tweets_box=None,
                       tweet_info_columns=('_timestamp', 'text_translated', 'num_persons', 'mode')):
    def get_layer(shapes: gpd.GeoDataFrame, color):
        def get_info_text():
            return '<h4>{}</h4>'.format(nuts_ids_column) + '<br>'.join(
                [str(shapes[col].values[0]) for col in info_columns])

        def hover_event_handler(**kwargs):
            info_widget_html.value = get_info_text()

            out = interactive_output(plot_time_hist,
                                     dict(data=fixed(relevant_data), timestamp_column=fixed('_timestamp'),
                                          value_column=fixed(color_column), xlims=fixed(date_limits)))
            time_hist.children = [out]

        def click_event_handler(**kwargs):
            if kwargs['event'] != 'click':
                return
            if tweets_box.placeholder == nuts_id:
                tweets_box.placeholder, tweets_box.value = '', ''
            elif len(relevant_data) > 0:
                tweets_box.value = relevant_data[list(tweet_info_columns)].fillna('').sort_values(
                    tweet_info_columns[0]).to_html()
                tweets_box.placeholder = nuts_id

        nuts_id = shapes[nuts_ids_column].values[0]
        style = {'color': color, 'fillColor': color, 'opacity': 0.5, 'weight': 1.9, 'dashArray': '2',
                 'fillOpacity': 0.2}
        hover_style = {'fillColor': 'blue', 'fillOpacity': 0.2}
        layer = ipyleaflet.GeoData(geo_dataframe=shapes, style=style, hover_style=hover_style)
        if full_data is None or tweets_box is None or time_hist is None:
            return layer
        relevant_data = full_data[full_data[nuts_ids_column].str.startswith(nuts_id, na=False)].reset_index()
        if time_hist is not None and info_widget_html is not None:
            layer.on_hover(hover_event_handler)
        if tweets_box is not None:
            layer.on_click(click_event_handler)
        return layer

    def get_layer_group(shapes: gpd.GeoDataFrame, colors, group_name='', sorting_column='LEVL_CODE'):
        if sorting_column in shapes:
            sorting = shapes[sorting_column].argsort()
            shapes = shapes.iloc[sorting]
            colors = colors.iloc[sorting]
        layers = [get_layer(shapes.iloc[[i]], color=color) for i, color in enumerate(colors)]
        return ipyleaflet.LayerGroup(layers=layers, name=group_name)

    def get_colors(values: pd.Series, logarithmic: bool = False, cmap='viridis'):
        values: pd.Series = values + 1e-5
        norm_class = matplotlib.colors.LogNorm if logarithmic else matplotlib.colors.Normalize
        norm = norm_class(vmin=vmin, vmax=vmax)
        cm = matplotlib.cm.get_cmap(cmap)
        return values.apply(norm).apply(cm).apply(matplotlib.colors.to_hex)

    colors = get_colors(data[color_column], logarithmic=logarithmic, cmap=cmap)
    return get_layer_group(gpd.GeoDataFrame(data), colors=colors, group_name=nuts_ids_column)


def merge_df(data, nuts_shapes, nuts_ids_column, color_column, level='all'):
    def aggregate_all(df, levels=(0, 1, 2, 3), **kwargs):
        return pd.concat([aggregate_nuts_level(df, level=level, **kwargs) for level in levels])

    def aggregate_nuts_level(df, level, nuts_ids_column='NUTS_ID', aggregatable_columns=('num_persons',),
                             aggregation=np.sum):
        df = df.dropna(subset=aggregatable_columns)
        df[nuts_ids_column] = df[nuts_ids_column].str.slice(0, level + 2)
        agg_df = aggregation(df.groupby(nuts_ids_column)[aggregatable_columns]).reset_index()
        return agg_df[agg_df[nuts_ids_column].apply(len) == level + 2]

    agg_data = data.dropna(subset=[color_column]).groupby(nuts_ids_column).sum().reset_index()
    if level != 'all':
        agg_data = aggregate_nuts_level(agg_data, level=level, aggregatable_columns=[color_column],
                                        nuts_ids_column=nuts_ids_column)
    else:
        agg_data = aggregate_all(agg_data, aggregatable_columns=[color_column], nuts_ids_column=nuts_ids_column)

    merged_df = pd.merge(agg_data, nuts_shapes, left_on=nuts_ids_column, right_on='NUTS_ID')
    return merged_df.dropna(subset=['NUTS_ID', color_column])


def plot_cbar(name, vmin=0, vmax=1, logarithmic=False):
    fig, ax = plt.subplots(figsize=(.3, 10))
    norm = matplotlib.colors.LogNorm(vmin, vmax) if logarithmic else matplotlib.colors.Normalize(vmin, vmax)
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(name), norm=norm, orientation='vertical')
    return cbar


def plot_time_hist(data, timestamp_column, value_column, xlims):
    df = data.dropna(subset=[value_column, timestamp_column])
    if len(df) == 0:
        return
    dates = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
    time_hist = df.groupby(dates).sum()[value_column].reset_index()
    time_hist['zero'] = 0
    plt.box(False), plt.xlim(*xlims)
    if len(time_hist) > 1:
        plot = plt.plot(time_hist[timestamp_column], time_hist[value_column], label=value_column, marker='o')
        plt.fill_between(time_hist[timestamp_column], time_hist[value_column], time_hist['zero'], alpha=.4)
    else:
        plot = plt.scatter(time_hist[timestamp_column], time_hist[value_column], label=value_column)
        plt.ylim(0, 1.05 * time_hist[value_column].max())
    plt.legend(), plt.xlabel(''), plt.xticks(rotation=45), plt.tight_layout()
    return plot


def plot_geo_shapes_vis(data, nuts_shapes, date_range, nuts_ids_columns=('origin', 'destination'),
                        color_column='num_persons', logarithmic=False, cmap='viridis', timestamp_column='_timestamp'):
    def date_filter(data, date_range):
        dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
        return data[(date_range[0] <= dates) & (dates <= date_range[1])]

    def show_level_layers(change):
        def change_layers(layer_group: ipyleaflet.LayerGroup, all_layers):
            new_layers = [l for l in all_layers if l.data['features'][0]['properties']['LEVL_CODE'] == new_level]
            layer_group.layers = new_layers if new_level != 'all' else all_layers

        layer_groups = [l for l in m.layers if type(l) is ipyleaflet.LayerGroup]
        if len(full_groups) == 0:
            full_groups.extend([l.layers for l in layer_groups])
        new_level = change['new']
        for layer_group, full_group in zip(layer_groups, full_groups):
            change_layers(layer_group, full_group)

    data = date_filter(data, date_range).dropna(subset=nuts_ids_columns, how='all')
    full_data = data.copy()
    full_groups = []
    m = ipyleaflet.Map(center=(51, 10), zoom=4, scroll_wheel_zoom=True)
    m.layout.height = '800px'

    merged_dfs = [merge_df(data=data, nuts_shapes=nuts_shapes, nuts_ids_column=nuts_ids_column,
                           color_column=color_column, level=level) for nuts_ids_column in nuts_ids_columns]
    vmin, vmax = 1, np.max([df[color_column].max() for df in merged_dfs])

    country_widget_html = widgets.HTML('''Hover over a Region<br>Click it to see tweets''')
    country_widget_html.layout.margin = '0px 20px 20px 20px'
    country_widget = ipyleaflet.WidgetControl(widget=country_widget_html, position='topright')
    m.add_control(country_widget)

    cbar_widget_box = interactive_output(plot_cbar, dict(name=fixed(cmap), vmin=fixed(vmin), vmax=fixed(vmax),
                                                         logarithmic=fixed(logarithmic)))
    cbar_widget = ipyleaflet.WidgetControl(widget=cbar_widget_box, position='bottomright')
    m.add_control(cbar_widget)

    time_hist = widgets.HBox([])
    time_hist_widget = ipyleaflet.WidgetControl(widget=time_hist, position='bottomleft')
    m.add_control(time_hist_widget)

    tweets_table = widgets.HTML(layout=widgets.Layout(overflow='scroll_hidden'))
    tweets_box = widgets.HBox([tweets_table], layout=Layout(max_height='400px', overflow_y='auto'))
    tweets_box_widget = ipyleaflet.WidgetControl(widget=tweets_box, position='bottomleft')
    m.add_control(tweets_box_widget)

    level_selector = widgets.Dropdown(options=['all', *sorted(nuts_shapes['LEVL_CODE'].unique())],
                                      description='NUTS levels')

    level_selector.observe(handler=show_level_layers, type='change', names=('value',))
    level_widget = ipyleaflet.WidgetControl(widget=level_selector, position='topleft')
    m.add_control(level_widget)

    m.add_control(ipyleaflet.LayersControl())
    m.add_control(ipyleaflet.FullScreenControl())

    for merged_df, nuts_ids_column in zip(merged_dfs, nuts_ids_columns):
        layer = get_shapes_heatmap(data=merged_df, nuts_ids_column=nuts_ids_column, color_column=color_column,
                                   logarithmic=logarithmic, cmap=cmap, info_widget_html=country_widget_html, vmin=vmin,
                                   vmax=vmax, full_data=full_data, time_hist=time_hist, date_limits=date_range,
                                   tweets_box=tweets_table)
        m.add_layer(layer)

    display(m)


def geo_vis_shapes_app(data, simplify_nuts_shapes=True):
    nuts_ids_columns = ['origin', 'destination']
    nuts_shapes = get_nuts_shapes(simplify=simplify_nuts_shapes, tol=1e-3)

    cmap = widgets.Dropdown(options=['viridis', 'inferno', 'magma', 'winter', 'cool'], description='colormap')
    logarithmic = widgets.Checkbox(description='logarithmic')
    time_slider = get_time_slider(data)
    controls = widgets.VBox([widgets.HBox([cmap, logarithmic]), time_slider])

    geo_vis = interactive_output(plot_geo_shapes_vis,
                                 dict(nuts_shapes=fixed(nuts_shapes), data=fixed(data), logarithmic=logarithmic,
                                      cmap=cmap, nuts_ids_columns=fixed(nuts_ids_columns), date_range=time_slider))

    display(controls, geo_vis)


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
    m = ipyleaflet.Map(center=(51, 10), zoom=4, scroll_wheel_zoom=True)
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
