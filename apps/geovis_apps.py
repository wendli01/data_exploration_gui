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
                       vmin=0, vmax=1, time_hist=None, full_data=None, date_limits=None, tweets_table=None,
                       table_columns=('_timestamp', 'text_translated', 'num_persons', 'mode')):
    def get_layer(shapes: gpd.GeoDataFrame, color):
        def get_info_text():
            return '<h4>{}</h4>'.format(nuts_ids_column) + '<br>'.join(
                [str(shapes[col].values[0]) for col in info_columns if col in shapes.columns])

        def hover_event_handler(**kwargs):
            info_widget_html.value = get_info_text()
            out = interactive_output(plot_time_hist,
                                     dict(data=fixed(relevant_data), timestamp_column=fixed('_timestamp'),
                                          value_column=fixed(color_column), xlims=fixed(date_limits)))
            time_hist.layout.margin = '5px'
            time_hist.children = [out]

        def click_event_handler(**kwargs):
            if kwargs['event'] != 'click':
                return
            if tweets_table.placeholder == nuts_id:
                tweets_table.placeholder, tweets_table.value = '', ''
            elif len(relevant_data) > 0:
                header = '<b>{} {}</b>'.format(nuts_ids_column, nuts_id)
                tweets_data = relevant_data[[c for c in table_columns if c in relevant_data.columns]].sort_values(
                    table_columns[0]).dropna(axis='columns', how='all')
                tweets_table.value = header + tweets_data.to_html(na_rep='', index=False)
                tweets_table.layout.margin = '5px'
                tweets_table.placeholder = nuts_id

        nuts_id = shapes[nuts_ids_column].values[0]
        style = {'color': color, 'fillColor': color, 'opacity': 0.5, 'weight': 1.9, 'dashArray': '2',
                 'fillOpacity': 0.2}
        hover_style = {'fillColor': color, 'fillOpacity': 0.5, 'weight': 5}
        layer = ipyleaflet.GeoData(geo_dataframe=shapes, style=style, hover_style=hover_style)
        if full_data is None or tweets_table is None or time_hist is None:
            return layer
        relevant_data = full_data[full_data[nuts_ids_column].str.startswith(nuts_id, na=False)].reset_index()
        if time_hist is not None and info_widget_html is not None:
            layer.on_hover(hover_event_handler)
        if tweets_table is not None:
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
        get_single = lambda v: get_color(v, logarithmic, cmap, vmin, vmax)
        return values.apply(get_single)

    colors = get_colors(data[color_column], logarithmic=logarithmic, cmap=cmap)
    return get_layer_group(gpd.GeoDataFrame(data), colors=colors, group_name=nuts_ids_column)


def get_color(val: float, logarithmic: bool = False, cmap='viridis', vmin=0, vmax=1):
    norm_class = matplotlib.colors.LogNorm if logarithmic else matplotlib.colors.Normalize
    norm = norm_class(vmin=vmin, vmax=vmax)
    cm = matplotlib.cm.get_cmap(cmap)
    return matplotlib.colors.to_hex(cm(norm(val + 1e-5)))


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


def plot_geo_shapes_vis(data, nuts_shapes, nuts_ids_columns=('origin', 'destination'),
                        color_column='num_persons', timestamp_column='_timestamp'):
    def plot_cbar(name, logarithmic=False):
        vmin, vmax = app_state['vmin'], app_state['vmax']
        if vmin == vmax or any(pd.isna([vmin, vmax])):
            return
        fig, ax = plt.subplots(figsize=(.3, 10))
        norm = matplotlib.colors.LogNorm(vmin, vmax) if logarithmic else matplotlib.colors.Normalize(vmin, vmax)
        cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(name), norm=norm, orientation='vertical')
        return cbar

    def date_filter(data, date_range):
        dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
        return data[(date_range[0] <= dates) & (dates <= date_range[1])]

    def change_date_range(change):
        data = date_filter(app_state['data'], change['new']).dropna(subset=nuts_ids_columns, how='all')
        merged_dfs = [merge_df(data=data, nuts_shapes=nuts_shapes, nuts_ids_column=nuts_ids_column,
                               color_column=color_column, level=level_selector.value) for nuts_ids_column in
                      nuts_ids_columns]
        app_state['vmax'] = np.max([df[color_column].max() for df in merged_dfs])
        interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))

        m.layers = [l for l in m.layers if type(l) != ipyleaflet.LayerGroup]
        for merged_df, nuts_ids_column in zip(merged_dfs, nuts_ids_columns):
            table_columns = ['_timestamp', 'text_translated', 'num_persons', 'mode',
                             *[col for col in nuts_ids_columns if col != nuts_ids_column]]
            layer = get_shapes_heatmap(data=merged_df, nuts_ids_column=nuts_ids_column, color_column=color_column,
                                       info_widget_html=country_widget, vmin=app_state['vmin'], vmax=app_state['vmax'],
                                       full_data=app_state['full_data'], time_hist=time_hist, date_limits=change['new'],
                                       tweets_table=tweets_table, cmap=app_state['cmap'], table_columns=table_columns,
                                       logarithmic=app_state['logarithmic'])
            m.add_layer(layer)

        if 'full_groups' not in app_state:
            app_state['full_groups'] = [l.layers for l in m.layers if type(l) is ipyleaflet.LayerGroup]

        cbar_widget.children = [interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))]

    def change_level_layers(change={}):
        def change_layers(layer_group: ipyleaflet.LayerGroup, all_layers: list):
            new_layers = [l for l in all_layers if
                          l.data['features'][0]['properties']['LEVL_CODE'] == app_state['level']]
            layer_group.layers = new_layers if app_state['level'] != 'all' else all_layers

        layer_groups = [l for l in m.layers if type(l) is ipyleaflet.LayerGroup]
        if 'new' in change:
            app_state['level'] = change['new']
        for layer_group, full_group in zip(layer_groups, app_state['full_groups']):
            change_layers(layer_group, full_group)

    def change_colormap():
        cmap, logarithmic = app_state['cmap'], app_state['logarithmic']

        def update_layer(l):
            val = l.data['features'][0]['properties'][color_column]
            color = get_color(val, cmap=cmap, vmin=app_state['vmin'], vmax=app_state['vmax'], logarithmic=logarithmic)
            l.style.update({'fillColor': color, 'color': color})
            l.hover_style.update({'fillColor': color, 'color': color})
            new_layer = ipyleaflet.GeoJSON(data=l.data, style=l.style, hover_style=l.hover_style)
            new_layer._hover_callbacks = l._hover_callbacks
            new_layer._click_callbacks = l._click_callbacks
            return new_layer

        app_state['full_groups'] = [[update_layer(l) for l in layers] for layers in app_state['full_groups']]

        change_level_layers()

    def change_colormap_name(change):
        app_state['cmap'] = change['new']
        change_colormap()

    def change_colormap_log(change):
        app_state['logarithmic'] = change['new']
        change_colormap()

    def add_widget(widget, pos, margin='0px 0px 0px 0px'):
        widget.layout.margin = margin
        widget_control = ipyleaflet.WidgetControl(widget=widget, position=pos)
        m.add_control(widget_control)

    def loading_wrapper(function):
        def loading_func(*args, **kwargs):
            loading = ipyleaflet.WidgetControl(widget=widgets.HTML('loading...', layout=Layout(margin='10px')),
                                               position='bottomright')
            m.add_control(loading)
            function(*args, **kwargs)
            m.remove_control(loading)

        return loading_func

    def on_zoom(change, max_level=3, min_level=0, offset=-5):
        if m.zoom != app_state['zoom']:
            app_state['zoom'] = m.zoom
            if level_on_zoom.value:
                level = min(max_level, max(min_level, m.zoom + offset))
                app_state['level'] = level
                level_selector.value = level

    app_state = dict(zoom=4, data=data.dropna(subset=nuts_ids_columns, how='all'), cmap='viridis', logarithmic=False,
                     vmin=1, vmax=1, full_data=data.copy(), level='all')
    m = ipyleaflet.Map(center=(51, 10), zoom=app_state['zoom'], scroll_wheel_zoom=True, zoom_control=False)
    m.layout.height = '800px'

    country_widget = widgets.HTML('''Hover over a Region<br>Click it to see tweets''')
    add_widget(country_widget, pos='topright', margin='10px')

    time_hist = widgets.HBox([])
    add_widget(time_hist, pos='bottomleft')

    tweets_table = widgets.HTML(layout=widgets.Layout(overflow='scroll_hidden'))
    tweets_box = widgets.HBox([tweets_table], layout=Layout(max_height='400px', overflow_y='auto', max_width='900px'))
    add_widget(tweets_box, pos='bottomleft')

    time_slider = get_time_slider(app_state['data'])
    time_slider.observe(loading_wrapper(change_date_range), type='change', names=('value',))
    add_widget(time_slider, 'topleft', margin='5px')

    level_selector = widgets.Dropdown(options=['all', *sorted(nuts_shapes['LEVL_CODE'].unique())],
                                      description='NUTS levels', layout=Layout(max_width='180px'))
    level_selector.observe(handler=loading_wrapper(change_level_layers), type='change', names=('value',))
    level_on_zoom = widgets.Checkbox(value=True, description='with zoom', layout=Layout(max_width='180px'))
    level_control = widgets.VBox([level_selector, level_on_zoom])

    cmap_selector = widgets.Dropdown(options=['viridis', 'inferno', 'magma', 'winter', 'cool'], description='colormap',
                                     layout=Layout(max_width='180px'))
    logarithmic_cbox = widgets.Checkbox(description='logarithmic', layout=Layout(max_width='180px'))
    cmap_control = widgets.VBox([cmap_selector, logarithmic_cbox])
    cmap_selector.observe(handler=loading_wrapper(change_colormap_name), type='change', names=('value',))
    logarithmic_cbox.observe(handler=loading_wrapper(change_colormap_log), type='change', names=('value',))
    add_widget(widgets.HBox([level_control, cmap_control]), pos='topleft', margin='5px')

    cbar_widget = widgets.HBox([interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))])
    add_widget(cbar_widget, pos='bottomright')

    m.add_control(ipyleaflet.LayersControl())
    m.add_control(ipyleaflet.FullScreenControl())
    m.observe(handler=on_zoom)

    change_date_range(dict(new=time_slider.value))
    display(m)


def geo_vis_shapes_app(data, simplify_nuts_shapes=True):
    nuts_ids_columns = ['origin', 'destination']
    nuts_shapes = get_nuts_shapes(simplify=simplify_nuts_shapes, tol=1e-3)

    geo_vis = interactive_output(plot_geo_shapes_vis,
                                 dict(nuts_shapes=fixed(nuts_shapes), data=fixed(data),
                                      nuts_ids_columns=fixed(nuts_ids_columns)))
    display(geo_vis)


def _wkb_hex_to_point(s):
    return list(shapely.wkb.loads(s, hex=True).coords)[0][::-1]


def _to_html(val):
    if type(val) in (list, tuple, np.array):
        return ', '.join(list(map(_to_html, val)))
    if type(val) is str:
        if val.startswith('http'):
            disp = val
            if val.endswith('png') or val.endswith('.jpg'):
                disp = '<img src="{}" width="250px" style="padding:3px">'.format(val, val)
            return '<a href={} target="_blank">{}</a>'.format(val, disp)
    return str(val)


def get_marker_cluster(data, geom_column, info_box: widgets.HTML, title_columns=()):
    def get_title(d):
        return '<br>'.join([_to_html(d[c]) for c in title_columns if d[c] not in (np.nan, None)])

    def get_hover_event_handler(info):
        def hover_event_handler(**kwargs):
            info_box.value = info

        return hover_event_handler

    locs = data[geom_column].apply(_wkb_hex_to_point)
    dicts = data.to_dict(orient='rows')

    markers = [ipyleaflet.Marker(location=loc, title=str(loc), draggable=False) for loc in locs]
    clusters = ipyleaflet.MarkerCluster(markers=markers, name='Marker Cluster')
    for marker, d in zip(clusters.markers, dicts):
        marker.on_mouseover(get_hover_event_handler(get_title(d)))
    return clusters


def plot_geo_data_cluster(data, geom_column, timestamp_column, date_range, title_columns):
    def date_filter(data, date_range):
        dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
        return data[(date_range[0] <= dates) & (dates <= date_range[1])]

    def toggle_tweets_table_visibility(change):
        app_state['is_in_bounds'] = None
        if change['old'] and not change['new']:
            tweets_table.value, app_state['is_in_bounds'] = '', None
        else:
            show_tweets_table()

    def show_tweets_table(*_):
        def in_bounds(loc):
            return all(bounds[0] < loc) and all(loc < bounds[1])

        if len(m.bounds) == 0:
            return

        bounds = np.array(m.bounds)
        locs = data[geom_column].apply(_wkb_hex_to_point)
        is_in_bounds = locs.apply(in_bounds)
        if tweets_table_cb.value and (
                app_state['is_in_bounds'] is None or not np.array_equal(is_in_bounds, app_state['is_in_bounds'])):
            if is_in_bounds.sum() > 0:
                tweets_table.value = data.loc[is_in_bounds, title_columns].reset_index(drop=True).to_html(
                    formatters={'media': _to_html}, escape=False, na_rep='', index=False)
            else:
                tweets_table.value = ''
        tweets_table_cb.description = 'Show {} Tweets'.format(is_in_bounds.sum())
        app_state['is_in_bounds'] = is_in_bounds

    data = date_filter(data, date_range).dropna(subset=[geom_column])
    m = ipyleaflet.Map(center=(51, 10), zoom=4, scroll_wheel_zoom=True)
    m.layout.height = '800px'
    app_state = dict(is_in_bounds=None)

    info_box = widgets.HTML('Hover over a marker', layout=Layout(margin='10px'))
    m.add_control(ipyleaflet.WidgetControl(widget=info_box, position='topright'))

    tweets_table = widgets.HTML(layout=widgets.Layout(overflow='scroll_hidden'))
    tweets_table_box = widgets.HBox([tweets_table],
                                    layout=Layout(max_height='500px', overflow_y='auto', max_width='900px'))
    tweets_table_cb = widgets.Checkbox(value=False)
    tweets_table_cb.observe(toggle_tweets_table_visibility
                            , type='change', names=('value',))
    tweets_box = widgets.VBox([tweets_table_cb, tweets_table_box])
    m.add_control(ipyleaflet.WidgetControl(widget=tweets_box, position='bottomleft'))

    marker_clusters = get_marker_cluster(data, geom_column, title_columns=title_columns, info_box=info_box)
    m.add_layer(marker_clusters)

    heatmap = ipyleaflet.Heatmap(locations=list(data[geom_column].apply(_wkb_hex_to_point).values), name='Heatmap',
                                 min_opacity=.1, blur=20, radius=20, max_zoom=12)

    m.add_layer(heatmap)
    m.add_control(ipyleaflet.FullScreenControl())
    m.add_control(ipyleaflet.LayersControl())
    m.observe(show_tweets_table)
    display(m)


def geo_vis_cluster_app(data, timestamp_column='_timestamp', geom_column='geom_tweet'):
    time_slider = get_time_slider(data)
    title_columns = widgets.SelectMultiple(options=sorted(data.columns), description='Information to show',
                                           value=('text', 'text_translated', '_timestamp', 'media'))
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
