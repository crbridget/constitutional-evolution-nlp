import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
pio.renderers.default = 'browser'


def _code_mapping(df, src, targ):
    """ Map labels in src and targ columns to integers """

    labels = pd.concat([df[src], df[targ]]).unique().tolist()

    # Get integer codes
    codes = range(len(labels))

    # Create label to code mapping
    lc_map = dict(zip(labels, codes))

    # Substitute names for codes in dataframe
    new_df = df.copy()
    new_df[src] = new_df[src].map(lc_map)
    new_df[targ] = new_df[targ].map(lc_map)

    # return the new df along with the list of labels that were converted
    return new_df, labels


def make_sankey(df, src, targ, vals=None,  **kwargs):
    """ Generate a sankey diagram
    df - Dataframe
    src - Source column
    targ - Target column
    vals - Values column (optional)
    kwargs - optional supported params: pad, thickness, line_color, line_width.
    """

    # Handle optional vals column
    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)  # all values are identical (e.g., 1)

    # Convert column labels to integer codes
    df, labels = _code_mapping(df, src, targ)

    # Extract customizations from kwargs
    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 50)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 0)

    # Construct sankey figure
    link = {'source': df[src], 'target': df[targ], 'value': values,
            'line': {'color': line_color, 'width': line_width}}

    node = {'label': labels, 'pad': pad, 'thickness': thickness,
            'line': {'color': line_color, 'width': line_width}}
    
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    # Optionally adjusting the width and height of the sankey diagram
    width = kwargs.get('width', 1200)
    height = kwargs.get('height', 800)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )


    # Make sankey simply returns the figure
    # If you want to actually display the figure, use show sankey
    return fig


def show_sankey(df, src, targ, vals=None, png=None, **kwargs):
    """
    Make AND Show the sankey diagram.   Optionally save it to a file
    df - The dataframe
    src - The source column
    targ - The target column
    vals - optional values column (line thickness)
    png - name of the .png image file to be generated
    kwargs - optional customizations like thickness, line color, etc.
    """
    fig = make_sankey(df, src, targ, vals, **kwargs)
    fig.show()
    if png:
        fig.write_image(png)


