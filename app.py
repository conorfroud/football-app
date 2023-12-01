import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from highlight_text import fig_text
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy import stats
import plotly.graph_objs as go
from matplotlib_venn import venn3
from math import pi
from mplsoccer import Pitch
from mplsoccer import PyPizza
from PIL import Image
import gspread

st.set_page_config(layout="wide")

pd.set_option("display.width", None)  # None means no width limit

# Create a function for each tab's content

def main_tab(df2):
    
    # Create a list of league options
    league_options = df2['League'].unique()

    # Create a list of score type options
    score_type_options = df2['Score Type'].unique()

    # Get the minimum and maximum age values from the DataFrame
    min_age = int(df2['Age'].min())
    max_age = int(df2['Age'].max())

    # Get the unique contract expiry years from the DataFrame
    contract_expiry_years = sorted(df2['Contract expires'].unique())

    # Get the minimum and maximum player market value (in euros) from the DataFrame
    min_player_market_value = int(df2['Market value (millions)'].min())
    max_player_market_value = int(df2['Market value (millions)'].max())

    min_stoke_score = 0.0
    max_stoke_score = 100.0

    # Add a sidebar dropdown box for leagues
    selected_league = st.sidebar.selectbox("Select a League", league_options)

    # Add a sidebar dropdown box for score types
    selected_score_type = st.sidebar.selectbox("Select a Score Type", score_type_options)

    stoke_range = st.sidebar.slider("Select Stoke Score Range", min_value=min_stoke_score, max_value=max_stoke_score, value=(min_stoke_score, max_stoke_score))
    
    # Add a slider for selecting the age range
    age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Add a multiselect box for selecting contract expiry years
    selected_contract_expiry_years = st.sidebar.multiselect("Select Contract Expiry Years", contract_expiry_years, default=contract_expiry_years)

    # Add a slider for selecting the player market value (in euros) range
    player_market_value_range = st.sidebar.slider("Select Player Market Value Range (Euro)", min_value=min_player_market_value, max_value=max_player_market_value, value=(min_player_market_value, max_player_market_value))

    # Add a slider for selecting the Average Distance Percentile range
    avg_distance_percentile_range = st.sidebar.slider("Select Average Distance Percentile Range", min_value=0, max_value=100, value=(0, 100))

    # Add a slider for selecting the Top 5 PSV-99 Percentile range
    top_5_psv_99_percentile_range = st.sidebar.slider("Select Top 5 PSV-99 Percentile Range", min_value=0, max_value=100, value=(0, 100))

   # Define a dictionary that maps 'Score Type' to columns
    score_type_column_mapping = {
        'Striker': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile', 'Contract expires', 'Market value (millions)', 'xG (ST)', 'Non-Penalty Goals (ST)', 'Shots (ST)', 'OBV Shot (ST)', 'Open Play xA (ST)', 'OBV Dribble & Carry (ST)', 'PAdj Pressures (ST)', 'Aerial Wins (ST)', 'Aerial Win % (ST)', 'total_gbe_score', 'GBE_Result'],
        'Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (W)', 'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)', 'xG (W)', 'Non-Penalty Goals (W)', 'Shots (W)', 'OBV Pass (W)', 'Open Play xA (W)', 'Successful Dribbles (W)', 'OBV Dribble & Carry (W)', 'total_gbe_score', 'GBE_Result'],
        'Attacking Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CAM)',	'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)', 'xG (CAM)', 'Non-Penalty Goals (CAM)', 'Shots (CAM)', 'OBV Pass (CAM)', 'Open Play xA (CAM)', 'Key Passes (CAM)', 'Throughballs (CAM)', 'Successful Dribbles (CAM)', 'OBV Dribble & Carry (CAM)', 'total_gbe_score', 'GBE_Result'],
        'Central Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (8)',	'Top 5 PSV-99 (8)', 'Contract expires', 'Market value (millions)', 'xG (8)', 'Non-Penalty Goals (8)', 'OBV Pass (8)', 'Open Play xA (8)', 'Deep Progressions (8)', 'Successful Dribbles (8)', 'OBV Dribble & Carry (8)', 'total_gbe_score', 'GBE_Result'],
        'Defensive Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)', 'Deep Progressions (6)', 'OBV Pass (6)', 'OBV Dribble & Carry (6)', 'PAdj Tackles (6)', 'PAdj Interceptions (6)', 'Tackle/Dribbled Past % (6)', 'OBV Defensive Action (6)', 'total_gbe_score', 'GBE_Result'],
        'Left Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (LB)', 'PAdj Interceptions (LB)', 'OBV Defensive Action (LB)', 'Tackle/Dribbled Past (LB)', 'Dribbled Past (LB)', 'OBV Dribble & Carry (LB)', 'Successful Dribbles (LB)', 'OBV Pass (LB)', 'Open Play xA (LB)', 'Key Passes (LB)', 'Successful Crosses (LB)', 'total_gbe_score', 'GBE_Result'],
        'Right Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (RB)', 'PAdj Interceptions (RB)', 'OBV Defensive Action (RB)', 'Tackle/Dribbled Past (RB)', 'Dribbled Past (RB)', 'OBV Dribble & Carry (RB)', 'Successful Dribbles (RB)', 'OBV Pass (RB)', 'Open Play xA (RB)', 'Key Passes (RB)', 'Successful Crosses (RB)', 'total_gbe_score', 'GBE_Result'],
        'Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CB)',	'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (CB)', 'Aerial Win % (CB)', 'PAdj Interceptions (CB)', 'PAdj Tackles (CB)', 'OBV Pass (CB)', 'Deep Progressions (CB)', 'OBV Dribble & Carry (CB)', 'OBV Defensive Action (CB)', 'total_gbe_score', 'GBE_Result'],
        'Stretch 9': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (S9)',	'Top 5 PSV-99 (S9)', 'Contract expires', 'Market value (millions)', 'xG (S9)', 'Non-Penalty Goals (S9)', 'Shots (S9)', 'OBV Shot (S9)', 'Open Play xA (S9)', 'OBV Dribble & Carry (S9)', 'PAdj Pressures (S9)', 'Runs in Behind (S9)', 'Threat of Runs in Behind (S9)', 'total_gbe_score', 'GBE_Result'],
        'Target 9': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (ST)',	'Top 5 PSV-99 (ST)', 'Contract expires', 'Market value (millions)', 'total_gbe_score', 'GBE_Result'],
        'Dribbling Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (DW)',	'Top 5 PSV (DW)', 'Contract expires', 'Market value (millions)', 'xG (DW)', 'Non-Penalty Goals (DW)', 'Shots (DW)', 'OBV Pass (DW)', 'Open Play xA (DW)', 'Successful Dribbles (DW)', 'OBV Dribble & Carry (DW)', 'total_gbe_score', 'GBE_Result'],
        'Creative Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CW)',	'Top 5 PSV (CW)', 'Contract expires', 'Market value (millions)', 'xG (CW)', 'Non-Penalty Goals (CW)', 'Shots (CW)', 'OBV Pass (CW)', 'Open Play xA (CW)', 'Successful Dribbles (CW)', 'OBV Dribble & Carry (CW)', 'total_gbe_score', 'GBE_Result'],
        'Goalscoring Wide Forward': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (WF)', 'Top 5 PSV (WF)', 'Contract expires', 'Market value (millions)', 'xG (WF)', 'Non-Penalty Goals (WF)', 'Shots (WF)', 'OBV Pass (WF)', 'Open Play xA (WF)', 'Successful Dribbles (WF)', 'OBV Dribble & Carry (WF)', 'total_gbe_score', 'GBE_Result'],
        'Running 10': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (R10)',	'Top 5 PSV (R10)', 'Contract expires', 'Market value (millions)', 'xG (R10)', 'Non-Penalty Goals (R10)', 'Shots (R10)', 'OBV Pass (R10)', 'Open Play xA (R10)', 'Successful Dribbles (R10)', 'OBV Dribble & Carry (R10)', 'PAdj Pressures (R10)', 'Pressure Regains (R10)', 'HI Distance (R10)', 'total_gbe_score', 'GBE_Result'],
        'Creative 10': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (C10)', 'Top 5 PSV (C10)', 'Contract expires', 'Market value (millions)', 'xG (C10)', 'Non-Penalty Goals (C10)', 'Shots (C10)', 'OBV Pass (C10)', 'Open Play xA (C10)', 'Key Passes (C10)', 'Successful Dribbles (C10)', 'OBV Dribble & Carry (C10)', 'total_gbe_score', 'GBE_Result'],
        'Progressive 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (P8)', 'Top 5 PSV (P8)', 'Contract expires', 'Market value (millions)', 'xG (P8)', 'OBV Pass (P8)', 'Open Play xA (P8)', 'Successful Dribbles (P8)', 'OBV Dribble & Carry (P8)', 'Deep Progressions (P8)', 'PAdj Pressures (P8)', 'total_gbe_score', 'GBE_Result'],
        'Running 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (R8)', 'Top 5 PSV (R8)', 'Contract expires', 'Market value (millions)', 'xG (R8)', 'Non-Penalty Goals (R8)', 'OBV Pass (R8)', 'Open Play xA (R8)', 'OBV Dribble & Carry (R8)', 'Deep Progressions (R8)', 'PAdj Pressures (R8)', 'Pressure Regains (R8)', 'Runs Threat Per Match (R8)', 'total_gbe_score', 'GBE_Result'],
        'Progressive 6': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (P6)', 'Top 5 PSV-99 (P6)', 'Contract expires', 'Market value (millions)', 'OBV Defensive Action (P6)', 'Deep Progressions (P6)', 'OBV Dribble & Carry (P6)', 'PAdj Tackles & Interceptions (P6)', 'OBV Pass (P6)', 'Forward Pass % (P6)', 'total_gbe_score', 'GBE_Result'],
        'Defensive 6': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (D6)', 'Top 5 PSV-99 (D6)', 'Contract expires', 'Market value (millions)', 'PAdj Pressures (D6)', 'OBV Defensive Action (D6)', 'Passing % (D6)', 'Tackle / Dribbled Past % (D6)', 'PAdj Tackles & Interceptions (D6)', 'Ball Recoveries (D6)', 'total_gbe_score', 'GBE_Result'],
        'Attacking LB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (ALB)', 'Top 5 PSV-99 (ALB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ALB)', 'PAdj Tackles (ALB)', 'PAdj Interceptions (ALB)', 'OBV Pass (ALB)', 'OBV Dribble & Carry (ALB)', 'Tackle / Dribbled Past % (ALB)', 'Threat of Runs (ALB)', 'Successful Crosses (ALB)', 'total_gbe_score', 'GBE_Result'],
        'Defensive LB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (DLB)', 'Top 5 PSV-99 (DLB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DLB)', 'PAdj Tackles (DLB)', 'PAdj Interceptions (DLB)', 'OBV Dribble & Carry (DLB)', 'OBV Defensive Action (DLB)', 'Tackle / Dribbled Past % (DLB)', 'PAdj Pressures (DLB)', 'Dribbled Past (DLB)', 'Aerial Win % (DLB)', 'total_gbe_score', 'GBE_Result'],
        'Attacking RB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (ARB)', 'Top 5 PSV-99 (ARB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ARB)', 'PAdj Tackles (ARB)', 'PAdj Interceptions (ARB)', 'OBV Pass (ARB)', 'OBV Dribble & Carry (ARB)', 'Tackle / Dribbled Past % (ARB)', 'Threat of Runs (ARB)', 'Successful Crosses (ARB)', 'total_gbe_score', 'GBE_Result'],
        'Defensive RB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (DRB)', 'Top 5 PSV-99 (DRB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DRB)', 'PAdj Tackles (DRB)', 'PAdj Interceptions (DRB)', 'OBV Dribble & Carry (DRB)', 'OBV Defensive Action (DRB)', 'Tackle / Dribbled Past % (DRB)', 'PAdj Pressures (DRB)', 'Dribbled Past (DRB)', 'Aerial Win % (DRB)', 'total_gbe_score', 'GBE_Result'],
        'Ball Playing Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (BCB)', 'Top 5 PSV-99 (BCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (BCB)', 'PAdj Interceptions (BCB)', 'PAdj Tackles (BCB)', 'OBV Pass (BCB)', 'OBV Dribble & Carry (BCB)', 'OBV Defensive Action (BCB)', 'Deep Progressions (BCB)', 'Pressured Change in Passing % (BCB)', 'total_gbe_score', 'GBE_Result'],
        'Dominant Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (DCB)', 'Top 5 PSV-99 (DCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (DCB)', 'Aerial Win % (DCB)', 'OBV Defensive Action (DCB)', 'Tackle / Dribbled Past % (DCB)', 'Blocks Per Shot (DCB)', 'total_gbe_score', 'GBE_Result'],
    }

    # Update the selected columns to include 'Score Type'
    selected_columns = score_type_column_mapping.get(selected_score_type, [])

    filtered_df = df2[(df2['League'] == selected_league) &
                (df2['Score Type'] == selected_score_type) &
                (df2['Age'] >= age_range[0]) &
                (df2['Age'] <= age_range[1]) &
                (df2['Contract expires'].isin(selected_contract_expiry_years)) &
                (df2['Market value (millions)'] >= player_market_value_range[0]) &
                (df2['Market value (millions)'] <= player_market_value_range[1]) &
                (df2['Stoke Score'] >= stoke_range[0]) &
                (df2['Stoke Score'] <= stoke_range[1]) &
                (df2[selected_columns[5]] >= avg_distance_percentile_range[0]) &
                (df2[selected_columns[5]] <= avg_distance_percentile_range[1]) &
                (df2[selected_columns[6]] >= top_5_psv_99_percentile_range[0]) &
                (df2[selected_columns[6]] <= top_5_psv_99_percentile_range[1])]

# Display the filtered DataFrame with selected columns
    st.dataframe(filtered_df[selected_columns], hide_index=True)

    # Add a download button to export the filtered DataFrame to a CSV file
    if not filtered_df.empty:
        csv_export = filtered_df[selected_columns].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_export,
            key="download_csv",
            file_name="filtered_data.csv",
            on_click=None,  # You can add a function to handle click events if needed
        )

def about_tab(df2):

    # Define the allowed score types
    allowed_score_types = ["Striker", "Winger", "Attacking Midfield", "Central Midfield", "Defensive Midfield", "Left Back", "Right Back", "Centre Back", "Stretch 9"]

    selected_player_1 = st.sidebar.selectbox(
        "Select Player 1:",
        options=df2["Player Name"].unique(),
        index=0  # Set the default index to the first player
    )

    selected_player_2 = st.sidebar.selectbox(
        "Select Player 2:",
        options=df2["Player Name"].unique(),
        index=1  # Set the default index to the second player
    )

    selected_player_df_1 = df2[df2["Player Name"] == selected_player_1]
    selected_player_df_2 = df2[df2["Player Name"] == selected_player_2]

    profile_options = selected_player_df_1[selected_player_df_1["Score Type"].isin(allowed_score_types)]["Score Type"].unique()

    selected_profile = st.sidebar.selectbox(
      "Select Profile:",
      options=profile_options,
      index=0  # Set the default index to the first profile
)

    # Define 'columns' based on the selected profile
    
    if selected_profile == "Striker":
       columns_1 = ["Player Name", "xG (ST)", "Non-Penalty Goals (ST)", "Shots (ST)", "OBV Shot (ST)", "Open Play xA (ST)", "OBV Dribble & Carry (ST)", "PAdj Pressures (ST)", "Average Distance Percentile", "Top 5 PSV-99 Percentile"]
       plot_title_1 = f"Forward Metrics for {selected_player_1}"

       columns_2 = ["Player Name", "xG (ST)", "Non-Penalty Goals (ST)", "Shots (ST)", "OBV Shot (ST)", "Open Play xA (ST)", "OBV Dribble & Carry (ST)", "PAdj Pressures (ST)", "Average Distance Percentile", "Top 5 PSV-99 Percentile"]
       plot_title_2 = f"Forward Metrics for {selected_player_2}"
    
    elif selected_profile == "Winger":
       columns_1 = ["Player Name", "xG (W)", "Non-Penalty Goals (W)", "Shots (W)", "Open Play xA (W)", "OBV Pass (W)", "Successful Dribbles (W)", "OBV Dribble & Carry (W)", "Average Distance (W)", "Top 5 PSV (W)"]
       plot_title_1 = f"Winger Metric Percentiles for {selected_player_1}"

       columns_2 = ["Player Name", "xG (W)", "Non-Penalty Goals (W)", "Shots (W)", "Open Play xA (W)", "OBV Pass (W)", "Successful Dribbles (W)", "OBV Dribble & Carry (W)", "Average Distance (W)", "Top 5 PSV (W)"]
       plot_title_2 = f"Winger Metric Percentiles for {selected_player_2}"

    selected_df_1 = selected_player_df_1[selected_player_df_1["Score Type"] == selected_profile_1]
    selected_df_2 = selected_player_df_2[selected_player_df_2["Score Type"] == selected_profile_2]

    percentiles_df_1 = selected_df_1[columns_1]
    percentiles_df_2 = selected_df_2[columns_2]

    percentiles_df_1 = percentiles_df_1.melt(id_vars="Player Name", var_name="Percentile Type", value_name="Percentile")
    percentiles_df_2 = percentiles_df_2.melt(id_vars="Player Name", var_name="Percentile Type", value_name="Percentile")
    
    # Load the Roboto font
    font_path = "Roboto-Bold.ttf"  # Replace with the actual path to the Roboto font
    prop = font_manager.FontProperties(fname=font_path)
    font_path1 = "Roboto-Regular.ttf"
    prop1 = font_manager.FontProperties(fname=font_path1)
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    with col3:

      params = percentiles_df_1["Percentile Type"]
      values1 = percentiles_df_1["Percentile"]
      values2 = percentiles_df_2["Percentile"]

    # Instantiate PyPizza class
      baker = PyPizza(
        params=params,
        background_color="#FFFFFF",
        straight_line_color="#222222",
        straight_line_lw=1,
        last_circle_lw=1,
        last_circle_color="#222222",
        other_circle_ls="-.",
        other_circle_lw=1
    )

    # Create the pizza plot
      fig, ax = baker.make_pizza(
      values1,                     # list of values
      compare_values=values2,    # comparison values
      figsize=(8, 8),             # adjust figsize according to your need
      kwargs_slices=dict(
        facecolor="#FF34B3", edgecolor="#222222",
        zorder=1, linewidth=1
    ),                          # values to be used when plotting slices
      kwargs_compare=dict(
        facecolor="#7EC0EE", edgecolor="#222222",
        zorder=2, linewidth=1,
    ),
      kwargs_params=dict(
        color="#000000", fontsize=12,
        va="center"
    ),                          # values to be used when adding parameter
      kwargs_values=dict(
        color="#000000", fontsize=12,
        zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="#FF34B3",
            boxstyle="round,pad=0.2", lw=1
        )
    ),                          # values to be used when adding parameter-values labels
      kwargs_compare_values=dict(
        color="#000000", fontsize=12, zorder=3,
        bbox=dict(edgecolor="#000000", facecolor="#7EC0EE", boxstyle="round,pad=0.2", lw=1)
    ),                          # values to be used when adding parameter-values labels
)


    st.pyplot(fig)


def scatter_plot(df):

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:

    # Sidebar with variable selection
       st.sidebar.header('Select Variables')
       x_variable = st.sidebar.selectbox('X-axis variable', df.columns, index=df.columns.get_loc('xG'))
       y_variable = st.sidebar.selectbox('Y-axis variable', df.columns, index=df.columns.get_loc('Open Play xG Assisted'))

# Create a multi-select dropdown for filtering by primary_position
       selected_positions = st.sidebar.multiselect('Filter by Primary Position', df['position_1'].unique())

       selected_league = st.sidebar.selectbox('Select League', df['competition_name'].unique())

# Create a multi-select dropdown for selecting players
    #selected_players = st.sidebar.multiselect('Select Players', df['Player Name'])

# Sidebar for filtering by 'minutes' played
       min_minutes = int(df['Minutes'].min())
       max_minutes = int(df['Minutes'].max())
       selected_minutes = st.sidebar.slider('Select Minutes Played Range', min_value=min_minutes, max_value=max_minutes, value=(300, max_minutes))

# Sidebar for filtering by league (allow only one league to be selected)
    
# Filter data based on user-selected positions, players, minutes played, and league
       filtered_df = df[(df['position_1'].isin(selected_positions) | (len(selected_positions) == 0)) & 
                 #(df['Player Name'].isin(selected_players) | (len(selected_players) == 0)) &
                 (df['Minutes'] >= selected_minutes[0]) &
                 (df['Minutes'] <= selected_minutes[1]) &
                 (df['competition_name'] == selected_league)]

# Calculate Z-scores for the variables
       filtered_df['z_x'] = (filtered_df[x_variable] - filtered_df[x_variable].mean()) / filtered_df[x_variable].std()
       filtered_df['z_y'] = (filtered_df[y_variable] - filtered_df[y_variable].mean()) / filtered_df[y_variable].std()

# Define a threshold for labeling outliers (you can customize this threshold)
    threshold = st.sidebar.slider('Label Threshold', min_value=0.1, max_value=5.0, value=2.0)

# Create a scatter plot using Plotly with the filtered data
    fig = px.scatter(filtered_df, x=x_variable, y=y_variable, hover_data={'Player Name': True, 'team_name':True, 'age':True, x_variable:False, y_variable:False})

# Customize the marker color and size
    fig.update_traces(marker=dict(size=12, color='#7EC0EE'))

# Set the plot size
    fig.update_layout(width=800, height=600)

# Filter and label outliers
    outliers = filtered_df[(filtered_df['z_x'].abs() > threshold) | (filtered_df['z_y'].abs() > threshold)]

    fig.add_trace(
     go.Scatter(
        x=outliers[x_variable],
        y=outliers[y_variable],
        text=outliers['Player Name'],
        mode='text',
        showlegend=False,
        textposition='top center'
    )
)

    fig.update_layout(annotations=[], hovermode='closest')

# Display the plot in Streamlit
    with col2:
        st.plotly_chart(fig)

def comparison_tab(df):

    # Filter the DataFrame based on selected players
    selected_players = st.sidebar.multiselect("Select Players", df["Player Name"])
    
    # Sidebar: Metric selection
    selected_metrics = st.sidebar.multiselect("Select Metrics", df.columns[1:])

    # Add a "Total" option for selected metrics
    total_option = st.sidebar.checkbox("Total", key="total_checkbox")

    # Metrics to exclude from total calculation
    exclude_from_total = ['Top 5 PSV-99']

    # Remove excluded metrics from selected metrics if total option is selected
    if total_option:
        selected_metrics = [metric for metric in selected_metrics if metric not in exclude_from_total]

    filtered_df = df[df["Player Name"].isin(selected_players)]

    def highlight_best_player(s):
        is_best = s == s.max()
        return ['background-color: #00CD00' if v else '' for v in is_best]

    # Create a new DataFrame for calculated totals
    calculated_df = filtered_df.copy()

    # Calculate totals if the "Total" checkbox is selected
    if total_option:
        selected_metrics_without_minutes = [metric for metric in selected_metrics]
        calculated_df[selected_metrics_without_minutes] = calculated_df[selected_metrics_without_minutes].multiply((calculated_df["Minutes"]/90), axis="index")

    # Display the table with conditional formatting
    if selected_metrics:
        if filtered_df.empty:
            st.warning("No players selected. Please select at least one player.")
        else:
            selected_columns =  ["Player Name"] + ["Minutes"] + selected_metrics
            if total_option:
                formatted_df = calculated_df[selected_columns].copy()
            else:
                formatted_df = filtered_df[selected_columns].copy()
            formatted_df = formatted_df.style.apply(highlight_best_player, subset=selected_metrics)
            # Format numbers to two decimal places
            formatted_df = formatted_df.format({"Minutes": "{:.0f}"}, subset=["Minutes"])
            formatted_df = formatted_df.format("{:.2f}", subset=selected_metrics)
            st.dataframe(formatted_df, hide_index=True)
    else:
        st.warning("Select at least one metric to compare.")


# Load the DataFrame
df = pd.read_csv("belgiumdata.csv")
df2 = pd.read_csv("championshipscores.csv")

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Stoke Score", "Player Profile", "Scatter Plot", "Comparison Tab"])

# Based on the selected tab, display the corresponding content
if selected_tab == "Stoke Score":
    main_tab(df2)
if selected_tab == "Player Profile":
    about_tab(df2)  # Pass the DataFrame to the about_tab function
if selected_tab == "Scatter Plot":
    scatter_plot(df)
elif selected_tab == "Comparison Tab":
    comparison_tab(df)

