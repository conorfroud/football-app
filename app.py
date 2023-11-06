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
        'Striker': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile', 'Contract expires', 'Market value (millions)'],
        'Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (W)', 'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)'],
        'Attacking Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CAM)',	'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)'],
        'Central Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (8)',	'Top 5 PSV-99 (8)', 'Contract expires', 'Market value (millions)'],
        'Defensive Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)'],
        'Left Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'Contract expires', 'Market value (millions)'],
        'Right Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'Contract expires', 'Market value (millions)'],
        'Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CB)',	'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)'],
        'Stretch 9': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (ST)',	'Top 5 PSV-99 (ST)', 'Contract expires', 'Market value (millions)'],
        'Target 9': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (ST)',	'Top 5 PSV-99 (ST)', 'Contract expires', 'Market value (millions)'],
        'Dribbling Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (W)',	'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)'],
        'Creative Winger': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (W)',	'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)'],
        'Goalscoring Wide Forward': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (W)',	'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)'],
        'Running 10': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CAM)',	'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)'],
        'Creative 10': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CAM)', 'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)'],
        'Progressive 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (P8)', 'Top 5 PSV (P8)', 'Contract expires', 'Market value (millions)'],
        'Running 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (R8)', 'Top 5 PSV (R8)', 'Contract expires', 'Market value (millions)'],
        'Progressive 6': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)'],
        'Defensive 6': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)'],
        'Attacking LB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (FB)', 'Top 5 PSV (FB)', 'Contract expires', 'Market value (millions)'],
        'Defensive LB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (FB)', 'Top 5 PSV (FB)', 'Contract expires', 'Market value (millions)'],
        'Attacking RB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (FB)', 'Top 5 PSV (FB)', 'Contract expires', 'Market value (millions)'],
        'Defensive RB': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (FB)', 'Top 5 PSV (FB)', 'Contract expires', 'Market value (millions)'],
        'Ball Playing Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CB)', 'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)'],
        'Dominant Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (CB)', 'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)'],
    }

    # Update the selected columns to include 'Score Type'
    selected_columns = score_type_column_mapping.get(selected_score_type, [])
    selected_columns.append('Score Type')  # Add 'Score Type' to the selected columns

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
    allowed_score_types = ["Striker", "Winger", "Attacking Midfield", "Central Midfield", "Defensive Midfield", "Left Back", "Right Back", "Centre Back"]

    selected_player = st.sidebar.selectbox(
      "Select a Player:",
      options=df2["Player Name"].unique(),
      index=0  # Set the default index to the first player
)

    selected_player_df = df2[df2["Player Name"] == selected_player]

# Filter the available profiles based on the allowed score types
    available_profiles = selected_player_df[selected_player_df["Score Type"].isin(allowed_score_types)]["Score Type"].unique()

    selected_profile = st.sidebar.selectbox(
      "Select a Profile:",
      options=available_profiles,
      index=0  # Set the default index to the first profile
)

    # Define 'columns' based on the selected profile
    if selected_profile == "Striker":
        columns = ["Player Name", "xG per 90", "Non-Pen Goals per 90", "Shots per 90", "OBV Shot per 90", "xA per 90", "Dribble & Carry OBV", "PAdj Pressures", "Average Distance Percentile", "Top 5 PSV-99 Percentile"]
        plot_title = f"Forward Metrics for {selected_player}"
    elif selected_profile == "Winger":
        columns = ["Player Name", "NP xG per 90 (W)", "Non-Pen Goals per 90 (W)", "NP Shots per 90 (W)", "xA per 90 (W)", "Dribbles per 90 (W)", "OBV Dribble & Carry (W)",  "Average Distance (W)", "Top 5 PSV (W)"]
        plot_title = f"Winger Metric Percentiles for {selected_player}"
    elif selected_profile == "Attacking Midfield":
        columns = ["Player Name", "NP xG per 90 (CAM)", "Non-Pen Goals per 90 (CAM)", "NP Shots per 90 (CAM)", "OBV Pass per 90 (CAM)", "xA per 90 (CAM)", "Dribbles per 90 (CAM)", "OBV Dribble & Carry (CAM)",  "Average Distance (CAM)", "Top 5 PSV (CAM)"]
        plot_title = f"Attacking Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Central Midfield":
        columns = ["Player Name", "NP xG (8)", "NP Goals (8)", "OBV Pass (8)", "OP xA (8)", "Deep Progressions (8)", "Dribbles (8)", "OBV Dribble & Carry (8)",  "Average Distance (8)", "Top 5 PSV-99 (8)"]
        plot_title = f"Central Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Defensive Midfield":
        columns = ["Player Name", "Deep Progressions (6)", "OBV Pass (6)", "Dribbles (6)", "OBV Dribble & Carry (6)", "Tackle/Dribbled Past % (6)", "PAdj Tackles (6)", "PAdj Interceptions (6)",  "Average Distance (6)", "Top 5 PSV-99 (6)"]
        plot_title = f"Defensive Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Left Back":
        columns = ["Player Name", "PAdj Tackles (LB)", "PAdj Interceptions (LB)", "OBV Defensive Action (LB)", "Dribbled Past (LB)", "Successful Dribbles (LB)", "OBV Dribble & Carry (LB)", "Successful Crosses (LB)", "Open Play xA (LB)", "OBV Pass (LB)", "Average Distance (LB)", "Top 5 PSV-99 (LB)"]
        plot_title = f"Left Back Metric Percentiles for {selected_player}"
    elif selected_profile == "Right Back":
        columns = ["Player Name", "PAdj Tackles (RB)", "PAdj Interceptions (RB)", "OBV Defensive Action (RB)", "Dribbled Past (RB)", "Successful Dribbles (RB)", "OBV Dribble & Carry (RB)", "Successful Crosses (RB)", "Open Play xA (RB)", "OBV Pass (RB)", "Average Distance (RB)", "Top 5 PSV-99 (RB)"]
        plot_title = f"Right Back Metric Percentiles for {selected_player}"
    elif selected_profile == "Centre Back":
        columns = ["Player Name", "Aerial Wins (CB)", "Aerial Win % (CB)", "PAdj Tackles (CB)", "PAdj Interceptions (CB)", "Dribbled Past (CB)", "OBV Defensive Action (CB)", "Deep Progressions (CB)", "OBV Pass (CB)", "OBV Dribble & Carry (CB)",  "Average Distance (CB)", "Top 5 PSV-99 (CB)"]
        plot_title = f"Centre Back Metric Percentiles for {selected_player}"
    else:
        # Define columns and plot title for the default profile
        columns = []
        plot_title = f"Default Profile Metrics for {selected_player}"

    # Assuming selected_df is your DataFrame containing the data
    selected_df = selected_player_df[selected_player_df["Score Type"] == selected_profile]

    percentiles_df = selected_df[columns]
    percentiles_df = percentiles_df.melt(id_vars="Player Name", var_name="Percentile Type", value_name="Percentile")
    
    # Load the Roboto font
    font_path = "Roboto-Bold.ttf"  # Replace with the actual path to the Roboto font
    prop = font_manager.FontProperties(fname=font_path)
    font_path1 = "Roboto-Regular.ttf"
    prop1 = font_manager.FontProperties(fname=font_path1)
    
    col1, col2, col3, col4, col5= st.columns([1,1, 5, 1, 1])
    
    with col3:
        
        params = percentiles_df["Percentile Type"]
        values1 = percentiles_df["Percentile"]

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
        fig2, ax = baker.make_pizza(
        values1,                     
        figsize=(8, 8),            
        kwargs_slices=dict(
           facecolor="#7EC0EE", edgecolor="#222222",
           zorder=1, linewidth=1
    ),
        kwargs_compare=dict(
           facecolor="#7EC0EE", edgecolor="#222222",
           zorder=2, linewidth=1,
    ),
        kwargs_params=dict(
           color="#000000", fontsize=10, va="center"
    ),
        kwargs_values=dict(
           color="#000000", fontsize=12, zorder=3,
           bbox=dict(
            edgecolor="#000000", facecolor="#7EC0EE",
            boxstyle="round,pad=0.2", lw=1
        )
    ),
        kwargs_compare_values=dict(
           color="#000000", fontsize=12, zorder=3,
           bbox=dict(edgecolor="#000000", facecolor="#7EC0EE", boxstyle="round,pad=0.2", lw=1)
    )
)
    
        st.pyplot(fig2)

def scatter_plot(df):

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:

    # Sidebar with variable selection
       st.sidebar.header('Select Variables')
       x_variable = st.sidebar.selectbox('X-axis variable', df.columns, index=df.columns.get_loc('np_xg_90'))
       y_variable = st.sidebar.selectbox('Y-axis variable', df.columns, index=df.columns.get_loc('op_xa_90'))

# Create a multi-select dropdown for filtering by primary_position
       selected_positions = st.sidebar.multiselect('Filter by Primary Position', df['position_1'].unique())

       selected_league = st.sidebar.selectbox('Select League', df['competition_name'].unique())

# Create a multi-select dropdown for selecting players
    #selected_players = st.sidebar.multiselect('Select Players', df['player_name'])

# Sidebar for filtering by 'minutes' played
       min_minutes = int(df['minutes'].min())
       max_minutes = int(df['minutes'].max())
       selected_minutes = st.sidebar.slider('Select Minutes Played Range', min_value=min_minutes, max_value=max_minutes, value=(250, max_minutes))

# Sidebar for filtering by league (allow only one league to be selected)
    
# Filter data based on user-selected positions, players, minutes played, and league
       filtered_df = df[(df['position_1'].isin(selected_positions) | (len(selected_positions) == 0)) & 
                 #(df['player_name'].isin(selected_players) | (len(selected_players) == 0)) &
                 (df['minutes'] >= selected_minutes[0]) &
                 (df['minutes'] <= selected_minutes[1]) &
                 (df['competition_name'] == selected_league)]

# Calculate Z-scores for the variables
       filtered_df['z_x'] = (filtered_df[x_variable] - filtered_df[x_variable].mean()) / filtered_df[x_variable].std()
       filtered_df['z_y'] = (filtered_df[y_variable] - filtered_df[y_variable].mean()) / filtered_df[y_variable].std()

# Define a threshold for labeling outliers (you can customize this threshold)
    threshold = st.sidebar.slider('Label Threshold', min_value=0.1, max_value=5.0, value=2.0)

# Create a scatter plot using Plotly with the filtered data
    fig = px.scatter(filtered_df, x=x_variable, y=y_variable, hover_data={'player_name': True, 'team_name':True, x_variable:False, y_variable:False})

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
        text=outliers['player_name'],
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
    
    # Title and description
    st.write("Select players and metrics to compare in a table.")

    # Sidebar: Player selection
    selected_players = st.sidebar.multiselect("Select Players", df["player_name"])

    # Sidebar: Metric selection
    selected_metrics = st.sidebar.multiselect("Select Metrics", df.columns[1:])

    # Add a "Total" option for selected metrics
    total_option = st.sidebar.checkbox("Total", key="total_checkbox")

    # Filter the DataFrame based on selected players
    filtered_df = df[df["player_name"].isin(selected_players)]

    # Define a function to calculate totals based on selected metrics
    def calculate_totals(sub_df, metrics, total_option):
        if not total_option:
            return sub_df
        for metric in metrics:
            if metric != "minutes":
                sub_df[f"{metric}_total"] = sub_df[metric] * sub_df["minutes"]
        return sub_df

    # Display the table with conditional formatting
    if selected_metrics:
        if filtered_df.empty:
            st.warning("No players selected. Please select at least one player.")
        else:
            selected_columns = ["player_name"] + selected_metrics
            formatted_df = calculate_totals(filtered_df[selected_columns].copy(), selected_metrics, total_option)
            formatted_df = formatted_df.style.apply(highlight_best_player, subset=selected_metrics)
            # Format numbers to two decimal places
            formatted_df = formatted_df.format("{:.2f}", subset=selected_metrics)
            st.dataframe(formatted_df, hide_index=True)
    else:
        st.warning("Select at least one metric to compare.")

def highlight_best_player(s):
    is_best = s == s.max()
    return ['background-color: #00CD00' if v else '' for v in is_best]

comparison_tab(df)

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

