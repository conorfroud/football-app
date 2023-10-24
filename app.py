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
        'Central Midfield': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (8)',	'Top 5 PSV (8)', 'Contract expires', 'Market value (millions)'],
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
        'Progressive 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (8)', 'Top 5 PSV (8)', 'Contract expires', 'Market value (millions)'],
        'Running 8': ['Player Name', 'Age', 'Team', 'League', 'Stoke Score', 'Average Distance (8)', 'Top 5 PSV (8)', 'Contract expires', 'Market value (millions)'],
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
    st.dataframe(filtered_df[selected_columns])

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

    selected_player = st.sidebar.selectbox(
        "Select a Player:",
        options=df2["Player Name"].unique(),
        index=0  # Set the default index to the first player
    )

    selected_player_df = df2[df2["Player Name"] == selected_player]

    available_profiles = selected_player_df["Score Type"].unique()
    selected_profile = st.sidebar.selectbox(
        "Select a Profile:",
        options=available_profiles,
        index=0  # Set the default index to the first profile
    )

    # Define 'columns' based on the selected profile
    if selected_profile == "Striker":
        columns = ["Player Name", "Top 5 PSV-99 Percentile", "Average Distance Percentile", "PAdj Pressures", "Dribble & Carry OBV", "xA per 90", "player_season_obv_shot_90 Percentile", "Shots per 90", "Non-Pen Goals per 90", "xG per 90"]
        plot_title = f"Forward Metrics for {selected_player}"
    elif selected_profile == "Winger":
        columns = ["Player Name", "Average Distance (W)", "Top 5 PSV (W)", "OBV Dribble & Carry (W)", "Dribbles per 90 (W)",  "xA per 90 (W)", "NP Shots per 90 (W)", "Non-Pen Goals per 90 (W)", "NP xG per 90 (W)"]
        plot_title = f"Winger Metric Percentiles for {selected_player}"
    elif selected_profile == "Attacking Midfield":
        columns = ["Player Name", "Average Distance Percentile", "Top 5 PSV-99 Percentile", "Dribble & Carry OBV", "Dribbles per 90 (10)",  "xA per 90", "Shots per 90", "Non-Pen Goals per 90 (10)", "NP xG per 90 (10)"]
        plot_title = f"Attacking Midfield Metric Percentiles for {selected_player}"
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

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(16, 10))
    # Set the background color
    #fig.patch.set_facecolor("#F5F5F5")
    #ax.set_facecolor("#F5F5F5")
    ax.barh(percentiles_df["Percentile Type"], percentiles_df["Percentile"], color="#7EC0EE")
    ax.set_xlabel("League Average", fontproperties=prop1)
    ax.set_ylabel("Percentile Type", fontproperties=prop1)
    ax.set_title(plot_title, fontproperties=prop, fontsize=22, pad=20)  # Adjust the pad value to move the title higher
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop)
    ax.set_xlim(0, 100)
    for index, value in enumerate(percentiles_df["Percentile"]):
        ax.text(value + 1, index, f"{value:.0f}", va="center", fontproperties=prop)

    plt.axvline(x=50, color='Black', linestyle='--', label='Horizontal Line at x=50')
    #fig_text(x = 0.5, y = 0.03, s = "League Average", color = 'Black', family="Roboto", fontsize=10, fontweight="bold")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#ccc8c8')
    ax.spines['bottom'].set_color('#ccc8c8')

    plt.tight_layout()

    #st.pyplot(fig)

    col1, col2, col3, col4, col5= st.columns([1,1, 5, 1, 1])
    
    with col3:
        
        params = percentiles_df["Percentile Type"]
        values1 = percentiles_df["Percentile"]

# Instantiate PyPizza class
        baker = PyPizza(
        params=params,
        background_color="#F5F5F5",
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
           color="#000000", fontsize=12, va="center"
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
    st.plotly_chart(fig)

# Load the DataFrame
df = pd.read_csv("belgiumdata.csv")
df2 = pd.read_csv("championshipscores.csv")

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Stoke Score", "Player Profile", "Scatter Plot"])

# Based on the selected tab, display the corresponding content
if selected_tab == "Stoke Score":
    main_tab(df2)
if selected_tab == "Player Profile":
    about_tab(df2)  # Pass the DataFrame to the about_tab function
elif selected_tab == "Scatter Plot":
    scatter_plot(df)
#elif selected_tab == "Venn Diagram":
    #venn_tab(df)  # Pass the DataFrame to the about_tab function

