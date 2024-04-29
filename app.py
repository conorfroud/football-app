import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_gsheets import GSheetsConnection
from mplsoccer.pitch import Pitch, VerticalPitch
import io
import base64
import requests

st.set_page_config(layout="wide")

pd.set_option("display.width", None)  # None means no width limit

# Create a function for each tab's content

def main_tab(df2):
    
    # Create a list of league options
    league_options = df2['League'].unique()
    
    # Define the custom order for leagues
    custom_league_order = ['English Championship', 'Belgian Jupiler Pro League', 'Dutch Eredivisie', 'Portuguese Primeira Liga', 'Band 2']
    
    # Filter out the custom ordered leagues and sort them alphabetically
    custom_ordered_leagues = sorted([league for league in custom_league_order if league in league_options])
    
    # Add the remaining leagues in their original order
    remaining_leagues = [league for league in league_options if league not in custom_ordered_leagues]
    
    # Concatenate the custom ordered leagues and remaining leagues
    league_options_ordered = custom_ordered_leagues + remaining_leagues

    # Create a list of score type options
    score_type_options = df2['Score Type'].unique()

    # Get the minimum and maximum age values from the DataFrame
    min_age = int(df2['Age'].min())
    max_age = int(df2['Age'].max())

    # Get the unique contract expiry years from the DataFrame
    contract_expiry_years = sorted(df2['Contract expires'].unique())

    # Create a list of primary position options
    primary_position_options = df2['Primary Position'].unique()

    # Get the minimum and maximum player market value (in euros) from the DataFrame
    min_player_market_value = int(df2['Market value (millions)'].min())
    max_player_market_value = int(df2['Market value (millions)'].max())

    min_stoke_score = 0.0
    max_stoke_score = 100.0

    # Add a sidebar multiselect box for leagues with default selections
    selected_leagues = st.sidebar.multiselect("Select Leagues", league_options, default=['English Championship'])

    # Filter seasons based on selected leagues
    filtered_season_options = df2[df2['League'].isin(selected_leagues)]['Season'].unique()

    # Reverse the order of the filtered seasons
    filtered_season_options = filtered_season_options[::-1]

    # Add a sidebar dropdown box for selecting seasons
    selected_season = st.sidebar.selectbox("Select a Season", filtered_season_options)

    # Add a sidebar dropdown box for score types
    selected_score_type = st.sidebar.selectbox("Select a Score Type", score_type_options)

    stoke_range = st.sidebar.slider("Select Stoke Score Range", min_value=min_stoke_score, max_value=max_stoke_score, value=(min_stoke_score, max_stoke_score))

    # Add a slider for selecting the age range
    age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Add a slider for selecting the L/R Footedness % range
    lr_footedness_range = st.sidebar.slider("Select L/R Footedness % Range", min_value=0, max_value=100, value=(0, 100))

    # Add a multiselect box for selecting primary positions
    selected_primary_positions = st.sidebar.multiselect("Select Primary Positions", primary_position_options, default=primary_position_options)

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
        'Striker': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile', 'Contract expires', 'Market value (millions)', 'xG (ST)', 'Non-Penalty Goals (ST)', 'Shots (ST)', 'OBV Shot (ST)', 'Open Play xA (ST)', 'OBV Dribble & Carry (ST)', 'PAdj Pressures (ST)', 'Aerial Wins (ST)', 'Aerial Win % (ST)', 'L/R Footedness %'],
        'Winger': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Distance (W)', 'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)', 'xG (W)', 'Non-Penalty Goals (W)', 'Shots (W)', 'OBV Pass (W)', 'Open Play xA (W)', 'Successful Dribbles (W)', 'OBV Dribble & Carry (W)', 'L/R Footedness %'],
        'Attacking Midfield': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CAM)',	'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)', 'xG (CAM)', 'Non-Penalty Goals (CAM)', 'Shots (CAM)', 'OBV Pass (CAM)', 'Open Play xA (CAM)', 'Key Passes (CAM)', 'Throughballs (CAM)', 'Successful Dribbles (CAM)', 'OBV Dribble & Carry (CAM)', 'L/R Footedness %'],
        'Central Midfield': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (8)',	'Top 5 PSV-99 (8)', 'Contract expires', 'Market value (millions)', 'xG (8)', 'Non-Penalty Goals (8)', 'OBV Pass (8)', 'Open Play xA (8)', 'Deep Progressions (8)', 'Successful Dribbles (8)', 'OBV Dribble & Carry (8)', 'L/R Footedness %'],
        'Defensive Midfield': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)', 'Deep Progressions (6)', 'OBV Pass (6)', 'OBV Dribble & Carry (6)', 'PAdj Tackles (6)', 'PAdj Interceptions (6)', 'Tackle/Dribbled Past % (6)', 'OBV Defensive Action (6)', 'L/R Footedness %'],
        'Left Back': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (LB)', 'PAdj Interceptions (LB)', 'OBV Defensive Action (LB)', 'Tackle/Dribbled Past (LB)', 'Dribbled Past (LB)', 'OBV Dribble & Carry (LB)', 'Successful Dribbles (LB)', 'OBV Pass (LB)', 'Open Play xA (LB)', 'Key Passes (LB)', 'Successful Crosses (LB)', 'L/R Footedness %'],
        'Right Back': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (RB)', 'PAdj Interceptions (RB)', 'OBV Defensive Action (RB)', 'Tackle/Dribbled Past (RB)', 'Dribbled Past (RB)', 'OBV Dribble & Carry (RB)', 'Successful Dribbles (RB)', 'OBV Pass (RB)', 'Open Play xA (RB)', 'Key Passes (RB)', 'Successful Crosses (RB)', 'L/R Footedness %'],
        'Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CB)',	'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (CB)', 'Aerial Win % (CB)', 'PAdj Interceptions (CB)', 'PAdj Tackles (CB)', 'OBV Pass (CB)', 'Deep Progressions (CB)', 'OBV Dribble & Carry (CB)', 'OBV Defensive Action (CB)', 'L/R Footedness %'],
        'Stretch 9': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (S9)',	'Top 5 PSV-99 (S9)', 'Contract expires', 'Market value (millions)', 'xG (S9)', 'Non-Penalty Goals (S9)', 'Shots (S9)', 'OBV Shot (S9)', 'Open Play xA (S9)', 'OBV Dribble & Carry (S9)', 'PAdj Pressures (S9)', 'Runs in Behind (S9)', 'Threat of Runs in Behind (S9)', 'L/R Footedness %'],
        'Target 9': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ST)',	'Top 5 PSV-99 (ST)', 'Contract expires', 'Market value (millions)'],
        'Dribbling Winger': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DW)',	'Top 5 PSV (DW)', 'Contract expires', 'Market value (millions)', 'xG (DW)', 'Non-Penalty Goals (DW)', 'Shots (DW)', 'OBV Pass (DW)', 'Open Play xA (DW)', 'Successful Dribbles (DW)', 'OBV Dribble & Carry (DW)', 'L/R Footedness %'],
        'Creative Winger': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CW)',	'Top 5 PSV (CW)', 'Contract expires', 'Market value (millions)', 'xG (CW)', 'Non-Penalty Goals (CW)', 'Shots (CW)', 'OBV Pass (CW)', 'Open Play xA (CW)', 'Successful Dribbles (CW)', 'OBV Dribble & Carry (CW)', 'L/R Footedness %'],
        'Goalscoring Wide Forward': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (WF)', 'Top 5 PSV (WF)', 'Contract expires', 'Market value (millions)', 'xG (WF)', 'Non-Penalty Goals (WF)', 'Shots (WF)', 'OBV Pass (WF)', 'Open Play xA (WF)', 'Successful Dribbles (WF)', 'OBV Dribble & Carry (WF)', 'L/R Footedness %'],
        'Running 10': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (R10)',	'Top 5 PSV (R10)', 'Contract expires', 'Market value (millions)', 'xG (R10)', 'Non-Penalty Goals (R10)', 'Shots (R10)', 'OBV Pass (R10)', 'Open Play xA (R10)', 'Successful Dribbles (R10)', 'OBV Dribble & Carry (R10)', 'PAdj Pressures (R10)', 'Pressure Regains (R10)', 'HI Distance (R10)', 'L/R Footedness %'],
        'Creative 10': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (C10)', 'Top 5 PSV (C10)', 'Contract expires', 'Market value (millions)', 'xG (C10)', 'Non-Penalty Goals (C10)', 'Shots (C10)', 'OBV Pass (C10)', 'Open Play xA (C10)', 'Key Passes (C10)', 'Successful Dribbles (C10)', 'OBV Dribble & Carry (C10)', 'L/R Footedness %'],
        'Progressive 8': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (P8)', 'Top 5 PSV (P8)', 'Contract expires', 'Market value (millions)', 'xG (P8)', 'OBV Pass (P8)', 'Open Play xA (P8)', 'Successful Dribbles (P8)', 'OBV Dribble & Carry (P8)', 'Deep Progressions (P8)', 'PAdj Pressures (P8)', 'L/R Footedness %'],
        'Running 8': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (R8)', 'Top 5 PSV (R8)', 'Contract expires', 'Market value (millions)', 'xG (R8)', 'Non-Penalty Goals (R8)', 'OBV Pass (R8)', 'Open Play xA (R8)', 'OBV Dribble & Carry (R8)', 'Deep Progressions (R8)', 'PAdj Pressures (R8)', 'Pressure Regains (R8)', 'Runs Threat Per Match (R8)', 'L/R Footedness %'],
        'Progressive 6': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (P6)', 'Top 5 PSV-99 (P6)', 'Contract expires', 'Market value (millions)', 'OBV Defensive Action (P6)', 'Deep Progressions (P6)', 'OBV Dribble & Carry (P6)', 'PAdj Tackles & Interceptions (P6)', 'OBV Pass (P6)', 'Forward Pass % (P6)', 'L/R Footedness %'],
        'Defensive 6': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (D6)', 'Top 5 PSV-99 (D6)', 'Contract expires', 'Market value (millions)', 'PAdj Pressures (D6)', 'OBV Defensive Action (D6)', 'Passing % (D6)', 'Tackle / Dribbled Past % (D6)', 'PAdj Tackles & Interceptions (D6)', 'Ball Recoveries (D6)', 'L/R Footedness %'],
        'Attacking LB': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ALB)', 'Top 5 PSV-99 (ALB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ALB)', 'PAdj Tackles (ALB)', 'PAdj Interceptions (ALB)', 'OBV Pass (ALB)', 'OBV Dribble & Carry (ALB)', 'Tackle / Dribbled Past % (ALB)', 'Threat of Runs (ALB)', 'Successful Crosses (ALB)', 'L/R Footedness %'],
        'Defensive LB': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DLB)', 'Top 5 PSV-99 (DLB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DLB)', 'PAdj Tackles (DLB)', 'PAdj Interceptions (DLB)', 'OBV Dribble & Carry (DLB)', 'OBV Defensive Action (DLB)', 'Tackle / Dribbled Past % (DLB)', 'PAdj Pressures (DLB)', 'Dribbled Past (DLB)', 'Aerial Win % (DLB)', 'L/R Footedness %'],
        'Attacking RB': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ARB)', 'Top 5 PSV-99 (ARB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ARB)', 'PAdj Tackles (ARB)', 'PAdj Interceptions (ARB)', 'OBV Pass (ARB)', 'OBV Dribble & Carry (ARB)', 'Tackle / Dribbled Past % (ARB)', 'Threat of Runs (ARB)', 'Successful Crosses (ARB)', 'L/R Footedness %'],
        'Defensive RB': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DRB)', 'Top 5 PSV-99 (DRB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DRB)', 'PAdj Tackles (DRB)', 'PAdj Interceptions (DRB)', 'OBV Dribble & Carry (DRB)', 'OBV Defensive Action (DRB)', 'Tackle / Dribbled Past % (DRB)', 'PAdj Pressures (DRB)', 'Dribbled Past (DRB)', 'Aerial Win % (DRB)', 'L/R Footedness %'],
        'Ball Playing Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (BCB)', 'Top 5 PSV-99 (BCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (BCB)', 'PAdj Interceptions (BCB)', 'PAdj Tackles (BCB)', 'OBV Pass (BCB)', 'OBV Dribble & Carry (BCB)', 'OBV Defensive Action (BCB)', 'Deep Progressions (BCB)', 'Pressured Change in Passing % (BCB)', 'L/R Footedness %'],
        'Dominant Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Primary Position', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DCB)', 'Top 5 PSV-99 (DCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (DCB)', 'Aerial Win % (DCB)', 'OBV Defensive Action (DCB)', 'Tackle / Dribbled Past % (DCB)', 'Blocks Per Shot (DCB)', 'L/R Footedness %'],
    }

    # Update the selected columns to include 'Score Type' and 'Season'
    selected_columns = score_type_column_mapping.get(selected_score_type, [])

    # Modify the filtering condition to include selected primary positions
    # Modify the filtering condition for the 'Season' column to treat '2023' as a single entity
    filtered_df = df2[
      (df2['League'].isin(selected_leagues)) &
      (df2['Season'].astype(str) == str(selected_season)) &  # Convert to string for exact match
      (df2['Score Type'] == selected_score_type) &
      (df2['Age'] >= age_range[0]) &
      (df2['Age'] <= age_range[1]) &
      (df2['Contract expires'].isin(selected_contract_expiry_years)) &
      (df2['Market value (millions)'] >= player_market_value_range[0]) &
      (df2['Market value (millions)'] <= player_market_value_range[1]) &
      (df2['Stoke Score'] >= stoke_range[0]) &
      (df2['Stoke Score'] <= stoke_range[1]) &
      (df2[selected_columns[6]].ge(avg_distance_percentile_range[0])) &
      (df2[selected_columns[6]].le(avg_distance_percentile_range[1])) &
      (df2[selected_columns[7]].ge(top_5_psv_99_percentile_range[0])) &
      (df2[selected_columns[7]].le(top_5_psv_99_percentile_range[1])) &
      (df2['L/R Footedness %'].ge(lr_footedness_range[0])) &
      (df2['L/R Footedness %'].le(lr_footedness_range[1])) &
      (df2['Primary Position'].isin(selected_primary_positions))  # Include selected primary positions
  ]

    # Sort the filtered DataFrame by "Stoke Score" column in descending order
    filtered_df = filtered_df.sort_values(by='Stoke Score', ascending=False)

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

    df2 = df2[df2['League'] != 'Band 2 Leagues']

    # Define the allowed score types
    allowed_score_types = ["Striker", "Winger", "Attacking Midfield", "Central Midfield", "Defensive Midfield", "Left Back", "Right Back", "Centre Back", "Stretch 9"]

    # Select player 1
    selected_player_1 = st.sidebar.selectbox(
        "Select Player 1 (Blue):",
        options=df2["Player Name"].unique(),
        index=0  # Set the default index to the first player
    )

    # Filter available players for Player 2 based on the 'Score Type' of Player 1
    available_players_2 = df2[df2["Score Type"] == df2[df2["Player Name"] == selected_player_1]["Score Type"].values[0]]["Player Name"].unique()

    # Select player 2
    selected_player_2 = st.sidebar.selectbox(
        "Select Player 2 (Pink):",
        options=available_players_2,
        index=1  # Set the default index to the second player
    )

    # Player 1 DataFrame
    selected_player_df_1 = df2[df2["Player Name"] == selected_player_1]

    # Player 2 DataFrame
    selected_player_df_2 = df2[df2["Player Name"] == selected_player_2]

    # Profile options based on Player 1
    profile_options = selected_player_df_1[selected_player_df_1["Score Type"].isin(allowed_score_types)]["Score Type"].unique()

    # Default profile selection
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
        columns_1 = ["Player Name", "xG (W)", "Non-Penalty Goals (W)", "Shots (W)", "Open Play xA (W)", "OBV Pass (W)", "Successful Dribbles (W)", "OBV Dribble & Carry (W)", "Distance (W)", "Top 5 PSV (W)"]
        plot_title_1 = f"Winger Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "xG (W)", "Non-Penalty Goals (W)", "Shots (W)", "Open Play xA (W)", "OBV Pass (W)", "Successful Dribbles (W)", "OBV Dribble & Carry (W)", "Distance (W)", "Top 5 PSV (W)"]
        plot_title_2 = f"Winger Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Attacking Midfield":
        columns_1 = ["Player Name", "xG (CAM)", "Non-Penalty Goals (CAM)", "Shots (CAM)", "OBV Pass (CAM)", "Open Play xA (CAM)", "Throughballs (CAM)", "Successful Dribbles (CAM)", "OBV Dribble & Carry (CAM)", "Average Distance (CAM)", "Top 5 PSV (CAM)"]
        plot_title_1 = f"Attacking Midfield Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "xG (CAM)", "Non-Penalty Goals (CAM)", "Shots (CAM)", "OBV Pass (CAM)", "Open Play xA (CAM)", "Throughballs (CAM)", "Successful Dribbles (CAM)", "OBV Dribble & Carry (CAM)", "Average Distance (CAM)", "Top 5 PSV (CAM)"]
        plot_title_2 = f"Attacking Midfield Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Central Midfield":
        columns_1 = ["Player Name", "xG (8)", "Non-Penalty Goals (8)", "OBV Pass (8)", "Open Play xA (8)", "Deep Progressions (8)", "Successful Dribbles (8)", "OBV Dribble & Carry (8)", "Average Distance (8)", "Top 5 PSV-99 (8)"]
        plot_title_1 = f"Attacking Midfield Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "xG (8)", "Non-Penalty Goals (8)", "OBV Pass (8)", "Open Play xA (8)", "Deep Progressions (8)", "Successful Dribbles (8)", "OBV Dribble & Carry (8)", "Average Distance (8)", "Top 5 PSV-99 (8)"]
        plot_title_2 = f"Attacking Midfield Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Defensive Midfield":
        columns_1 = ["Player Name", "OBV Pass (6)", "OBV Dribble & Carry (6)", "Pass Forward % (6)", "PAdj Pressures (6)", "PAdj Tackles & Interceptions (6)", "Tackle/Dribbled Past % (6)", "Ball Recoveries (6)", "Average Distance (6)", "Top 5 PSV-99 (6)"]
        plot_title_1 = f"Defensive Midfield Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "OBV Pass (6)", "OBV Dribble & Carry (6)", "Pass Forward % (6)", "PAdj Pressures (6)", "PAdj Tackles & Interceptions (6)", "Tackle/Dribbled Past % (6)", "Ball Recoveries (6)", "Average Distance (6)", "Top 5 PSV-99 (6)"]
        plot_title_2 = f"Defensive Midfield Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Left Back":
        columns_1 = ["Player Name", "PAdj Tackles & Interceptions (LB)", "Tackle/Dribbled Past (LB)", "OBV Defensive Action (LB)", "Dribbled Past (LB)", "OBV Dribble & Carry (LB)", "Successful Crosses (LB)", "Open Play xA (LB)", "OBV Pass (LB)", "Aerial Win % (LB)", "Average Distance (LB)", "Top 5 PSV-99 (LB)"]
        plot_title_1 = f"Left Back Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "PAdj Tackles & Interceptions (LB)", "Tackle/Dribbled Past (LB)", "OBV Defensive Action (LB)", "Dribbled Past (LB)", "OBV Dribble & Carry (LB)", "Successful Crosses (LB)", "Open Play xA (LB)", "OBV Pass (LB)", "Aerial Win % (LB)", "Average Distance (LB)", "Top 5 PSV-99 (LB)"]
        plot_title_2 = f"Left Back Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Right Back":
        columns_1 = ["Player Name", "PAdj Tackles & Interceptions (RB)", "Tackle/Dribbled Past (RB)", "OBV Defensive Action (RB)", "Dribbled Past (RB)", "OBV Dribble & Carry (RB)", "Successful Crosses (RB)", "Open Play xA (RB)", "OBV Pass (RB)", "Aerial Win % (RB)", "Average Distance (RB)", "Top 5 PSV-99 (RB)"]
        plot_title_1 = f"Right Back Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "PAdj Tackles & Interceptions (RB)", "Tackle/Dribbled Past (RB)", "OBV Defensive Action (RB)", "Dribbled Past (RB)", "OBV Dribble & Carry (RB)", "Successful Crosses (RB)", "Open Play xA (RB)", "OBV Pass (RB)", "Aerial Win % (RB)", "Average Distance (RB)", "Top 5 PSV-99 (RB)"]
        plot_title_2 = f"Right Back Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Centre Back":
        columns_1 = ["Player Name", "Aerial Wins (CB)", "Aerial Win % (CB)", "PAdj Tackles & Interceptions (CB)", "Tackle / Dribbled Past % (CB)", "OBV Defensive Action (CB)", "Blocks per Shot (CB)", "Deep Progressions (CB)", "OBV Pass (CB)", "Pressure Change in Passing % (CB)", "OBV Dribble & Carry (CB)", "Top 5 PSV-99 (CB)"]
        plot_title_1 = f"Centre Back Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "Aerial Wins (CB)", "Aerial Win % (CB)", "PAdj Tackles & Interceptions (CB)", "Tackle / Dribbled Past % (CB)", "OBV Defensive Action (CB)", "Blocks per Shot (CB)", "Deep Progressions (CB)", "OBV Pass (CB)", "Pressure Change in Passing % (CB)", "OBV Dribble & Carry (CB)", "Top 5 PSV-99 (CB)"]
        plot_title_2 = f"Centre Back Metric Percentiles for {selected_player_2}"

    elif selected_profile == "Stretch 9":
        columns_1 = ["Player Name", "xG (S9)", "Non-Penalty Goals (S9)", "Shots (S9)", "OBV Shot (S9)", "Open Play xA (S9)", "OBV Dribble & Carry (S9)", "Top 5 PSV-99 (S9)", "Runs in Behind (S9)", "Threat of Runs in Behind (S9)"]
        plot_title_1 = f"Stretch 9 Metric Percentiles for {selected_player_1}"

        columns_2 = ["Player Name", "xG (S9)", "Non-Penalty Goals (S9)", "Shots (S9)", "OBV Shot (S9)", "Open Play xA (S9)", "OBV Dribble & Carry (S9)", "Top 5 PSV-99 (S9)", "Runs in Behind (S9)", "Threat of Runs in Behind (S9)"]
        plot_title_2 = f"Stretch 9 Metric Percentiles for {selected_player_2}"

    # Filter DataFrames based on the selected profile
    selected_df_1 = selected_player_df_1[selected_player_df_1["Score Type"] == selected_profile]
    selected_df_2 = selected_player_df_2[selected_player_df_2["Score Type"] == selected_profile]

    # Get columns for percentiles
    percentiles_df_1 = selected_df_1[columns_1]
    percentiles_df_2 = selected_df_2[columns_2]

    # Melt DataFrames for PyPizza
    percentiles_df_1 = percentiles_df_1.melt(id_vars="Player Name", var_name="Percentile Type", value_name="Percentile")
    percentiles_df_2 = percentiles_df_2.melt(id_vars="Player Name", var_name="Percentile Type", value_name="Percentile")

    # Load the Roboto font
    font_path = "Roboto-Bold.ttf"  # Replace with the actual path to the Roboto font
    prop = font_manager.FontProperties(fname=font_path)
    font_path1 = "Roboto-Regular.ttf"
    prop1 = font_manager.FontProperties(fname=font_path1)

    # Create PyPizza plot
    col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
    with col3:
        params = percentiles_df_1["Percentile Type"]
        values1 = percentiles_df_1["Percentile"]

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
            values1,
            compare_values=percentiles_df_2["Percentile"].tolist(),
            figsize=(8, 8),
            kwargs_slices=dict(
                facecolor="#7EC0EE", edgecolor="#222222",
                zorder=1, linewidth=1
            ),
            kwargs_compare=dict(
                facecolor="#FF34B3", edgecolor="#222222",
                zorder=2, linewidth=1,
            ),
            kwargs_params=dict(
                color="#000000", fontsize=8,
                va="center"
            ),
            kwargs_values=dict(
                color="#000000", fontsize=12,
                zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="#7EC0EE",
                    boxstyle="round,pad=0.2", lw=1
                )
            ),
            kwargs_compare_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(edgecolor="#000000", facecolor="#FF34B3", boxstyle="round,pad=0.2", lw=1)
            ),
        )

        st.pyplot(fig)

# Function to calculate similarity against 'Striker' profiles
def calculate_similarity(selected_df, df2, columns):
    # Exclude the "Player Name" column
    selected_metrics = selected_df[columns[1:]].select_dtypes(include='number').values
    
    # Filter the DataFrame to include only 'Striker' profiles
    striker_df = df2[df2['Score Type'] == 'Striker'][columns[1:]].select_dtypes(include='number').values
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(selected_metrics, striker_df)
    
    # Create a DataFrame with similarity scores
    similarity_df = pd.DataFrame(similarity_matrix, index=selected_df["Player Name"], columns=df2[df2['Score Type'] == 'Striker']["Player Name"])
    
    return similarity_df

# Main function for the Streamlit app
def similarity_score(df2):

    allowed_score_types = ["Striker", "Winger", "Stretch 9", "Attacking Midfield", "Central Midfield", "Defensive Midfield", "Left Back", "Right Back", "Centre Back"]  # Add other score types as needed

    # Select a player and profile
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
        columns = ["Player Name", "xG (ST)", "Non-Penalty Goals (ST)", "Shots (ST)", "OBV Shot (ST)", "Open Play xA (ST)", "OBV Dribble & Carry (ST)", "PAdj Pressures (ST)", "Average Distance Percentile", "Top 5 PSV-99 Percentile"]
        plot_title = f"Forward Metrics for {selected_player}"
    elif selected_profile == "Winger":
        columns = ["Player Name", "xG (W)", "Non-Penalty Goals (W)", "Shots (W)", "OBV Pass (W)", "Open Play xA (W)",  "Successful Dribbles (W)", "OBV Dribble & Carry (W)", "Distance (W)", "Top 5 PSV (W)"]
        plot_title = f"Winger Metric Percentiles for {selected_player}"
    elif selected_profile == "Attacking Midfield":
        columns = ["Player Name", "xG (CAM)", "Non-Penalty Goals (CAM)", "Shots (CAM)", "OBV Pass (CAM)", "Open Play xA (CAM)", "Throughballs (CAM)", "Successful Dribbles (CAM)", "OBV Dribble & Carry (CAM)", "Average Distance (CAM)", "Top 5 PSV (CAM)"]
        plot_title = f"Attacking Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Central Midfield":
        columns = ["Player Name", "xG (8)", "Non-Penalty Goals (8)", "OBV Pass (8)", "Open Play xA (8)", "Deep Progressions (8)", "Successful Dribbles (8)", "OBV Dribble & Carry (8)", "Average Distance (8)", "Top 5 PSV-99 (8)"]
        plot_title = f"Central Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Defensive Midfield":
        columns = ["Player Name", "Deep Progressions (6)", "OBV Pass (6)", "OBV Dribble & Carry (6)", "Pass Forward % (6)", "PAdj Pressures (6)", "Pressure Regains (6)", "PAdj Tackles & Interceptions (6)", "Tackle/Dribbled Past % (6)", "OBV Defensive Action (6)", "Ball Recoveries (6)", "Average Distance (6)", "Top 5 PSV-99 (6)"]
        plot_title = f"Defensive Midfield Metric Percentiles for {selected_player}"
    elif selected_profile == "Left Back":
        columns = ["Player Name", "PAdj Tackles & Interceptions (LB)", "Tackle/Dribbled Past (LB)", "OBV Defensive Action (LB)", "Dribbled Past (LB)", "OBV Dribble & Carry (LB)", "Successful Crosses (LB)", "Open Play xA (LB)", "OBV Pass (LB)", "Aerial Win % (LB)", "Average Distance (LB)", "Top 5 PSV-99 (LB)"]
        plot_title = f"Left Back Metric Percentiles for {selected_player}"
    elif selected_profile == "Right Back":
        columns = ["Player Name", "PAdj Tackles & Interceptions (RB)", "Tackle/Dribbled Past (RB)", "OBV Defensive Action (RB)", "Dribbled Past (RB)", "OBV Dribble & Carry (RB)", "Successful Crosses (RB)", "Open Play xA (RB)", "OBV Pass (RB)", "Aerial Win % (RB)", "Average Distance (RB)", "Top 5 PSV-99 (RB)"]
        plot_title = f"Right Back Metric Percentiles for {selected_player}"
    elif selected_profile == "Centre Back":
        columns = ["Player Name", "Aerial Wins (CB)", "Aerial Win % (CB)", "PAdj Tackles & Interceptions (CB)", "Tackle / Dribbled Past % (CB)", "OBV Defensive Action (CB)", "Blocks per Shot (CB)", "Deep Progressions (CB)", "OBV Pass (CB)", "Pressure Change in Passing % (CB)", "OBV Dribble & Carry (CB)", "Top 5 PSV-99 (CB)"]
        plot_title = f"Centre Back Metric Percentiles for {selected_player}"
    elif selected_profile == "Stretch 9":
        columns = ["Player Name", "xG (S9)", "Non-Penalty Goals (S9)", "Shots (S9)", "OBV Shot (S9)", "Open Play xA (S9)", "Runs in Behind (S9)", "Threat of Runs in Behind (S9)", "Average Distance (S9)", "Top 5 PSV-99 (S9)"]
        plot_title = f"Stretch 9 Metric Percentiles for {selected_player}"
    else:
        # Define columns and plot title for the default profile
        columns = []
        plot_title = f"Default Profile Metrics for {selected_player}"

    # Assuming selected_df is your DataFrame containing the data
    selected_df = selected_player_df[selected_player_df["Score Type"] == selected_profile][columns[0:]]  # Exclude the "Player Name" column

    # Display selected DataFrame details
    #st.subheader("Selected DataFrame Details")
    #st.write(selected_df)

    # Extract only the metrics used in the pizza visualization for similarity calculation
    selected_metrics = selected_df.select_dtypes(include='number').values

    # Load the Roboto font
    font_path = "Roboto-Bold.ttf"  # Replace with the actual path to the Roboto font
    prop = font_manager.FontProperties(fname=font_path)
    font_path1 = "Roboto-Regular.ttf"
    prop1 = font_manager.FontProperties(fname=font_path1)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])

    with col3:
        params = selected_df.columns[1:]
        values1 = selected_df.iloc[0, 1:]  # Assuming you want metrics for the first player

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
                color="#000000", fontsize=8, va="center", 
            ),
            kwargs_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="#7EC0EE",
                    boxstyle="round,pad=0.2", lw=1
                ),
    
            ),
            kwargs_compare_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(edgecolor="#000000", facecolor="#7EC0EE", boxstyle="round,pad=0.2", lw=1),
                weight="bold"
            )
        )

        st.pyplot(fig2)

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

        # Create a multi-select dropdown for selecting leagues with 'English Championship' pre-selected
        default_leagues = ['English Championship']
        selected_leagues = st.sidebar.multiselect('Select Leagues', df['League'].unique(), default=default_leagues)

        # Sidebar for filtering by 'minutes' played
        min_minutes = int(df['Player Season Minutes'].min())
        max_minutes = int(df['Player Season Minutes'].max())
        selected_minutes = st.sidebar.slider('Select Minutes Played Range', min_value=min_minutes, max_value=max_minutes, value=(300, max_minutes))

        # Filter data based on user-selected positions, minutes played, and leagues
        filtered_df = df[(df['position_1'].isin(selected_positions) | (len(selected_positions) == 0)) &
                         (df['Player Season Minutes'] >= selected_minutes[0]) &
                         (df['Player Season Minutes'] <= selected_minutes[1]) &
                         (df['League'].isin(selected_leagues) | (len(selected_leagues) == 0))]

        # Calculate Z-scores for the variables
        filtered_df['z_x'] = (filtered_df[x_variable] - filtered_df[x_variable].mean()) / filtered_df[x_variable].std()
        filtered_df['z_y'] = (filtered_df[y_variable] - filtered_df[y_variable].mean()) / filtered_df[y_variable].std()

        # Define a threshold for labeling outliers (you can customize this threshold)
        threshold = st.sidebar.slider('Label Threshold', min_value=0.1, max_value=5.0, value=2.0)

        # Create a scatter plot using Plotly with the filtered data
        hover_data_fields = {'Player Name': True, 'Team': True, 'Age': True, 'Player Season Minutes': True, x_variable: False, y_variable: False, 'z_x': False, 'z_y': False}
        fig = px.scatter(filtered_df, x=x_variable, y=y_variable, hover_data=hover_data_fields)

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

        # Create a multi-select dropdown for selecting players
        selected_players = st.sidebar.multiselect('Select Players', filtered_df['Player Name'].unique())

        # Create a trace for selected players and customize hover labels
        if selected_players:
            selected_df = filtered_df[filtered_df['Player Name'].isin(selected_players)]
            selected_trace = go.Scatter(
                x=selected_df[x_variable],
                y=selected_df[y_variable],
                mode='markers+text',  # Combine markers and text
                marker=dict(size=12, color='red'),
                name='Selected Players',
                text=selected_df['Player Name'],  # Display player name as text label
                textposition='top center'
            )

            # Customize hover data for selected trace
            hover_data_fields_selected = {'Player Name': True, 'Team': True, 'Age': True, 'Minutes': True, x_variable: False, y_variable: False, 'z_x': False, 'z_y': False}
            fig.add_trace(selected_trace).update_traces(hoverinfo="text+x+y")

        # Display the plot in Streamlit
        st.plotly_chart(fig)
    
def comparison_tab(df):

    # Filter the DataFrame based on selected players
    selected_players = st.sidebar.multiselect("Select Players", df["Player Name"])
    
    # Add a filter for Player Season Minutes with a default minimum value of 500 minutes
    min_minutes = st.sidebar.slider("Minimum Player Season Minutes", min_value=0, max_value=int(df["Player Season Minutes"].max()), value=500)
    
    # Sidebar: Metric selection
    selected_metrics = st.sidebar.multiselect("Select Metrics", df.columns[1:])

    # Add a "Total" option for selected metrics
    total_option = st.sidebar.checkbox("Total", key="total_checkbox")

    # Metrics to exclude from total calculation
    exclude_from_total = ['Top 5 PSV-99']

    # Remove excluded metrics from selected metrics if total option is selected
    if total_option:
        selected_metrics = [metric for metric in selected_metrics if metric not in exclude_from_total]

    def highlight_best_player(s):
        is_best = s == s.max()
        return ['background-color: #00CD00' if v else '' for v in is_best]

    # Create a new DataFrame for calculated totals
    calculated_df = df.copy()

    # Filter DataFrame based on selected players and minimum minutes
    filtered_df = calculated_df[(calculated_df["Player Name"].isin(selected_players)) & (calculated_df["Player Season Minutes"] >= min_minutes)]

    # Calculate totals if the "Total" checkbox is selected
    if total_option:
        selected_metrics_without_minutes = [metric for metric in selected_metrics]
        filtered_df[selected_metrics_without_minutes] = filtered_df[selected_metrics_without_minutes].multiply((filtered_df["Player Season Minutes"]/90), axis="index")

    # Display the table with conditional formatting
    if selected_metrics:
        if filtered_df.empty:
            st.warning("No players selected. Please select at least one player.")
        else:
            selected_columns =  ["Player Name"] + ["Player Season Minutes"] + ["Team"] + selected_metrics
            if total_option:
                formatted_df = filtered_df[selected_columns].copy()
            else:
                formatted_df = filtered_df[selected_columns].copy()
            formatted_df = formatted_df.style.apply(highlight_best_player, subset=selected_metrics)
            # Format numbers to two decimal places
            formatted_df = formatted_df.format({"Player Season Minutes": "{:.0f}"}, subset=["Player Season Minutes"])
            formatted_df = formatted_df.format("{:.2f}", subset=selected_metrics)
            st.dataframe(formatted_df, hide_index=True)
    else:
        st.warning("Select at least one metric to compare.")

def calculate_similarity(player1, player2, columns, feature_importance):
    metrics1 = player1[columns].fillna(0).values * np.array([feature_importance.get(col, 0.5) for col in columns])
    metrics2 = player2[columns].fillna(0).values * np.array([feature_importance.get(col, 0.5) for col in columns])
    return np.linalg.norm(metrics1 - metrics2)

def rescale_similarity(x, max_val):
    return 100 - x * (100 / max_val)

def player_similarity_app(df2):
    # Dictionary to map positions to selectable additional metrics
    position_additional_metrics = {
        'Striker': [],
        'Winger': [],
        'Attacking Midfield': [],
        'Central Midfield': [],
        'Defensive Midfield': [],
        'Left Back': [],
        'Right Back': [],
        'Centre Back': [],
        'Stretch 9': []
    }

    # Dictionary to map positions to base metrics
    position_base_metrics = {
        'Striker': ['Non-Penalty Goals (ST)', 'Shots (ST)', 'OBV Shot (ST)', 'Open Play xA (ST)', 'Aerial Wins (ST)', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile'],
        'Winger': ['Non-Penalty Goals (W)', 'Shots (W)', 'Open Play xA (W)', 'OBV Pass (W)', 'Successful Dribbles (W)', 'OBV Dribble & Carry (W)', 'Distance (W)', 'Top 5 PSV (W)'],
        'Attacking Midfield': ['xG (CAM)', 'Non-Penalty Goals (CAM)', 'Shots (CAM)', 'Open Play xA (CAM)', 'OBV Pass (CAM)', 'Successful Dribbles (CAM)', 'OBV Dribble & Carry (CAM)', 'Average Distance (CAM)', 'Top 5 PSV (CAM)'],
        'Central Midfield': ['xG (8)', 'Non-Penalty Goals (8)', 'OBV Pass (8)', 'Open Play xA (8)', 'Successful Dribbles (8)', 'OBV Dribble & Carry (8)', 'Average Distance (8)', 'Top 5 PSV-99 (8)', 'PAdj Tackles & Interceptions (8)', 'Deep Progressions (8)'],
        'Defensive Midfield': ['Average Distance (6)', 'Top 5 PSV-99 (6)', 'OBV Defensive Action (6)', 'OBV Pass (6)', 'Deep Progressions (6)', 'Successful Dribbles (6)', 'OBV Dribble & Carry (6)', 'Tackle/Dribbled Past % (6)', 'PAdj Tackles & Interceptions (6)', 'Pass Forward % (6)', 'Turnovers (6)', 'PAdj Pressures (6)', 'Pressure Regains (6)', 'Ball Recoveries (6)'],
        'Left Back': ['Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'OBV Defensive Action (LB)', 'OBV Dribble & Carry (LB)', 'Tackle/Dribbled Past (LB)', 'Open Play xA (LB)', 'Successful Crosses (LB)', 'Dribbled Past (LB)', 'Successful Dribbles (LB)', 'OBV Pass (LB)', 'PAdj Tackles & Interceptions (LB)', 'Aerial Win % (LB)'],
        'Right Back': ['Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'OBV Defensive Action (RB)', 'OBV Dribble & Carry (RB)', 'Tackle/Dribbled Past (RB)', 'Open Play xA (RB)', 'Successful Crosses (RB)', 'Dribbled Past (RB)', 'Successful Dribbles (RB)', 'OBV Pass (RB)', 'PAdj Tackles & Interceptions (RB)', 'Aerial Win % (RB)'],
        'Centre Back': ['Top 5 PSV-99 (CB)', 'Aerial Win % (CB)', 'Aerial Wins (CB)', 'OBV Pass (CB)', 'OBV Dribble & Carry (CB)', 'OBV Defensive Action (CB)', 'Deep Progressions (CB)', 'PAdj Tackles & Interceptions (CB)', 'Tackle / Dribbled Past % (CB)', 'Blocks per Shot (CB)', 'Pressure Change in Passing % (CB)'],
        'Stretch 9': ['xG (S9)', 'Non-Penalty Goals (S9)', 'Shots (S9)', 'OBV Shot (S9)', 'Open Play xA (S9)', 'OBV Dribble & Carry (S9)', 'PAdj Pressures (S9)', 'Aerial Wins (S9)', 'Aerial Win % (S9)', 'Average Distance (S9)', 'Top 5 PSV-99 (S9)', 'Runs in Behind (S9)', 'Threat of Runs in Behind (S9)']
    }

    # Add a sidebar dropdown for selecting a player name
    player_name = st.sidebar.selectbox("Select a player's name:", df2['Player Name'].unique())
    
    # Add a sidebar radio button for selecting a position to compare
    position_to_compare = st.sidebar.radio("Select a position to compare:", ('Stretch 9', 'Winger', 'Attacking Midfield', 'Central Midfield', 'Defensive Midfield', 'Left Back', 'Right Back', 'Centre Back'))
    
    # Filter unique positions based on the selected position for similarity calculation
    available_positions = df2[df2['Score Type'] == position_to_compare]['Position'].unique()
    
    # Add a slider to filter players by age
    max_age = st.sidebar.slider("Select maximum age:", min_value=18, max_value=40, value=30)

    # Add a slider to filter players by 'Player Season Minutes'
    min_minutes = st.sidebar.slider("Select minimum 'Player Season Minutes':", min_value=0, max_value=int(df2['Player Season Minutes'].max()), value=0)

    # Add two sliders to filter players by 'L/R Footedness %' (minimum and maximum)
    min_lr_footedness = st.sidebar.slider("Select minimum 'L/R Footedness %':", min_value=0, max_value=100, value=0)
    max_lr_footedness = st.sidebar.slider("Select maximum 'L/R Footedness %':", min_value=0, max_value=100, value=100)

    # Automatically select all leagues by default
    selected_leagues = df2['League'].unique()

    # Filter unique leagues based on the selected position and filters
    filtered_leagues = df2[(df2['Score Type'] == position_to_compare) & (df2['Age'] <= max_age) & (df2['Player Season Minutes'] >= min_minutes) & (df2['L/R Footedness %'] >= min_lr_footedness) & (df2['L/R Footedness %'] <= max_lr_footedness)]['League'].unique()

    # Set the default value for selected_leagues based on availability
    if all(league in filtered_leagues for league in selected_leagues):
        default_selected_leagues = selected_leagues
    else:
        default_selected_leagues = filtered_leagues

    # Add a multi-select dropdown for filtering by 'League' with default value
    selected_leagues = st.sidebar.multiselect("Select leagues:", filtered_leagues, default=default_selected_leagues)

    # Add a multi-select dropdown for selecting a position filter with default value as all available positions
    selected_positions = st.sidebar.multiselect("Select position filters:", available_positions, default=available_positions)

    # Check if the selected player is in the dataset
    if player_name in df2['Player Name'].values:

        # Choose the reference player
        reference_player = player_name

        # Get additional metrics for the selected position
        additional_metrics = position_additional_metrics.get(position_to_compare, [])

        # Get base metrics for the selected position
        base_metrics = position_base_metrics.get(position_to_compare, [])

        # Combine base and additional metrics for the multiselect dropdown
        all_metric_options = base_metrics + additional_metrics

        # Generate keys for all metric options
        all_metric_keys = [f"{position_to_compare}_metric_{metric}" for metric in all_metric_options]

        # Multiselect dropdown for selecting base and additional metrics
        selected_metrics = st.sidebar.multiselect("Select base and additional metrics:", all_metric_options, default=base_metrics, key=f"{position_to_compare}_all_metrics")

        # Add a multiselect dropdown for adjusting feature importance with unique key
        feature_importance = {}
        st.sidebar.header("Feature Importance")
        for metric, key in zip(all_metric_options, all_metric_keys):
            feature_importance[metric] = st.sidebar.slider(f"Importance of {metric}:", min_value=0.0, max_value=1.0, value=0.5, key=key)

        # Calculate similarity scores for all players within the age, minutes, and league bracket
        similarities = {}
        reference_player_data = df2[(df2['Player Name'] == reference_player) & (df2['Score Type'] == position_to_compare)].iloc[0]

        # Find the maximum similarity score for scaling
        max_similarity = float('-inf')

        for _, player in df2.iterrows():
            if (
                player['Player Name'] != reference_player
                and player['Age'] <= max_age
                and player['Score Type'] == position_to_compare
                and player['Player Season Minutes'] >= min_minutes
                and player['League'] in selected_leagues
                and player['Position'] in selected_positions
                and (player['L/R Footedness %'] >= min_lr_footedness and player['L/R Footedness %'] <= max_lr_footedness)
            ):
                similarity_score = calculate_similarity(
                    reference_player_data,
                    player,
                    selected_metrics,  # Exclude the first three columns (Player Name, Team, Age)
                    feature_importance  # Pass feature importance to calculate_similarity
                )
                similarities[player['Player Name']] = similarity_score

                # Update max similarity score
                max_similarity = max(max_similarity, similarity_score)

        # Normalize similarity scores to the range [0, 100]
        for player_name, similarity_score in similarities.items():
            normalized_similarity = rescale_similarity(similarity_score, max_similarity)
            similarities[player_name] = normalized_similarity

        # Sort players by similarity score (descending)
        similar_players = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Display the top 50 most similar players within the selected age, minutes, and league bracket
        st.header(f"Most similar {position_to_compare}s to {reference_player} (Age <= {max_age}, Minutes >= {min_minutes}, L/R Footedness >= {min_lr_footedness}%, L/R Footedness <= {max_lr_footedness}%):")
        similar_players_df = pd.DataFrame(similar_players, columns=['Player Name', 'Similarity Score'])
        
        # Add 'Player Club', 'Age', 'Player Season Minutes', 'League', and 'L/R Footedness %' columns to the DataFrame
        similar_players_df = pd.merge(similar_players_df, df2[['Player Name', 'Team', 'Age', 'Player Season Minutes', 'League', 'Position', 'L/R Footedness %']], on='Player Name', how='left')
        
        # Remove duplicates in case of multiple matches in the age, minutes, and league filter
        similar_players_df = similar_players_df.drop_duplicates(subset='Player Name')
        
        st.dataframe(similar_players_df.head(250))
    else:
        st.error("Player not found in the selected position.")

def player_stat_search(df):

    # Sidebar for filtering by 'minutes' played
    min_minutes = int(df['Player Season Minutes'].min())
    max_minutes = int(df['Player Season Minutes'].max())
    selected_minutes = st.sidebar.slider('Select Minutes Played Range', min_value=min_minutes, max_value=max_minutes, value=(300, max_minutes))

    # Sidebar for filtering by 'age'
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    selected_age = st.sidebar.slider('Select Age Range', min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Create a multi-select dropdown for filtering by primary_position
    selected_positions = st.sidebar.multiselect('Filter by Primary Position', df['position_1'].unique())

    # Create a multi-select dropdown for selecting leagues with 'English Championship' pre-selected
    default_leagues = ['English Championship']
    selected_leagues = st.sidebar.multiselect('Select Leagues', df['League'].unique(), default=default_leagues)

    # Get the list of all columns in the DataFrame
    all_columns = df.columns.tolist()

    # Ensure that these columns are always included in selected_stats
    always_included_columns = ["Player Name", "Age", "Team", "Player Season Minutes", "League"]
    
    # Create a multiselect for stat selection
    selected_stats = st.multiselect("Select Columns", [col for col in all_columns if col not in always_included_columns], default=[])

    # Add the always included columns to the selected_stats
    selected_stats.extend(always_included_columns)

    # Filter the DataFrame based on selected filters
    filtered_df = df[(df['Player Season Minutes'] >= selected_minutes[0]) & (df['Player Season Minutes'] <= selected_minutes[1])]
    filtered_df = filtered_df[(df['Age'] >= selected_age[0]) & (df['Age'] <= selected_age[1])]
    if selected_positions:
        filtered_df = filtered_df[filtered_df['position_1'].isin(selected_positions)]
    if selected_leagues:
        filtered_df = filtered_df[filtered_df['League'].isin(selected_leagues)]

    # Compute minimum and maximum values for selected stats based on filtered players
    stat_min_max = {}
    for stat in selected_stats:
        if stat not in always_included_columns:
            min_stat = float(filtered_df[stat].min())  # Convert numpy.float64 to Python float
            max_stat = float(filtered_df[stat].max())  # Convert numpy.float64 to Python float
            stat_min_max[stat] = (min_stat, max_stat)

    # Create sliders for selected_stats using computed min and max values
    slider_filters = {}
    for stat, (min_val, max_val) in stat_min_max.items():
        slider_filters[stat] = st.sidebar.slider(f'Select {stat} Range', min_value=min_val, max_value=max_val, value=(min_val, max_val))

    # Apply filters based on selected stat sliders
    for stat, (min_val, max_val) in slider_filters.items():
        filtered_df = filtered_df[(filtered_df[stat] >= min_val) & (filtered_df[stat] <= max_val)]

    # Display the customized table with 'Age' as a constant column without index numbering
    selected_stats_ordered = always_included_columns + [col for col in selected_stats if col not in always_included_columns]
    st.dataframe(filtered_df[selected_stats_ordered], hide_index=True)

def stoke_score_wyscout(df3):
    
    # Create a list of league options
    league_options = df3['League'].unique()
    
    # Create a list of score type options
    score_type_options = df3['Score Type'].unique()

    # Get the minimum and maximum age values from the DataFrame
    min_age = int(df3['Age'].min())
    max_age = int(df3['Age'].max())

    min_stoke_score = 0.0
    max_stoke_score = 100.0

    # Add a sidebar multiselect box for leagues with default selections
    selected_leagues = st.sidebar.multiselect("Select Leagues", league_options, default=['Super League Greece'])

    # Add a sidebar dropdown box for score types
    selected_score_type = st.sidebar.selectbox("Select a Score Type", score_type_options)

    stoke_range = st.sidebar.slider("Select Stoke Score Range", min_value=min_stoke_score, max_value=max_stoke_score, value=(min_stoke_score, max_stoke_score))
    
    # Add a slider for selecting the age range
    age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Create a list of foot options
    foot_options = df3['foot'].unique()

    # Add a multiselect box for selecting foot
    selected_foot = st.sidebar.multiselect("Select Foot", foot_options, default=foot_options)

    # Add a slider for selecting the range of Aerial duels won, %
    aerial_duels_range = st.sidebar.slider("Select Aerial Duels Won Percentage Range", min_value=0, max_value=100, value=(0, 100))

    # Define a dictionary that maps 'Score Type' to columns
    score_type_column_mapping = {
        'Striker': ['Player', 'Age', 'Team', 'League', 'Position', 'Stoke Score'],
        'Winger': ['Player', 'Age', 'Team', 'League', 'Position', 'Stoke Score'],
        'Attacking Midfield': ['Player', 'Age', 'Team', 'League', 'Position', 'Stoke Score'],
        'Centre Back': ['Player', 'Age', 'Team', 'League', 'Position', 'Stoke Score', 'foot', 'Aerial duels won, %']
    }

    # Update the selected columns to include 'Score Type'
    selected_columns = score_type_column_mapping.get(selected_score_type, [])

     # Modify the filtering condition to include selected primary positions
    filtered_df = df3[
        (df3['League'].isin(selected_leagues)) &
        (df3['Score Type'] == selected_score_type) &
        (df3['Age'] >= age_range[0]) &
        (df3['Age'] <= age_range[1]) &
        (df3['Stoke Score'] >= stoke_range[0]) &
        (df3['Stoke Score'] <= stoke_range[1]) &
        (df3['foot'].isin(selected_foot))  # Filter for foot
    ]

    # Filter for Aerial duels won, % only if the selected position is Centre Back
    if selected_score_type == 'Centre Back':
        filtered_df = filtered_df[
            (filtered_df['Aerial duels won, %'] >= aerial_duels_range[0]) &
            (filtered_df['Aerial duels won, %'] <= aerial_duels_range[1])
        ]

    # Sort the filtered DataFrame by "Stoke Score" column in descending order
    filtered_df = filtered_df.sort_values(by='Stoke Score', ascending=False)

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

# Function to read data from Google Sheets and display it
def display_data():
    # Create a connection object.
    url = "https://docs.google.com/spreadsheets/d/1GAghNSTYJTVVl4I9Q-qOv_PGikuj_TQIgSp2sGXz5XM/edit?usp=sharing"

    conn = st.connection("gsheets", type=GSheetsConnection)

    data = conn.read(spreadsheet=url, usecols=[1, 2, 7, 10, 14, 16, 19, 23, 27, 41, 42, 88])

    # Convert 'Contract' column to a consistent type (e.g., string)
    data['Contract'] = data['Contract'].astype(str)

    # Extract unique entries in 'Contract' column
    unique_contracts = data['Contract'].unique()

    # Set the default selected expiry date to 2030
    selected_expiry_date = '2030'  # Assuming '2030' is a valid contract expiry date in your data

    # Add a sidebar dropdown box for selecting contract expiry date
    selected_expiry_date = st.sidebar.selectbox("Contract Expiry Before", sorted(unique_contracts), index=unique_contracts.tolist().index(selected_expiry_date))

    # Add a sidebar checkbox to select or exclude players from Stoke City
    include_stoke_city = st.sidebar.checkbox("Include Stoke City players", False)

    # Add a sidebar checkbox to select only players available for loan
    only_loan_available = st.sidebar.checkbox("Select only players available for loan", False)

    # Add a sidebar slider for selecting age range
    min_age, max_age = st.sidebar.slider("Select Age Range", min_value=data['Age'].min(), max_value=data['Age'].max(), value=(data['Age'].min(), data['Age'].max()))

    # Add a sidebar checkbox for selecting only domestic players
    domestic_only = st.sidebar.checkbox("Domestic players only?", False, help="Check to include only domestic players")

    # Filter data for players with contract expiry before selected date
    filtered_data = data[data['Contract'] < selected_expiry_date]

    # Filter data to include or exclude Stoke City players based on user choice
    if not include_stoke_city:
        filtered_data = filtered_data[filtered_data['Current Club'] != 'Stoke City']

    # Filter data by age range
    filtered_data = filtered_data[(filtered_data['Age'] >= min_age) & (filtered_data['Age'] <= max_age)]

    # Filter data based on 'Domestic / Abroad' column
    if domestic_only:
        filtered_data = filtered_data[filtered_data['Domestic / Abroad'] == 'Domestic']

    # Filter data to include only players available for loan if checkbox is checked
    if only_loan_available:
        filtered_data = filtered_data[filtered_data['Available for Loan?'] == 'Yes']

    # Filter data for RB, LB, LW, RW, DM, CM, AM, and ST positions
    rb_data = filtered_data[filtered_data['Position'] == 'RB']
    lb_data = filtered_data[filtered_data['Position'] == 'LB']
    lw_data = filtered_data[filtered_data['Position'] == 'LW']
    rw_data = filtered_data[filtered_data['Position'] == 'RW']
    dm_data = filtered_data[filtered_data['Position'] == 'CDM']
    cm_data = filtered_data[filtered_data['Position'] == 'CM']
    am_data = filtered_data[filtered_data['Position'] == 'AM']
    st_data = filtered_data[filtered_data['Position'] == 'CF']
    cb_data = filtered_data[filtered_data['Position'] == 'CB']  # New: Filter data for CBs

    # Filter CBs for left-footed and right-footed players
    left_footed_cb_data = cb_data[cb_data['Foot'] == 'L']
    right_footed_cb_data = cb_data[cb_data['Foot'] == 'R']

    # Select top 5 players for each position based on some criteria (for example, confidence score)
    top_5_rb_players = rb_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_lb_players = lb_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_lw_players = lw_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_rw_players = rw_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_dm_players = dm_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_cm_players = cm_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_am_players = am_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_st_players = st_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_left_footed_cb_players = left_footed_cb_data.sort_values(by='Confidence Score', ascending=False).head(5)  # New: Top 5 left-footed CBs
    top_5_right_footed_cb_players = right_footed_cb_data.sort_values(by='Confidence Score', ascending=False).head(5)  # New: Top 5 right-footed CBs

    # Plot the top 5 players for each position on the pitch visualization
    plot_players_on_pitch(top_5_rb_players, top_5_lb_players, top_5_lw_players, top_5_rw_players, top_5_dm_players, top_5_cm_players, top_5_am_players, top_5_st_players, top_5_left_footed_cb_players, top_5_right_footed_cb_players, filtered_data, data.columns)
        
    # Filter and select desired columns
    selected_columns = ["Player", "Current Club", "Position", "Contract", "Confidence Score Last Month"]
    filtered_data = data[selected_columns]

    # Display a table below the filtered table for original data sorted by 'Confidence Score Last Month'
    st.markdown("**Confidence Score Last Month**")
    sorted_original_data = filtered_data.sort_values(by='Confidence Score Last Month', ascending=False)

    # Display the filtered DataFrame with selected columns
    st.dataframe(sorted_original_data, hide_index=True)

# Plotting function
def plot_players_on_pitch(rb_players_data, lb_players_data, lw_players_data, rw_players_data, dm_players_data, cm_players_data, am_players_data, st_players_data, left_cb_players_data, right_cb_players_data, data, column_names):
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#ffffff', stripe=False, line_zorder=2, pad_top=0.1)

    fig, ax = pitch.draw(figsize=(12, 8))  # Adjust the figsize parameter to make the plot smaller
    ax.patch.set_alpha(1)
    background = "#ffffff"
    fig.set_facecolor(background)

    # Set the X-coordinate of the center of the pitch for each position
    positions_x = {'RB': 58, 'LB': 8, 'LW': 8, 'RW': 58, 'CDM': 33, 'CM': 48, 'AM': 18, 'CF': 33, 'CB': 33}  

    # Set the starting y-coordinate for each position
    start_y = {'RB': 38, 'LB': 38, 'LW': 80, 'RW': 80, 'CDM': 45, 'CM': 65, 'AM': 65, 'CF': 87, 'CB': 65}  

    # Annotate players for each position
    for position, players_data in zip(['RB', 'LB', 'LW', 'RW', 'CDM', 'CM', 'AM', 'CF'], [rb_players_data, lb_players_data, lw_players_data, rw_players_data, dm_players_data, cm_players_data, am_players_data, st_players_data]):
        for index, player in players_data.iterrows():
            ax.annotate(player[column_names[0]], xy=(positions_x[position], start_y[position]), xytext=(positions_x[position], start_y[position]),
                        textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
            start_y[position] -= 3  

    # Annotate left-footed CBs
    offset_left_cb = 0
    for index, player in left_cb_players_data.iterrows():
        ax.annotate(player[column_names[0]], xy=(23, 30), xytext=(23, 30 + offset_left_cb),
                    textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
        offset_left_cb -= 15  # Adjust the offset for left-footed CBs

    # Annotate right-footed CBs
    offset_right_cb = 0
    for index, player in right_cb_players_data.iterrows():
        ax.annotate(player[column_names[0]], xy=(42, 30), xytext=(42, 30 + offset_right_cb),
                    textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
        offset_right_cb -= 15  # Adjust the offset for right-footed CBs

    # Remove the red dot
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    st.pyplot(fig)
    
def streamlit_interface():
    # Pull data from Google Sheets
    url = "https://docs.google.com/spreadsheets/d/1GAghNSTYJTVVl4I9Q-qOv_PGikuj_TQIgSp2sGXz5XM/edit?usp=sharing"
    url1 = "https://docs.google.com/spreadsheets/d/1GAghNSTYJTVVl4I9Q-qOv_PGikuj_TQIgSp2sGXz5XM/edit#gid=1930222963"
    conn = st.connection("gsheets", type=GSheetsConnection)
    data = conn.read(spreadsheet=url)
    data1 = conn.read(spreadsheet=url1)
    
    # Sidebar select box filter for Player Name
    selected_player = st.sidebar.selectbox("Select Player", data['Player'].unique().tolist())

    # Filter data based on selected player name
    filtered_data = data[data['Player'] == selected_player]

    # Filter data1 based on the selected player's Transfermarkt URL
    player_url = filtered_data['Transfermarkt URL'].iloc[0]
    filtered_data1 = data1[data1['Transfermarkt URL'] == player_url]

    # Display player info card visualization
    st.markdown(f"### {selected_player} ###", unsafe_allow_html=True)
    
    # Splitting the player information into three columns
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        # Extract image URL from the cell
        image_url = filtered_data['Image'].iloc[0]
        if not pd.isnull(image_url):  # Check if image_url is not NaN
            # Fetch image from Google Drive
            response = requests.get(image_url)
            if response.status_code == 200:
                st.image(response.content, width=140)
            else:
                st.write("Image not available")
        else:
            st.write("No image available")
    
    # Display player information in the second column
    with col2:
        st.markdown(f"**Team:** {filtered_data['Current Club'].iloc[0]}")
        st.markdown(f"**Age:** {filtered_data['Age'].iloc[0]}")
        st.markdown(f"**Position:** {filtered_data['Position'].iloc[0]}")
        st.markdown(f"**Contract:** {filtered_data['Contract'].iloc[0]}")
        st.markdown(f"**Agent:** {filtered_data['Agent'].iloc[0]}")
        st.markdown(f"**Height:** {filtered_data['Average Height'].iloc[0]}")
        st.markdown(f"**Foot:** {filtered_data['Foot'].iloc[0]}")
        st.markdown(f"**Transfermarkt:** {filtered_data['Transfermarkt URL'].iloc[0]}")

    # Display additional player information in the third column
    with col3:
        st.markdown(f"**A Verdicts:** {filtered_data['A Verdicts'].iloc[0]}")
        st.markdown(f"**B Verdicts:** {filtered_data['B Verdicts'].iloc[0]}")
        st.markdown(f"**Average Player Performance:** {filtered_data['Average Player Performance'].iloc[0]}")
    
    st.markdown("---")  # Add a separator

    # Center align the text "**Player Reports:**"
    st.markdown(f"### Player Reports ###", unsafe_allow_html=True)
    
    # Display report data from data1
    for index, row in filtered_data1[['Player', 'Scout', 'Comments', 'Date of report', 'Player Level - Score']].iterrows():
        st.markdown(f"**Player:** {row['Player']}")
        st.markdown(f"**Scout:** {row['Scout']}")
        st.markdown(f"**Date of Report:** {row['Date of report']}")
        st.markdown(f"**Verdict:** {row['Player Level - Score']}")
        st.markdown(f"**Comments:** {row['Comments']}")
        st.markdown("---")  # Add a separator
      
# Load the DataFrame
df = pd.read_csv("belgiumdata.csv")
df2 = pd.read_csv("championshipscores.csv")
df3 = pd.read_csv("nonpriorityleaguesdata.csv")

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", [ "Player Profile", "Stoke Score", "Player Radar Single", "Player Radar Comparison", "Scatter Plot", "Multi Player Comparison Tab", "Similarity Score", "Stat Search", "Stoke Score - Wyscout", "Confidence Scores"])

# Based on the selected tab, display the corresponding content
if selected_tab == "Stoke Score":
    main_tab(df2)
if selected_tab == "Player Radar Comparison":
    about_tab(df2)  # Pass the DataFrame to the about_tab function
if selected_tab == "Player Radar Single":
    similarity_score(df2)
if selected_tab == "Scatter Plot":
    scatter_plot(df)
if selected_tab == "Similarity Score":
    player_similarity_app(df2)
if selected_tab == "Stat Search":
    player_stat_search(df)
if selected_tab == "Stoke Score - Wyscout":
    stoke_score_wyscout(df3)
if selected_tab == "Confidence Scores":
    display_data()
if selected_tab == "Player Profile":
    streamlit_interface()
elif selected_tab == "Multi Player Comparison Tab":
    comparison_tab(df)
