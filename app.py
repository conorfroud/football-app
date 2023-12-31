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
from sklearn.metrics.pairwise import cosine_similarity

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

    # Add a slider for selecting the L/R Footedness % range
    lr_footedness_range = st.sidebar.slider("Select L/R Footedness % Range", min_value=0, max_value=100, value=(0, 100))

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
        'Striker': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile', 'Contract expires', 'Market value (millions)', 'xG (ST)', 'Non-Penalty Goals (ST)', 'Shots (ST)', 'OBV Shot (ST)', 'Open Play xA (ST)', 'OBV Dribble & Carry (ST)', 'PAdj Pressures (ST)', 'Aerial Wins (ST)', 'Aerial Win % (ST)', 'L/R Footedness %'],
        'Winger': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Distance (W)', 'Top 5 PSV (W)', 'Contract expires', 'Market value (millions)', 'xG (W)', 'Non-Penalty Goals (W)', 'Shots (W)', 'OBV Pass (W)', 'Open Play xA (W)', 'Successful Dribbles (W)', 'OBV Dribble & Carry (W)', 'L/R Footedness %'],
        'Attacking Midfield': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CAM)',	'Top 5 PSV (CAM)', 'Contract expires', 'Market value (millions)', 'xG (CAM)', 'Non-Penalty Goals (CAM)', 'Shots (CAM)', 'OBV Pass (CAM)', 'Open Play xA (CAM)', 'Key Passes (CAM)', 'Throughballs (CAM)', 'Successful Dribbles (CAM)', 'OBV Dribble & Carry (CAM)', 'L/R Footedness %'],
        'Central Midfield': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (8)',	'Top 5 PSV-99 (8)', 'Contract expires', 'Market value (millions)', 'xG (8)', 'Non-Penalty Goals (8)', 'OBV Pass (8)', 'Open Play xA (8)', 'Deep Progressions (8)', 'Successful Dribbles (8)', 'OBV Dribble & Carry (8)', 'L/R Footedness %'],
        'Defensive Midfield': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'Contract expires', 'Market value (millions)', 'Deep Progressions (6)', 'OBV Pass (6)', 'OBV Dribble & Carry (6)', 'PAdj Tackles (6)', 'PAdj Interceptions (6)', 'Tackle/Dribbled Past % (6)', 'OBV Defensive Action (6)', 'L/R Footedness %'],
        'Left Back': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (LB)', 'PAdj Interceptions (LB)', 'OBV Defensive Action (LB)', 'Tackle/Dribbled Past (LB)', 'Dribbled Past (LB)', 'OBV Dribble & Carry (LB)', 'Successful Dribbles (LB)', 'OBV Pass (LB)', 'Open Play xA (LB)', 'Key Passes (LB)', 'Successful Crosses (LB)', 'L/R Footedness %'],
        'Right Back': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'Contract expires', 'Market value (millions)', 'PAdj Tackles (RB)', 'PAdj Interceptions (RB)', 'OBV Defensive Action (RB)', 'Tackle/Dribbled Past (RB)', 'Dribbled Past (RB)', 'OBV Dribble & Carry (RB)', 'Successful Dribbles (RB)', 'OBV Pass (RB)', 'Open Play xA (RB)', 'Key Passes (RB)', 'Successful Crosses (RB)', 'L/R Footedness %'],
        'Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CB)',	'Top 5 PSV-99 (CB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (CB)', 'Aerial Win % (CB)', 'PAdj Interceptions (CB)', 'PAdj Tackles (CB)', 'OBV Pass (CB)', 'Deep Progressions (CB)', 'OBV Dribble & Carry (CB)', 'OBV Defensive Action (CB)', 'L/R Footedness %'],
        'Stretch 9': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (S9)',	'Top 5 PSV-99 (S9)', 'Contract expires', 'Market value (millions)', 'xG (S9)', 'Non-Penalty Goals (S9)', 'Shots (S9)', 'OBV Shot (S9)', 'Open Play xA (S9)', 'OBV Dribble & Carry (S9)', 'PAdj Pressures (S9)', 'Runs in Behind (S9)', 'Threat of Runs in Behind (S9)', 'L/R Footedness %'],
        'Target 9': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ST)',	'Top 5 PSV-99 (ST)', 'Contract expires', 'Market value (millions)'],
        'Dribbling Winger': ['Player Name', 'Age', 'Team', 'Player Season Minutes', 'League', 'Stoke Score', 'Average Distance (DW)',	'Top 5 PSV (DW)', 'Contract expires', 'Market value (millions)', 'xG (DW)', 'Non-Penalty Goals (DW)', 'Shots (DW)', 'OBV Pass (DW)', 'Open Play xA (DW)', 'Successful Dribbles (DW)', 'OBV Dribble & Carry (DW)', 'L/R Footedness %'],
        'Creative Winger': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (CW)',	'Top 5 PSV (CW)', 'Contract expires', 'Market value (millions)', 'xG (CW)', 'Non-Penalty Goals (CW)', 'Shots (CW)', 'OBV Pass (CW)', 'Open Play xA (CW)', 'Successful Dribbles (CW)', 'OBV Dribble & Carry (CW)', 'L/R Footedness %'],
        'Goalscoring Wide Forward': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (WF)', 'Top 5 PSV (WF)', 'Contract expires', 'Market value (millions)', 'xG (WF)', 'Non-Penalty Goals (WF)', 'Shots (WF)', 'OBV Pass (WF)', 'Open Play xA (WF)', 'Successful Dribbles (WF)', 'OBV Dribble & Carry (WF)', 'L/R Footedness %'],
        'Running 10': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (R10)',	'Top 5 PSV (R10)', 'Contract expires', 'Market value (millions)', 'xG (R10)', 'Non-Penalty Goals (R10)', 'Shots (R10)', 'OBV Pass (R10)', 'Open Play xA (R10)', 'Successful Dribbles (R10)', 'OBV Dribble & Carry (R10)', 'PAdj Pressures (R10)', 'Pressure Regains (R10)', 'HI Distance (R10)', 'L/R Footedness %'],
        'Creative 10': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (C10)', 'Top 5 PSV (C10)', 'Contract expires', 'Market value (millions)', 'xG (C10)', 'Non-Penalty Goals (C10)', 'Shots (C10)', 'OBV Pass (C10)', 'Open Play xA (C10)', 'Key Passes (C10)', 'Successful Dribbles (C10)', 'OBV Dribble & Carry (C10)', 'L/R Footedness %'],
        'Progressive 8': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (P8)', 'Top 5 PSV (P8)', 'Contract expires', 'Market value (millions)', 'xG (P8)', 'OBV Pass (P8)', 'Open Play xA (P8)', 'Successful Dribbles (P8)', 'OBV Dribble & Carry (P8)', 'Deep Progressions (P8)', 'PAdj Pressures (P8)', 'L/R Footedness %'],
        'Running 8': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (R8)', 'Top 5 PSV (R8)', 'Contract expires', 'Market value (millions)', 'xG (R8)', 'Non-Penalty Goals (R8)', 'OBV Pass (R8)', 'Open Play xA (R8)', 'OBV Dribble & Carry (R8)', 'Deep Progressions (R8)', 'PAdj Pressures (R8)', 'Pressure Regains (R8)', 'Runs Threat Per Match (R8)', 'L/R Footedness %'],
        'Progressive 6': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (P6)', 'Top 5 PSV-99 (P6)', 'Contract expires', 'Market value (millions)', 'OBV Defensive Action (P6)', 'Deep Progressions (P6)', 'OBV Dribble & Carry (P6)', 'PAdj Tackles & Interceptions (P6)', 'OBV Pass (P6)', 'Forward Pass % (P6)', 'L/R Footedness %'],
        'Defensive 6': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (D6)', 'Top 5 PSV-99 (D6)', 'Contract expires', 'Market value (millions)', 'PAdj Pressures (D6)', 'OBV Defensive Action (D6)', 'Passing % (D6)', 'Tackle / Dribbled Past % (D6)', 'PAdj Tackles & Interceptions (D6)', 'Ball Recoveries (D6)', 'L/R Footedness %'],
        'Attacking LB': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ALB)', 'Top 5 PSV-99 (ALB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ALB)', 'PAdj Tackles (ALB)', 'PAdj Interceptions (ALB)', 'OBV Pass (ALB)', 'OBV Dribble & Carry (ALB)', 'Tackle / Dribbled Past % (ALB)', 'Threat of Runs (ALB)', 'Successful Crosses (ALB)', 'L/R Footedness %'],
        'Defensive LB': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DLB)', 'Top 5 PSV-99 (DLB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DLB)', 'PAdj Tackles (DLB)', 'PAdj Interceptions (DLB)', 'OBV Dribble & Carry (DLB)', 'OBV Defensive Action (DLB)', 'Tackle / Dribbled Past % (DLB)', 'PAdj Pressures (DLB)', 'Dribbled Past (DLB)', 'Aerial Win % (DLB)', 'L/R Footedness %'],
        'Attacking RB': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (ARB)', 'Top 5 PSV-99 (ARB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (ARB)', 'PAdj Tackles (ARB)', 'PAdj Interceptions (ARB)', 'OBV Pass (ARB)', 'OBV Dribble & Carry (ARB)', 'Tackle / Dribbled Past % (ARB)', 'Threat of Runs (ARB)', 'Successful Crosses (ARB)', 'L/R Footedness %'],
        'Defensive RB': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DRB)', 'Top 5 PSV-99 (DRB)', 'Contract expires', 'Market value (millions)', 'Open Play xA (DRB)', 'PAdj Tackles (DRB)', 'PAdj Interceptions (DRB)', 'OBV Dribble & Carry (DRB)', 'OBV Defensive Action (DRB)', 'Tackle / Dribbled Past % (DRB)', 'PAdj Pressures (DRB)', 'Dribbled Past (DRB)', 'Aerial Win % (DRB)', 'L/R Footedness %'],
        'Ball Playing Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (BCB)', 'Top 5 PSV-99 (BCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (BCB)', 'PAdj Interceptions (BCB)', 'PAdj Tackles (BCB)', 'OBV Pass (BCB)', 'OBV Dribble & Carry (BCB)', 'OBV Defensive Action (BCB)', 'Deep Progressions (BCB)', 'Pressured Change in Passing % (BCB)', 'L/R Footedness %'],
        'Dominant Centre Back': ['Player Name', 'Age', 'Team', 'League', 'Player Season Minutes', 'Stoke Score', 'Average Distance (DCB)', 'Top 5 PSV-99 (DCB)', 'Contract expires', 'Market value (millions)', 'Aerial Wins (DCB)', 'Aerial Win % (DCB)', 'OBV Defensive Action (DCB)', 'Tackle / Dribbled Past % (DCB)', 'Blocks Per Shot (DCB)', 'L/R Footedness %'],
    }

    # Update the selected columns to include 'Score Type'
    selected_columns = score_type_column_mapping.get(selected_score_type, [])

    filtered_df = df2[
        (df2['League'] == selected_league) &
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
        (df2[selected_columns[6]] <= top_5_psv_99_percentile_range[1]) &
        (df2['L/R Footedness %'] >= lr_footedness_range[0]) &
        (df2['L/R Footedness %'] <= lr_footedness_range[1])
    ]

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
    allowed_score_types = ["Striker", "Winger", "Attacking Midfield", "Central Midfield", "Left Back", "Right Back", "Centre Back", "Stretch 9"]

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
        selected_leagues = st.sidebar.multiselect('Select Leagues', df['competition_name'].unique(), default=default_leagues)

        # Sidebar for filtering by 'minutes' played
        min_minutes = int(df['Player Season Minutes'].min())
        max_minutes = int(df['Player Season Minutes'].max())
        selected_minutes = st.sidebar.slider('Select Minutes Played Range', min_value=min_minutes, max_value=max_minutes, value=(300, max_minutes))

        # Filter data based on user-selected positions, minutes played, and leagues
        filtered_df = df[(df['position_1'].isin(selected_positions) | (len(selected_positions) == 0)) &
                         (df['Player Season Minutes'] >= selected_minutes[0]) &
                         (df['Player Season Minutes'] <= selected_minutes[1]) &
                         (df['competition_name'].isin(selected_leagues) | (len(selected_leagues) == 0))]

        # Calculate Z-scores for the variables
        filtered_df['z_x'] = (filtered_df[x_variable] - filtered_df[x_variable].mean()) / filtered_df[x_variable].std()
        filtered_df['z_y'] = (filtered_df[y_variable] - filtered_df[y_variable].mean()) / filtered_df[y_variable].std()

        # Define a threshold for labeling outliers (you can customize this threshold)
        threshold = st.sidebar.slider('Label Threshold', min_value=0.1, max_value=5.0, value=2.0)

        # Create a scatter plot using Plotly with the filtered data
        hover_data_fields = {'Player Name': True, 'team_name': True, 'age': True, 'Player Season Minutes': True, x_variable: False, y_variable: False, 'z_x': False, 'z_y': False}
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
            hover_data_fields_selected = {'Player Name': True, 'team_name': True, 'age': True, 'Minutes': True, x_variable: False, y_variable: False, 'z_x': False, 'z_y': False}
            fig.add_trace(selected_trace).update_traces(hoverinfo="text+x+y")

        # Display the plot in Streamlit
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

def calculate_similarity(player1, player2, columns):
    metrics1 = player1[columns].fillna(0).values
    metrics2 = player2[columns].fillna(0).values
    return np.linalg.norm(metrics1 - metrics2)

# Add a function for rescaling similarity scores
def rescale_similarity(x, max_val):
    return 100 - x * (100 / max_val)
       
def player_similarity_app(df2):
    # Add a sidebar dropdown for selecting a player name
    player_name = st.sidebar.selectbox("Select a player's name:", df2['Player Name'].unique())
    
    # Add a sidebar radio button for selecting a position to compare
    position_to_compare = st.sidebar.radio("Select a position to compare:", ('Stretch 9', 'Winger', 'Attacking Midfield', 'Left Back', 'Right Back', 'Centre Back'))

    # Add a slider to filter players by age
    max_age = st.sidebar.slider("Select maximum age:", min_value=18, max_value=40, value=30)

    # Add a slider to filter players by 'Player Season Minutes'
    min_minutes = st.sidebar.slider("Select minimum 'Player Season Minutes':", min_value=0, max_value=int(df2['Player Season Minutes'].max()), value=0)

    # Automatically select all leagues by default
    selected_leagues = df2['League'].unique()

    # Filter unique leagues based on the selected position and filters
    filtered_leagues = df2[(df2['Score Type'] == position_to_compare) & (df2['Age'] <= max_age) & (df2['Player Season Minutes'] >= min_minutes)]['League'].unique()

    # Set the default value for selected_leagues based on availability
    if all(league in filtered_leagues for league in selected_leagues):
        default_selected_leagues = selected_leagues
    else:
        default_selected_leagues = filtered_leagues

    # Add a multi-select dropdown for filtering by 'League' with default value
    selected_leagues = st.sidebar.multiselect("Select leagues:", filtered_leagues, default=default_selected_leagues)

    # Check if the selected player is in the dataset
    if player_name in df2['Player Name'].values:
        # Choose the reference player
        reference_player = player_name

        # Define columns based on the selected position
        if position_to_compare == 'Striker':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'xG (ST)', 'Non-Penalty Goals (ST)', 'Shots (ST)', 'OBV Shot (ST)', 'Open Play xA (ST)', 'Aerial Wins (ST)', 'Average Distance Percentile', 'Top 5 PSV-99 Percentile']
        elif position_to_compare == 'Winger':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'xG (W)', 'Non-Penalty Goals (W)', 'Shots (W)', 'Open Play xA (W)', 'OBV Pass (W)', 'Successful Dribbles (W)', 'OBV Dribble & Carry (W)', 'Distance (W)', 'Top 5 PSV (W)']
        elif position_to_compare == 'Attacking Midfield':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'xG (CAM)', 'Non-Penalty Goals (CAM)', 'Shots (CAM)', 'Open Play xA (CAM)', 'OBV Pass (CAM)', 'Successful Dribbles (CAM)', 'OBV Dribble & Carry (CAM)', 'Average Distance (CAM)', 'Top 5 PSV (CAM)']
        elif position_to_compare == 'Central Midfield':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'xG (8)', 'Non-Penalty Goals (8)',	'OBV Pass (8)',	'Open Play xA (8)',	'Successful Dribbles (8)', 'OBV Dribble & Carry (8)', 'Average Distance (8)', 'Top 5 PSV-99 (8)', 'PAdj Tackles & Interceptions (8)', 'Deep Progressions (8)']
        elif position_to_compare == 'Defensive Midfield':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'Average Distance (6)', 'Top 5 PSV-99 (6)', 'OBV Defensive Action (6)', 'OBV Pass (6)', 'Deep Progressions (6)', 'Successful Dribbles (6)', 'OBV Dribble & Carry (6)', 'Tackle/Dribbled Past % (6)', 'PAdj Tackles & Interceptions (6)', 'Pass Forward % (6)', 'Turnovers (6)', 'PAdj Pressures (6)', 'Pressure Regains (6)', 'Ball Recoveries (6)']
        elif position_to_compare == 'Stretch 9':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'xG (S9)', 'Non-Penalty Goals (S9)', 'Shots (S9)', 'OBV Shot (S9)', 'Open Play xA (S9)', 'OBV Dribble & Carry (S9)', 'PAdj Pressures (S9)', 'Aerial Wins (S9)', 'Aerial Win % (S9)', 'Average Distance (S9)', 'Top 5 PSV-99 (S9)', 'Runs in Behind (S9)', 'Threat of Runs in Behind (S9)']
        elif position_to_compare == 'Centre Back':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'Top 5 PSV-99 (CB)',	'Aerial Win % (CB)', 'Aerial Wins (CB)', 'OBV Pass (CB)', 'OBV Dribble & Carry (CB)', 'OBV Defensive Action (CB)', 'Deep Progressions (CB)', 'PAdj Tackles & Interceptions (CB)', 'Tackle / Dribbled Past % (CB)', 'Blocks per Shot (CB)', 'Pressure Change in Passing % (CB)']
        elif position_to_compare == 'Left Back':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'Average Distance (LB)', 'Top 5 PSV-99 (LB)', 'OBV Defensive Action (LB)', 'OBV Dribble & Carry (LB)', 'Tackle/Dribbled Past (LB)', 'Open Play xA (LB)', 'Successful Crosses (LB)', 'Dribbled Past (LB)', 'Successful Dribbles (LB)', 'OBV Pass (LB)', 'PAdj Tackles & Interceptions (LB)', 'Aerial Win % (LB)']
        elif position_to_compare == 'Right Back':
            columns_to_compare = ['Player Name', 'Team', 'Age', 'League', 'Player Season Minutes', 'Average Distance (RB)', 'Top 5 PSV-99 (RB)', 'OBV Defensive Action (RB)', 'OBV Dribble & Carry (RB)', 'Tackle/Dribbled Past (RB)', 'Open Play xA (RB)', 'Successful Crosses (RB)', 'Dribbled Past (RB)', 'Successful Dribbles (RB)', 'OBV Pass (RB)', 'PAdj Tackles & Interceptions (RB)', 'Aerial Win % (RB)']

        # Calculate similarity scores for all players within the age, minutes, and league bracket
        similarities = {}
        reference_player_data = df2[(df2['Player Name'] == reference_player) & (df2['Score Type'] == position_to_compare)].iloc[0]

        # Find the maximum similarity score for scaling
        max_similarity = float('-inf')

        for _, player in df2.iterrows():
            if (player['Player Name'] != reference_player) and (player['Age'] <= max_age) and (player['Score Type'] == position_to_compare) and (player['Player Season Minutes'] >= min_minutes) and (player['League'] in selected_leagues):
                similarity_score = calculate_similarity(
                    reference_player_data,
                    player,
                    columns_to_compare[5:]  # Exclude the first three columns (Player Name, Player Club, Age)
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
        st.header(f"Most similar {position_to_compare}s to {reference_player} (Age <= {max_age}, Minutes >= {min_minutes}):")
        similar_players_df = pd.DataFrame(similar_players, columns=['Player Name', 'Similarity Score'])
        
        # Add 'Player Club', 'Age', 'Player Season Minutes', and 'League' columns to the DataFrame
        similar_players_df = pd.merge(similar_players_df, df2[['Player Name', 'Team', 'Age', 'Player Season Minutes', 'League']], on='Player Name', how='left')
        
        # Remove duplicates in case of multiple matches in the age, minutes, and league filter
        similar_players_df = similar_players_df.drop_duplicates(subset='Player Name')
        
        st.dataframe(similar_players_df.head(250))
    else:
        st.error("Player not found in the selected position.")

# Load the DataFrame
df = pd.read_csv("belgiumdata.csv")
df2 = pd.read_csv("championshipscores.csv")

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Stoke Score", "Player Radar Single", "Player Radar Comparison", "Scatter Plot", "Multi Player Comparison Tab", "Similarity Score"])

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
elif selected_tab == "Multi Player Comparison Tab":
    comparison_tab(df)

