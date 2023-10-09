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

    # Add a sidebar dropdown box for leagues
    selected_league = st.sidebar.selectbox("Select a League", league_options)

    # Add a sidebar dropdown box for score types
    selected_score_type = st.sidebar.selectbox("Select a Score Type", score_type_options)

    # Add a slider for selecting the age range
    age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Add a multiselect box for selecting contract expiry years
    selected_contract_expiry_years = st.sidebar.multiselect("Select Contract Expiry Years", contract_expiry_years, default=contract_expiry_years)

    # Add a slider for selecting the player market value (in euros) range
    player_market_value_range = st.sidebar.slider("Select Player Market Value Range (Euro)", min_value=min_player_market_value, max_value=max_player_market_value, value=(min_player_market_value, max_player_market_value))

    # Filter the DataFrame based on the selected filters
    filtered_df = df2[(df2['League'] == selected_league) &
                    (df2['Score Type'] == selected_score_type) &
                    (df2['Age'] >= age_range[0]) &
                    (df2['Age'] <= age_range[1]) &
                    (df2['Contract expires'].isin(selected_contract_expiry_years)) &
                    (df2['Market value (millions)'] >= player_market_value_range[0]) &
                    (df2['Market value (millions)'] <= player_market_value_range[1])]

    # Specify the columns you want to display in the final table
    selected_columns = ['Player Name', 'Age', 'Team', 'League', 'Position', 'Score Type', 'Stoke Score', 'Contract expires', 'Market value (millions)']

    # Display the filtered DataFrame with selected columns
    st.table(filtered_df[selected_columns])

    #league = st.sidebar.multiselect(
       # "Select the League:",
        #options=df["Team"].unique(),
        #default=df["Team"].unique()
   #)

    #position = st.sidebar.selectbox(
       # "Select the Position:",
        #options=df["Position"].unique(),
        #index=0  # Set the default index to the first position
    #)

    #df_selection = df.query(
       # "Team == @league & Position == @position"
    #)

    #selected_columns = ["Player Name", "Position", "Team"]  # Always include these columns
    
    #if score_type == "Striker Score":
        #selected_columns.append("Striker Score")
        #sorted_column = "Striker Score"
    #elif score_type == "Midfield Score":
        #selected_columns.append("Midfield Score")
        #sorted_column = "Midfield Score"

    #df_selection = df_selection.sort_values(by=sorted_column, ascending=False)
    #selected_df = df_selection[selected_columns]

    # Create a bar chart of top 10 players based on the selected score
    #st.subheader(f"Stoke City Score")
    #top_10_players = df_selection.nlargest(10, sorted_column)
    #fig = px.bar(
        #top_10_players,
       # x="Player Name",
        #y=sorted_column,
        #title=f"Top 10 Players by {score_type}",
        #labels={"Player Name": "Player", sorted_column: "Score"}
   # )
    
    # Add score annotations over each bar
    #for index, row in top_10_players.iterrows():
        #fig.add_annotation(
           # x=row["Player Name"],
           # y=row[sorted_column] + 2,  # Adjust y position for annotation
           # text=str(row[sorted_column]),
           # showarrow=False,
           # font=dict(size=10)
      #  )

    #st.plotly_chart(fig)
    #st.dataframe(selected_df, width=1500, height=500, hide_index=1)

def about_tab(df2):
    #st.title("Player Profile")

    selected_player = st.sidebar.selectbox(
        "Select a Player:",
        options=df2["Player Name"].unique(),
        index=0  # Set the default index to the first player
    )

# Create a selectbox to choose the profile
    selected_profile = st.sidebar.selectbox(
     "Select a Profile:",
     ["Forward Profile"]
)

# Assuming you want to filter df1 based on the selected player
    selected_df = df2[df2["Player Name"] == selected_player]

    # Reshape the data for plotting based on the selected profile
    if selected_profile == "Forward Profile":
        columns = ["Player Name", "player_season_np_xg_90 Percentile", "player_season_npg_90 Percentile", "player_season_np_shots_90 Percentile"]
        plot_title = f"Full Back Metrics for {selected_player}"
    elif selected_profile == "Winger Profile":
        columns = ["Player Name", "Average Distance", "Top 5 PSV-99", "OBV Dribble & Carry", "Succesful Dribbles", "OP xA", "NP Shots", "NP Goals", "NP xG"]  # Modify as needed
        plot_title = f"Winger Metric Percentiles for {selected_player}"
    
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

    selected_player_row = df2[df2["Player Name"] == selected_player].iloc[0]
    primary_position_text = f"Primary Position: {selected_player_row['primary_position']}"
    player_minutes_text = f"Player Minutes: {selected_player_row['player_season_minutes']}"

# Define the y-coordinate for the 'player_minutes_text' to position it below 'primary_position_text'
    #y_coord = -1.6

    #ax.text(2, y_coord, primary_position_text, ha="center", fontproperties=prop1, fontsize=14)
    #ax.text(2, y_coord - 0.2, player_minutes_text, ha="center", fontproperties=prop1, fontsize=14)

    st.pyplot(fig)
    
    #st.dataframe(df2)

    # Plot for Defensive Play (you can modify column names and logic as needed)
    #ax2 = axes[1]

    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax2.spines['bottom'].set_visible(False)
    #ax2.spines['left'].set_color('#ccc8c8')
    #ax2.spines['bottom'].set_color('#ccc8c8')

    #df_selection_defense = df_selection[["player_name", "PAdj Pressues", "Pressure Regains", "Dribble & Carry OBV"]]  # Modify as needed
    #percentiles_df_defense = df_selection_defense.melt(id_vars="player_name", var_name="Percentile Type", value_name="Percentile")
    #colors_defense = [get_bar_color(p) for p in percentiles_df_defense["Percentile"]]
    #ax2.barh(percentiles_df_defense["Percentile Type"], percentiles_df_defense["Percentile"], color=colors_defense)
    #ax2.set_xlabel("Percentile Value", fontproperties=prop)
    #ax2.set_ylabel("Percentile Type", fontproperties=prop)
    #ax2.set_title(f"Out of Possession for {selected_player}", fontproperties=prop)
    #for label in ax2.get_xticklabels() + ax2.get_yticklabels():
       #label.set_fontproperties(prop)
    #ax2.set_xlim(0, 100)
    #for index, value in enumerate(percentiles_df_defense["Percentile"]):
       #ax2.text(value + 1, index, f"{value:.0f}", va="center", fontproperties=prop)
       # 

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
    filtered_df = df[(df['primary_position'].isin(selected_positions) | (len(selected_positions) == 0)) & 
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
    fig = px.scatter(filtered_df, x=x_variable, y=y_variable)

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


def venn_tab(df1):

    # Sample percentile data for three metrics
    percentile_data = {
     'Metric A': [25, 50, 75],
     'Metric B': [30, 60, 80],
     'Metric C': [20, 50, 70],
}

    st.title('Venn Diagram of Percentile Values')

# Create Venn diagram using matplotlib-venn
    fig, ax = plt.subplots()
    
    venn3(subsets=(
     set(percentile_data['Metric A']),
     set(percentile_data['Metric B']),
     set(percentile_data['Metric C'])
),
    set_labels=('Metric A', 'Metric B', 'Metric C'))

# Display the Venn diagram in the Streamlit app
    st.pyplot(fig)

# Display the data table
    st.subheader('Percentile Data')
    st.table(percentile_data)

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

