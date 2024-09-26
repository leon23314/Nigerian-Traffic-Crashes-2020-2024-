import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import subprocess

# Define the Bash command
command = "kaggle datasets download -d akinniyiakinwande/nigerian-traffic-crashes-2020-2024"

# Execute the command
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Load the data into a DataFrame
crashes = pd.read_csv('C:/Users/ak230/Downloads/nigerian_traffic_data/Nigerian_Road_Traffic_Crashes_2020_2024.csv')



# Streamlit title
st.divider()
st.title("Analyse van Verkeersongevallen in Nigeria (2020-2024)")
st.divider()

#Auteurs
st.header("**Auteurs**: Koen van Hardeveld, Leon Kourzanov, Tirej Sendi, Jason Shi en Luuk Terweijden")


#Tekst voor de site zelf
st.subheader("Data")
text = '''Er is onderzocht naar een dataset omtrent verkeersongevallen in Nigeria.
In deze blog wordt er ingegaan op een paar grafieken en de eventuele bevindingen 
in de dataset. Daarna worden aanbevelingen gedaan op basis van de bevindingen.

De dataset is verkregen via Kaggle (bron: *https://www.kaggle.com/datasets/akinniyiakinwande/nigerian-traffic-crashes-2020-2024*)
Hierin staan de gegevens van 2020 tot 2024, maar er is ook onderzocht wat er in de toekomst
kan gebeuren.

Er zijn in deze blog vier figuren te herkennen. In het eerste figuur laten we zien wat de verkeersgegevens per staat zijn. 
Bij het tweede figuur wordt er een voorspelling gemaakt van het aantal crashes met de komende jaren. In het derde figuur is er per 
staat het aantal doden en gewonden door verkeersongevallen te herkennen. In het laatste figuur wordt topografisch weergegeven waar zich
de meeste ongelukken bevinden. 

Deze grafieken zijn interactief, wat als voordeel heeft dat de gebruiker makkelijk
kan zien wat de eventuele correlaties zijn. Ook updaten deze gegevens automatisch mee 
bij een aanpassing op de gebruikte dataset. 
'''
st.markdown(text)

#Figuur 1 tekst
st.header("Verkeersongevallen per staat per soort")
text = '''Verkeersongevallen kunnen onder meer gecategoriseerd worden onder de impact van de crash (zie *Figuur 1*).
Zo kan het zijn dat het ongeval alleen voor een crash heeft gezorgd en niet meer. Ook kan de inzittende gewond 
raken of heeft het een dodelijke impact. Opvallend is 
dat het aantal injured (gewond) vaak veruit het hoogst is ten opzichte van zowel een ongeval zonder complicaties als een dodelijk impact.  
'''
st.markdown(text)
# Dropdown menu to select a state
state_selection = st.selectbox("Selecteer een staat:", crashes['State'].unique())

# Filter the DataFrame based on the selected state
filtered_df = crashes[crashes['State'] == state_selection]

# Calculate totals
total_crashes = filtered_df['Total_Crashes'].sum()
total_injured = filtered_df['Num_Injured'].sum()
total_killed = filtered_df['Num_Killed'].sum()

# Define the metrics and their corresponding values
metrics = ['Total Crashes', 'Total Injured', 'Total Killed']
values = [total_crashes, total_injured, total_killed]

# Create the bar plot using Plotly
fig = go.Figure()

# Add bars to the figure
fig.add_trace(go.Bar(
    x=metrics,
    y=values,
    marker_color=['blue', 'orange', 'red'],
    text=values,  # Display values on bars
    textposition='outside'  # Position of the text
))

# Customize the layout
fig.update_layout(
    title=f"Verkeersgegevens voor {state_selection} (2020-2024)",
    yaxis_title="Aantal",
    xaxis_title="Metric",
    template='plotly_white',
    showlegend=False,
    yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)')  # Light grid lines
)

# Show the plot in Streamlit
st.plotly_chart(fig)
st.caption('''*Figuur 1: Totaal aantal verkeersongevallen per staat. 
           Hierbij wordt onderscheid gemaakt tussen gecrashet, gewond en gedood.*
           ''')


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

#Tekst Figuur 2
st.header("Voorspelling verkeersongelukken")

text = '''Vanuit de verkregen gegevens is er met behulp van een lineaire regressiemodel een voorspelling gemaakt over het verloop
van het aantal verkeersongevallen in de komende jaren (Zie figuur 2). Opvallend is een dalende trend richting het aantal ongevallen
in de komende jaren. Dit is een goed teken. 
'''
st.markdown(text)
# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('C:/Users/ak230/Downloads/Nigerian_Road_Traffic_Crashes_2020_2024.csv')

# Load the data
data = load_data()

# Extract year and quarter number
data['Year'] = data['Quarter'].str.extract(r'(\d{4})')[0].astype(int)
data['Quarter_Number'] = data['Quarter'].str.extract(r'(\d)')[0].astype(int)

# Create a new column for sequential quarters (e.g., 202001, 202002, ...)
data['Seq_Quarter'] = data['Year'].astype(str) + data['Quarter_Number'].astype(str).str.zfill(2)
data['Seq_Quarter'] = pd.to_datetime(data['Seq_Quarter'] + '01', format='%Y%m%d')

# Group by Year and Quarter and sum Total_Crashes
grouped_data = data.groupby(['Year', 'Quarter_Number'])['Total_Crashes'].sum().reset_index()
grouped_data['Quarter_Label'] = grouped_data['Year'].astype(str) + ' Q' + grouped_data['Quarter_Number'].astype(str)

# Prepare data for prediction
X = np.array(range(len(grouped_data))).reshape(-1, 1)  # Create sequential integer values for quarters
y = grouped_data['Total_Crashes'].values

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Year selection slider for predictions
current_year = grouped_data['Year'].max()  # Get the current maximum year in the dataset
years_to_predict = st.slider("Select Year to Predict To", min_value=current_year, max_value=current_year + 5, value=current_year + 1)

# Calculate the number of future quarters to predict
future_periods = (years_to_predict - current_year) * 4  # 4 quarters per year
future_X = np.array(range(len(grouped_data), len(grouped_data) + future_periods)).reshape(-1, 1)
predictions = model.predict(future_X)

# Create future quarter labels
last_year = grouped_data['Year'].iloc[-1]
last_quarter = grouped_data['Quarter_Number'].iloc[-1]

future_labels = []
for i in range(future_periods):
    new_quarter = (last_quarter + i) % 4 + 1
    new_year = last_year + (last_quarter + i) // 4
    future_labels.append(f"{new_year} Q{new_quarter}")

# Create a new DataFrame for future data
future_data = pd.DataFrame({'Quarter_Label': future_labels, 'Total_Crashes': predictions})

# Combine historical and predicted data
combined_data = pd.concat([grouped_data, future_data])

# Create a Plotly figure
fig = go.Figure()

# Add historical data
fig.add_trace(go.Scatter(
    x=grouped_data['Quarter_Label'],
    y=grouped_data['Total_Crashes'],
    mode='lines+markers',
    name='Gegeven Data',
    line=dict(color='blue', width=2)
))

# Add predicted data
fig.add_trace(go.Scatter(
    x=future_data['Quarter_Label'],
    y=future_data['Total_Crashes'],
    mode='lines+markers',
    name='Voorspelde Data',
    line=dict(color='orange', width=2, dash='dash')
))

# Update layout
fig.update_layout(
    title='Voorspelling van verkeersongelukken',
    xaxis_title='Kwartaal',
    yaxis_title='Totaal aantal verkeersongevallen',
    legend_title='Legenda',
    template='plotly_white'
)

# Streamlit plot
st.plotly_chart(fig)
st.caption('''*Figuur 2: Voorspelling van verkeersongelukken. Hierbij kan per jaar worden gekeken wat
           het aantal verkeersongelukken (vermoedelijk) gaat worden.*
           ''')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#Tekst Figuur 3
st.header("Totaal aantal ongevallen per staat")

text = '''Er is ook gekeken naar het aantal ongevallen per staat per kwartaal (Zie onderstaand figuur). Er lijkt hier geen opvallende trend in het
totaalplaatje, alleen dat het aantal crashes lijkt af te nemen. Om een checkbox te maken zoals in dit figuur is er een code nodig.
 Die ziet er als volgt uit met behulp van streamlit.checkbox:
'''
st.markdown(text)
code = '''Show_injured = st.checkbox("Show Number of injured")
Show_killed = st.checkbox("Show Number of killed")'''
st.code(code, language="python")
# Create a dropdown for states
selected_state = st.selectbox("Select a state:", crashes['State'].unique())

# Create checkboxes for showing the number of injured and killed
show_injured = st.checkbox("Show Number of Injured")
show_killed = st.checkbox("Show Number of Killed")

# Filter the data for the selected state
filtered_data = crashes[crashes['State'] == selected_state]

# Create a plot for the number of injured and/or killed
if show_injured or show_killed:
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size for better readability

    if show_injured:
        ax.bar(filtered_data['Quarter'], filtered_data['Num_Injured'], color='blue', alpha=0.7, label='Injured')

    if show_killed:
        ax.bar(filtered_data['Quarter'], filtered_data['Num_Killed'], color='red', alpha=0.7, label='Killed')

    # Add titles and labels
    ax.set_title(f'Crashes in {selected_state}')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add legend
    ax.legend()

    # Display the plot
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the crash data into a DataFrame
crashes = pd.read_csv('C:/Users/ak230/Downloads/Nigerian_Road_Traffic_Crashes_2020_2024.csv')

# Load the GeoJSON file
gdf = gpd.read_file('C:/Users/ak230/Downloads/nigeria_geojson.geojson')

# Calculate total crashes per state
crashes_per_state = crashes.groupby('State')['Total_Crashes'].sum().reset_index()

# Normalize state names for merging
crashes_per_state['State'] = crashes_per_state['State'].replace({'Abuja': 'FCT'})
gdf['state'] = gdf['state'].replace({'Abuja': 'FCT'})

# Merge the crash data with GeoDataFrame
gdf = gdf.merge(crashes_per_state, left_on='state', right_on='State', how='left')

# Streamlit title and text
st.header("Verkeersongevallen per Staat in Nigeria")
text = '''Tot slot is er ook een topografische weergave gemaakt met de top 5 verkeersongevallen per staat (zie figuur 4). Dit is bovendien
ook per crash oorzaak te bekijken. 
'''
st.markdown(text)
# Create two columns for layout
col1, col2 = st.columns([1, 2])  # Adjust the ratios as needed

# Checkbox container in the first column
with col1:
    st.subheader("Select Crash Causes")
    
    # Define crash causes with improved labels
    causes = {
        'SPV': 'Speed Violation (SPV)',
        'DAD': 'Driving under Alcohol/Drug Influence (DAD)',
        'PWR': 'Poor Weather (PWR)',
        'FTQ': 'Fatigue (FTQ)',
        'Other_Factors': 'Other Factors'
    }
    
    selected_causes = []

    # Create checkboxes for each cause
    for key, label in causes.items():
        if st.checkbox(label):
            selected_causes.append(key)

# Plotting the map in the second column
with col2:
    # Filter the crashes based on selected causes
    if selected_causes:
        # Create a boolean mask for the selected causes
        mask = crashes[selected_causes].sum(axis=1) > 0
        filtered_crashes = crashes[mask]

        # Calculate crashes per state based on filtered data
        filtered_crashes_per_state = filtered_crashes.groupby('State')['Total_Crashes'].sum().reset_index()

        # Get the top states with the most crashes
        top_states = filtered_crashes_per_state.nlargest(5, 'Total_Crashes')

        # Check if FCT is not in the top states, if not, add it
        if 'FCT' not in top_states['State'].values:
            fct_crashes = crashes_per_state[crashes_per_state['State'] == 'FCT']
            if not fct_crashes.empty and fct_crashes['Total_Crashes'].values[0] > 0:
                top_states = pd.concat([top_states, fct_crashes]).drop_duplicates().nlargest(5, 'Total_Crashes')
    else:
        # If no cause is selected, show all states
        top_states = crashes_per_state

    # Merge with GeoDataFrame for visualization
    gdf_top = gdf.merge(top_states, left_on='state', right_on='State', how='left', suffixes=('', '_filtered'))

    # Fill missing values with 0 for Total_Crashes
    gdf_top['Total_Crashes'] = gdf_top['Total_Crashes_filtered'].fillna(0)

    # Plotting the map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Adjusted size for the map
    gdf_top.boundary.plot(ax=ax, linewidth=1, color='black')
    gdf_top.plot(column='Total_Crashes', ax=ax, legend=True, cmap='Blues', missing_kwds={"color": "lightgrey"})

    # Annotate state names on the map without associated values
    gdf_top['centroid'] = gdf_top.geometry.centroid
    for x, y, state, total in zip(gdf_top['centroid'].x, gdf_top['centroid'].y, gdf_top['state'], gdf_top['Total_Crashes']):
        # Set text color to white for darker states and black for lighter states
        text_color = 'white' if total > 1000 else 'black'
        ax.text(x, y, state, fontsize=7, ha='center', color=text_color)

    ax.set_title("Verkeersongevallen per Staat in Nigeria")
    plt.axis('off')

    # Show the plot in Streamlit
    st.pyplot(fig)
    st.caption('Figuur 4: Topografische weergave van aantal verkeersongevallen per staat in Nigeria.')

st.header("Aanbevelingen")
text = '''Vanuit bovenstaande grafieken is opgevallen dat het aantal verkeersongevallen wel in algemene trend afneemt, dus dat is goed.
Wel zijn er in de grote nigeriaanse staten nog relatief veel ongelukken. Opvallend is dat met name andere factoren niet zo snel afnemen. 
Dit zou dus te maken kunnen hebben met de kwaliteit van de wegen. Daarom wordt het volgende advies gegeven: Investeer in de infrastructuur/
verkeersveiligheid!
'''
st.markdown(text)