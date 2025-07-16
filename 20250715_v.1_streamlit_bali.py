# -*- coding: utf-8 -*-
"""
streamlit_traceability.py

This script converts the Dash-based traceability dashboard into a Streamlit application.
It fetches data from KoboToolbox, performs spatial analysis, and displays interactive
visualizations and a map using Streamlit and Folium.
"""

# --- 1. LIBRARIES ---
import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import json
import plotly.express as px
import folium
from streamlit_folium import st_folium
import rasterio
import os
from branca.element import MacroElement
from jinja2 import Template
import numpy as np
import matplotlib.colors as mcolors

# --- 2. INITIAL PAGE CONFIGURATION ---
# Set the layout to wide mode for a better dashboard experience
st.set_page_config(layout="wide")

# --- 3. DATA LOADING AND PROCESSING ---
# This part remains largely the same as your original script.
# We can use @st.cache_data to speed up the app by caching the data pull.

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_kobo_data():
    """Fetches data from the KoboToolbox API."""
    KOBO_TOKEN = "036d4c8aeb6a0c011630339e605e7e8bb5500c7b"
    ASSET_UID = "aNkj5BVuLuqGfqustJMNaM"
    KOBO_API_URL = f"https://kc.kobotoolbox.org/api/v2/assets/{ASSET_UID}/data.json/"
    HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

    try:
        response = requests.get(KOBO_API_URL, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()['results']
        df = pd.DataFrame(data)
        
        # --- Data Cleaning and Type Conversion ---
        # Convert numeric columns, coercing errors to NaN
        numeric_cols = ["plot_area", "C2_Total_synthetic_ast_year_on_farm_kg", 
                        "main_crop_productivity", "C1_Organic_fertiliz_ast_year_on_farm_kg"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date column
        df['Data_collection_date'] = pd.to_datetime(df['Data_collection_date'])
        
        # Split location into latitude and longitude
        df[['lat', 'lon']] = df['B2_Plot_location'].str.split(' ', expand=True).iloc[:, :2].astype(float)
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from KoboToolbox: {e}")
        return pd.DataFrame() # Return empty dataframe on error
    
@st.cache_resource # Cache the GeoDataFrames
def load_spatial_data():
    """Loads auxiliary spatial data (Peatland and Protected Areas)."""
    try:
        # NOTE: Update these paths to be accessible by your Streamlit app
        # It's best to place them in the same directory or a subdirectory.
        peatland_khGambut_gdf = gpd.read_file("bali/INDONESIA PEATLAND 2017.zip")
        protected_areas_gdf = gpd.read_file("bali/bali_protected_areas.zip")

        # Ensure CRS is consistent (WGS84)
        peatland_khGambut_gdf = peatland_khGambut_gdf.to_crs(epsg=4326)
        protected_areas_gdf = protected_areas_gdf.to_crs(epsg=4326)

        # Convert any datetime columns in protected_areas_gdf to string
        # to prevent JSON serialization errors with Folium.
        for col in protected_areas_gdf.columns:
            if pd.api.types.is_datetime64_any_dtype(protected_areas_gdf[col]):
                protected_areas_gdf[col] = protected_areas_gdf[col].astype(str)

        deforYear = rasterio.open("bali/bali_deforestation_year.tif")

        return peatland_khGambut_gdf, protected_areas_gdf, deforYear
    except Exception as e:
        st.error(f"Error loading spatial data files. Make sure the files are in the correct path: {e}")
        return None, None, None

# Load all data
df = load_kobo_data()
peatland_gdf, protected_areas_gdf, deforYear= load_spatial_data()

# Stop the app if data loading failed
if df.empty or protected_areas_gdf is None:
    st.warning("Data loading failed. Halting application.")
    st.stop()

# --- 4. SPATIAL ANALYSIS: INTERSECTION ---
# This logic is moved from the original script directly here.

# Convert survey data to a GeoDataFrame
survey_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['lon'], df['lat']),
    crs=protected_areas_gdf.crs
)

# Perform spatial join
points_in_protected_areas = gpd.sjoin(survey_gdf, protected_areas_gdf, how="inner", predicate="intersects")

# Create a boolean column to mark points inside protected areas
df['in_protected_area'] = df.index.isin(points_in_protected_areas.index)

# For deforested areas
# Create a boolean column to mark points inside protected areas
# 1. Get the coordinates of each survey point
coords = [(p.x, p.y) for p in survey_gdf.geometry]

# 2. Sample the raster at each coordinate to get the pixel value
# The .sample() method returns a generator, so we convert it to a list.
# Each item will be a numpy array with the raster value, e.g., array([3.])
# We take the first element `val[0]` from each array.
defor_values = [val[0] for val in deforYear.sample(coords)]

# 3. Map the raw raster values (0-4) to actual years
# This gives you a more useful column with the specific year.
# Points outside a deforested area will have a 'None' or 'NaN' value.
year_map = {0: 2020, 1: 2021, 2: 2022, 3: 2023, 4: 2024}
df['deforestation_year'] = [year_map.get(val) for val in defor_values]

# 4. Create the boolean column 'in_deforested_area'
# This will be True if 'deforestation_year' has a value, and False otherwise.
df['in_deforested_area'] = df['deforestation_year'].notna()

# --- 5. UI: SIDEBAR ---
# In Streamlit, it's common to put filters in a sidebar.
st.sidebar.title("Welcome to The Traceability Dashboard!")
st.sidebar.markdown("How to use this dashboard:")
st.sidebar.caption(
"""
1. When you first open the dashboard, it will show all records
2. The charts will straightforwardly appear; while it will take more time for the spatial information to finish running
3. Delete unwanted Farmer Groups from the filter to show only desired records
4. Click the three-dot icon in the topright corner and select 'Rerun' if you have submitted a new record
"""
)
st.sidebar.image('bali/RCT_Logo.png', width=100)
st.sidebar.caption("© 2025 ReClimaTech")

# Initialize session state for farmer group selection if it doesn't exist
if 'selected_groups' not in st.session_state:
    st.session_state.selected_groups = [group for group in df['A13_Farmer_group_cooperative'].unique() if pd.notna(group)]

# Calculate the number of unique cooperatives first
total_cooperatives = df['A13_Farmer_group_cooperative'].nunique()

# --- 6. MAIN DASHBOARD LAYOUT ---
logo_col, title_col, submit_col, group_col = st.columns([2, 8, 2, 2])

with logo_col:
    logo_container = st.container(border=True, height=120)
    logo_container.image("bali/RCT_Logo.png")

with title_col:
    title_container = st.container(border=True, height=120)
    title_container.subheader("**Traceability Tool**")
    title_container.markdown("Let's trace for a better, sustainable farm practice and management © 2025 ReClimaTech")

with submit_col:
    submit_container = st.container(border=True, height=120)
    submit_container.metric("Total Submission", len(df["_id"]))

with group_col:
    group_container = st.container(border=True, height=120)
    group_container.metric("Total Group", value=total_cooperatives)

# --- TABS ---
tabs = st.tabs(["Dashboard", "About"])

with tabs[0]:
    # --- FILTER ---
    # st.subheader("Filters")

    # Get unique farmer groups
    farmer_group_options = [group for group in df['A13_Farmer_group_cooperative'].unique() if pd.notna(group)]
    
    name_map = {
        "kub_jaya_abadi": "KUB Jaya Abadi",
        "kub_sejahtera_bahagia": "KUB Sejahtera Bahagia",
        "kub_tani_jaya": "KUB Tani Jaya"}
    
    # Use session_state to manage the multiselect widget
    st.session_state.selected_groups = st.multiselect(
        'Select Farmer Group(s):',
        options=farmer_group_options,
        default=st.session_state.selected_groups,
        # This function tells Streamlit how to display each option
        format_func=lambda group: name_map.get(group, group) 
    )

    # Filter the dataframe based on the session_state selection
    if st.session_state.selected_groups:
        filtered_df = df[df['A13_Farmer_group_cooperative'].isin(st.session_state.selected_groups)].copy()
    else:
        filtered_df = df.copy()

    # --- INDICATOR & FARM MANAGEMENT CHARTS ---
    # st.subheader("Plot Information & Farm Management")

    # Calculate metrics from the filtered dataframe
    avg_plot_area = filtered_df["plot_area"].mean()
    avg_synth_fert = filtered_df["C2_Total_synthetic_ast_year_on_farm_kg"].mean()
    avg_prod = filtered_df["main_crop_productivity"].mean()
    avg_org_fert = filtered_df["C1_Organic_fertiliz_ast_year_on_farm_kg"].mean()

    # --- PIE CHARTS (CORRECTED & IMPROVED LAYOUT) ---
    def create_pie_chart(data, column_name, title, name_map=None):
        """Helper function to create styled Plotly pie charts."""
        pie_data = data[column_name].value_counts().reset_index()
        pie_data.columns = ['Answer', 'Count']

        # ✅ FIX: This section was missing. It applies the new names.
        if name_map:
            pie_data['Answer'] = pie_data['Answer'].map(name_map).fillna(pie_data['Answer'])
        
        fig = px.pie(pie_data, values='Count', names='Answer', title=title,
                    hole=0.25,width=300, height=200)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=5))
        return fig

    # --- DISPLAY INDICATOR & FARM MANAGEMENT PIE CHARTS ---
    indicator_1, indicator_2, indicator_3, indicator_4 = st.columns(4)

    with indicator_1:
        st.metric("Avg. Plot Area (ha)", f"{avg_plot_area:.2f}", border=True)
        #st.metric("Total Submission (points)", len(filtered_df["_id"]), border=True)

    with indicator_2:
        st.metric("Avg. Crop Productivity (kg/ha)", f"{avg_prod:.2f}", border=True)

    with indicator_3:
        st.metric("Avg. Synthetic Fertilizer (kg/ha)", f"{avg_synth_fert:.2f}", border=True)

    with indicator_4:
        st.metric("Avg. Organic Fertilizer (kg/ha)", f"{avg_org_fert:.2f}", border=True)

    # --- DISPLAY PIE CHARTS --- #
    # Function for creating a bar chart
    def create_bar_chart(data, column_name, chart_title, x_axis_title, y_axis_title):
        """Helper function to create a styled Plotly bar chart."""
        # Get the value counts and prepare the data
        bar_data = data[column_name].value_counts().reset_index()
        bar_data.columns = [column_name, 'Count']

        # Create the bar chart figure
        fig = px.bar(
            bar_data,
            x=column_name,
            y='Count',
            title=chart_title,
            text_auto=True  # Automatically display the count on top of the bars
        )

        # Customize the layout
        fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            width=300,
            height=250,
            margin=dict(l=10, r=10, t=45, b=5)
        )
        return fig

    st.markdown("**Household Information and Farm Management**")
    
    hi_1, hi_2, hi_3, hi_4 = st.columns(4)
    
    with hi_1:
        pieCont_1 = st.container(border=True, height=240)
        education_level_names = {
                    'none': 'None',
                    'primary_school': 'Primary',
                    'secondary_school': 'Secondary',
                    'tertiary_school': 'Tertiary'
                }
        pieCont_1.plotly_chart(create_pie_chart(
                    filtered_df, 
                    "A6_Last_education_level", 
                    "Education Level",
                    name_map=education_level_names
                ), use_container_width=False)
        
        pieCont_2 = st.container(border=True, height=240)
        pieCont_2.plotly_chart(create_pie_chart(
            filtered_df, "Are_you_applying_chemical_pest", 
            "Pesticide Application"), 
            use_container_width=False)
    
    with hi_2:
        pieCont_3 = st.container(border=True, height=240)
        gender_names = {
                'male': 'Male',
                'female': 'Female'
            }
        pieCont_3.plotly_chart(create_pie_chart(
                    filtered_df, "A4_Gender", "Farmer Gender",
                    name_map=gender_names
                ), use_container_width=False)
        
        pieCont_4 = st.container(border=True, height=240)
        pieCont_4.plotly_chart(create_pie_chart(
            filtered_df, "Are_you_applying_chemical_herb", "Herbicide Application"), use_container_width=False)
        
    with hi_3:
        barCont_1 = st.container(border=True, height=240)
        # Create the figure using the new helper function
        main_crop_bar_chart = create_bar_chart(
            data=filtered_df,
            column_name="B4_Main_commodity",
            chart_title="Main Commodity",
            x_axis_title="Crop Type",
            y_axis_title="Count"
        )

        barCont_1.plotly_chart(main_crop_bar_chart)

        pieCont_5 = st.container(border=True, height=240)
        agro_practice_names = {
                'fully_implement': 'Fully',
                'partially_implement': 'Partially',
                'no': 'No'
            }
        pieCont_5.plotly_chart(create_pie_chart(
            filtered_df, "C5_Type_of_agroforestry_practice", "Agroforestry Practice",
            name_map=agro_practice_names), use_container_width=False)

    with hi_4:
        barCont_2 = st.container(border=True, height=240)
        other_crop_bar_chart = create_bar_chart(
            data=filtered_df,
            column_name="B5_Other_crops_beyo_d_the_main_commodity",
            chart_title="Other Commodity",
            x_axis_title="Crop Type",
            y_axis_title="Count"
        )
        barCont_2.plotly_chart(other_crop_bar_chart)

        pieCont_6 = st.container(border=True, height=240)
        pieCont_6.plotly_chart(create_pie_chart(
            filtered_df, "C7_Do_you_irrigate_your_farm", "Irrigation Practice"), use_container_width=False)

    # --- 7. INTERACTIVE MAP & LISTS ---
    # st.subheader("Survey Distribution Map and Farmer Data")

    st.markdown("**Household List**")

    infocol_container = st.container(border=True, height=200)
    table_data = filtered_df.copy()
    table_data['Farmer Name'] = table_data['A1_Producer_farmer_name_first_name'] + ' ' + table_data[
        'A2_Producer_farmer_name_last_name']

    # Select and rename the columns
    display_df = table_data[['Farmer Name', 
                                'A3_Farmer_ID', 
                                'A13_Farmer_group_cooperative',
                                'plot_area',
                                'B4_Main_commodity',
                                'harvested_amount',
                                'C1_Organic_fertiliz_ast_year_on_farm_kg',
                                'C2_Total_synthetic_ast_year_on_farm_kg',
                                'C3_1_If_yes_how_of_herbicides_per_year',
                                'C4_1_If_yes_how_of_pesticides_per_year',
                                'C5_Type_of_agroforestry_practice']]
    display_df = display_df.rename(columns={
        'A3_Farmer_ID': 'Farmer ID',
        'A13_Farmer_group_cooperative': 'Group',
        'plot_area': 'Plot area(ha)',
        'B4_Main_commodity': 'Main Commodity',
        'harvested_amount': 'Harvested Amount',
        'C1_Organic_fertiliz_ast_year_on_farm_kg':'Organic Fertilizer/year (kg/ha)',
        'C2_Total_synthetic_ast_year_on_farm_kg':'Synthetic Fertilizer/year (kg/ha)',
        'C3_1_If_yes_how_of_herbicides_per_year':'Herbicides (kg/year)',
        'C4_1_If_yes_how_of_pesticides_per_year': 'Pesticides (kg/year)',
        'C5_Type_of_agroforestry_practice': 'Agroforestry Practice'
    })

    # Display the interactive table
    infocol_container.dataframe(display_df, use_container_width=True)

    st.markdown("**Spatial Information**")

    def create_folium_map(points_df, peat_gdf, protected_gdf, defor_raster):
        center_lat, center_lon = -8.386294049318078, 115.16917694461883
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

        # --- DEFINE COLORS & LABELS (used in layers and legend) ---
        peat_color = '#4E7254'
        pa_color_map = {
            "Hutan Lindung": "#9D9101", "Taman Wisata Alam": "#B32428",
            "Hutan Suaka Alam dan Wisata": "#E6D690", "Cagar Alam": "#4E3B31",
            "Taman Buru": "#4A192C", "Taman Nasional": "#4C514A",
            "Taman Hutan Raya": "#474B4E", "Suaka Margasatwa": "#6C3B2A",
            "Kawasan Suaka Alam/Kawasan Pelestarian Alam": "#1B5583"
        }
        defor_year_map = {0: "2020", 1: "2021", 2: "2022", 3: "2023", 4: "2024"}
        defor_color_map = ['#FFFF00', '#FFAA00', "#DE7D4C", '#FF0000', "#7A0909"]

        # Layer 1: Peatland
        if peat_gdf is not None:
            folium.GeoJson(peat_gdf, name="Peatland",
                        style_function=lambda x: {'fillColor': peat_color, 'color': peat_color, 'weight': 2, 'fillOpacity': 0.5},
                        tooltip=folium.GeoJsonTooltip(fields=['NAMA_KHG'], aliases=['Peatland:']),
                        show=False).add_to(m)

        # Layer 2: Protected Areas
        if protected_gdf is not None:
            folium.GeoJson(protected_gdf, name="Protected Areas",
                        style_function=lambda f: {'fillColor': pa_color_map.get(f['properties']['NAMOBJ'], 'gray'),
                                                    'color': pa_color_map.get(f['properties']['NAMOBJ'], 'gray'), 'weight': 2, 'fillOpacity': 0.5},
                        tooltip=folium.GeoJsonTooltip(fields=['NAMOBJ'], aliases=['Protected Area:']), show=False).add_to(m)

        # Layer 3: Deforestation Raster (using ImageOverlay for performance)
        if defor_raster:
            raster_data = defor_raster.read(1)
            colored_raster = np.zeros((raster_data.shape[0], raster_data.shape[1], 4), dtype=np.uint8)
            for value, year in defor_year_map.items():
                color_rgba = mcolors.to_rgba(defor_color_map[value], alpha=1.0)
                colored_raster[raster_data == value] = [int(c * 255) for c in color_rgba]
            if defor_raster.nodata is not None:
                colored_raster[raster_data == defor_raster.nodata] = [0, 0, 0, 0]
            raster_bounds = [[defor_raster.bounds.bottom, defor_raster.bounds.left],
                            [defor_raster.bounds.top, defor_raster.bounds.right]]
            folium.raster_layers.ImageOverlay(
                image=colored_raster, bounds=raster_bounds, name="Deforestation Year",
                opacity=0.7, show=False
            ).add_to(m)

        # Layer 4: Survey Points
        marker_group = folium.FeatureGroup(name="Survey Data").add_to(m)
        for _, row in points_df.iterrows():
            if row['in_protected_area']:
                color = 'red'
            elif row['in_deforested_area']:
                color = '#FF8C00' # A nice dark orange
            else:
                color = 'black'
            popup_html = f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;700&display=swap');
                
                .popup-content {{
                    font-family: 'Source Sans 3', sans-serif;
                    font-size: 13px;
                }}
                </style>
                
                <div class="popup-content">
                    <b>Enumerator:</b> {row.get('Enumerator_name', 'N/A')}<br>
                    <b>Farmer Name:</b> {row.get('A1_Producer_farmer_name_first_name', 'N/A')}<br>
                    <b>Farmer ID:</b> {row.get('A3_Farmer_ID', 'N/A')}<br>
                    <b>Group:</b> {row.get('A13_Farmer_group_cooperative', 'N/A')}<br>
                    <b>Plot Area (ha):</b> {row.get('plot_area', 'N/A'):.2f}<br>
                    <b>Productivity:</b> {row.get('main_crop_productivity', 'N/A'):.2f}
                </div>
                """
            iframe = folium.IFrame(popup_html, width=250, height=150)
            popup = folium.Popup(iframe)

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Farmer ID: {row.get('A3_Farmer_ID', 'N/A')}",
                popup=popup
            ).add_to(marker_group)
                               
        # --- NEW: COMPREHENSIVE LEGEND ---
        legend_template = """
        {% macro html(this, kwargs) %}
        <div id="maplegend" class="maplegend"
            style="position: absolute; top: 70px; left: 10px; z-index:1000; font-size:11px; font-family: 'Source Sans 3', sans-serif;">
            
            <style type="text/css">
            .maplegend .legend-toggle-label {
                cursor: pointer; 
                background-color: #FFFFFF; 
                border: 0px solid grey; 
                padding: 10px; 
                font-weight: bold; 
                width: 150px;
                display: block;
            }
            .maplegend .legend-content {
                display: none;
                background-color: white; 
                border: 0px solid grey; 
                border-top: none; 
                padding: 10px; 
                width: 150px;
            }
            .maplegend .legend-toggle-checkbox:checked ~ .legend-content {
                display: block;
            }
            </style>
            
            <input class="legend-toggle-checkbox" id="legend-toggle" type="checkbox" style="display: none;">
            
            <label for="legend-toggle" class="legend-toggle-label">🗺️&nbsp; Map Legend</label>
            
            <div class="legend-content">
                <b>Survey Points</b><br>
                {% for label, color in this.survey_legend.items() %}
                <i class="fa fa-circle" style="color:{{ color }}"></i>&nbsp; {{ label }}<br>
                {% endfor %}
                <hr style="margin: 5px 0;">

                <b>Land Cover</b><br>
                <i class="fa fa-square" style="color:{{ this.peat_color }}"></i>&nbsp; Peatland<br>
                {% for label, color in this.pa_color_map.items() %}
                    <i class="fa fa-square" style="color:{{ color }}"></i>&nbsp; {{ label }}<br>
                {% endfor %}
                <hr style="margin: 5px 0;">

                <b>Deforestation Year</b><br>
                {% for value, year in this.defor_year_map.items() %}
                <i class="fa fa-square" style="color:{{ this.defor_color_map[value] }}"></i>&nbsp; {{ year }}<br>
                {% endfor %}
            </div>
        </div>
        {% endmacro %}
        """
        # Data for the survey points section of the legend
        survey_legend = {
            'Survey Data': 'black',
            'In Deforested Area': '#FF8C00', # Use the same dark orange
            'In Protected Area': 'red'
        }

        macro = MacroElement()
        macro._template = Template(legend_template)
        
        # Pass all necessary data to the template
        macro.survey_legend = survey_legend
        macro.peat_color = peat_color
        macro.pa_color_map = pa_color_map
        macro.defor_year_map = defor_year_map
        macro.defor_color_map = defor_color_map

        m.add_child(macro)
        
        folium.LayerControl().add_to(m)
        return m

    map_col, alert_col = st.columns([4, 2.5])

    with map_col:
        # <<< MODIFIED: Session state logic to prevent map regeneration >>>
        # Check if the filter has changed or if the map doesn't exist in the session state.
        if 'map' not in st.session_state or st.session_state.get('map_filter_state') != st.session_state.selected_groups:
            # If so, update the filter state in session
            st.session_state.map_filter_state = st.session_state.selected_groups.copy()
            # Generate a new map using the currently filtered data
            map_to_display = create_folium_map(filtered_df, peatland_gdf, protected_areas_gdf, deforYear)
            # Save the newly created map to the session state
            st.session_state.map = map_to_display
        
        # Display the map from session state using folium_static.
        # This will not cause a rerun when you pan or zoom.
        st_folium(st.session_state.map, height=450, width='100%', returned_objects=[])

    with alert_col:
        alert1_container = st.container(border=True, height=120)
        protected_alerts_df = filtered_df[filtered_df['in_deforested_area']]
        num_deforested = len(protected_alerts_df)
        alert1_container.markdown("**Alert 1: Deforested Areas**")
        if num_deforested == 0:
            alert1_container.info("No survey points found in deforested areas.")
        else:
            with alert1_container.expander(f"Show {num_deforested} farmer(s)"):
                for _, row in protected_alerts_df.iterrows():
                    st.badge(f"ID {row['A3_Farmer_ID']} is in the deforested area (yr. 2020-2024).", icon="⚠️",color='orange')

        alert2_container = st.container(border=True, height=120)
        deforested_alerts_df = filtered_df[filtered_df['in_protected_area']]
        num_protected = len(deforested_alerts_df)
        alert2_container.markdown("**Alert 2: Protected Areas**")
        if num_protected == 0:
            alert2_container.info("No survey points found in protected areas.")
        else:
            with alert2_container.expander(f"Show {num_protected} farmer(s)"):
                for _, row in deforested_alerts_df.iterrows():
                    st.badge(f"ID {row['A3_Farmer_ID']} is in a protected area.", icon="🚨", color='red')

with tabs[1]:
    st.subheader("About ReClimaTech")
    st.text("At ReClimaTech we connect nature, communities, and businesses to foster sustainable growth through nature-based solutions with tailored and tech-driven consulting & advisory.")
