"""
Class: CS230
Name: Matthew Del Sesto
Data: Volcanoes.csv
URL:  http://192.168.86.29:8501
Description: Final Project on Volcano Data
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Global Volcano Explorer",
    page_icon="🌋",
    layout="wide"
)

#[DA1] Cleaning Data
def load_volcano_data(filepath, filter_by_country=None):
    try:
        data = pd.read_csv(filepath, encoding='utf-8', header=1)
    except UnicodeDecodeError:
        data = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', header=1)
    data.columns = [col.lower().strip() for col in data.columns]

    # Map critical columns
    column_mapping = {
        'volcano name': 'volcano_name',
        'country': 'country',
        'volcanic region': 'volcanic_region',
        'volcanic province': 'volcanic_province',
        'volcano landform': 'volcano_landform',
        'primary volcano type': 'primary_volcano_type',
        'activity evidence': 'activity_evidence',
        'last known eruption': 'last_known_eruption',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'elevation (m)': 'elevation_(m)',
        'tectonic setting': 'tectonic_setting',
        'dominant rock type': 'dominant_rock_type'
    }

    # Rename columns based on the mapping
    data = data.rename(columns=column_mapping)

    # Filter by country if specified
    if filter_by_country:
        if 'country' in data.columns:
            data = data[data['country'].str.contains(filter_by_country, case=False, na=False)]

    return data


# [PY2] A function that returns more than one value
def get_volcano_stats(data, region=None):
    # Apply filter if region is specified
    if region:
        filtered_data = data[data['volcanic_region'] == region]
    else:
        filtered_data = data

    # Calculate statistics
    count = len(filtered_data)
    avg_elevation = filtered_data['elevation_(m)'].mean() if count > 0 else 0

    # Find most common volcano type
    if count > 0 and 'primary_volcano_type' in filtered_data.columns:
        type_counts = filtered_data['primary_volcano_type'].value_counts()
        common_type = type_counts.index[0] if not type_counts.empty else "Unknown"
    else:
        common_type = "Unknown"

    # Get maximum elevation
    max_elevation = filtered_data['elevation_(m)'].max() if count > 0 else 0

    return count, avg_elevation, common_type, max_elevation


# [PY3] Error checking with try/except
def parse_elevation_range(min_val, max_val):
    try:
        min_elevation = float(min_val) if min_val else float('-inf')
        max_elevation = float(max_val) if max_val else float('inf')

        if min_elevation > max_elevation:
            st.error("Minimum elevation cannot be greater than maximum elevation.")
            return None, None

        return min_elevation, max_elevation
    except ValueError:
        st.error("Please enter valid numeric values for elevation range.")
        return None, None


# [PY5] A dictionary for volcanic region properties
region_properties = {
    "Mediterranean and Western Asia": {"color": "red", "marker": "o",
                                       "description": "Includes volcanoes in Italy, Turkey, and surrounding areas"},
    "Africa and Red Sea": {"color": "orange", "marker": "s", "description": "Volcanoes across the African continent"},
    "Middle East and Indian Ocean": {"color": "yellow", "marker": "^",
                                     "description": "Includes volcanoes in Arabia and Indian Ocean islands"},
    "New Zealand to Fiji": {"color": "green", "marker": "d", "description": "Pacific island arc volcanoes"},
    "Melanesia and Australia": {"color": "blue", "marker": "p",
                                "description": "Includes volcanoes in Papua New Guinea and Australia"},
    "Indonesia": {"color": "indigo", "marker": "*", "description": "One of the most volcanically active regions"},
    "Philippines and SE Asia": {"color": "violet", "marker": "h",
                                "description": "Includes volcanoes in Philippines and mainland Southeast Asia"},
    "Japan, Taiwan, Marianas": {"color": "brown", "marker": "+",
                                "description": "Island arc system with frequent eruptions"},
    "Kurile Islands": {"color": "pink", "marker": "x", "description": "Arc of volcanic islands in Russia's Far East"},
    "Kamchatka and Mainland Asia": {"color": "gray", "marker": "D",
                                    "description": "Includes the highly active Kamchatka Peninsula"},
    "Alaska": {"color": "olive", "marker": "v", "description": "Aleutian Islands and mainland Alaska"},
    "Canada and Western USA": {"color": "cyan", "marker": ">", "description": "Includes Cascade Range volcanoes"},
    "Hawaii and Pacific Ocean": {"color": "magenta", "marker": "<",
                                 "description": "Hawaiian Islands and other Pacific volcanoes"},
    "Mexico and Central America": {"color": "teal", "marker": "1", "description": "Part of the Pacific Ring of Fire"},
    "South America": {"color": "navy", "marker": "2", "description": "Includes the Andes volcanic belt"},
    "West Indies": {"color": "purple", "marker": "3", "description": "Caribbean island arc volcanoes"},
    "Iceland and Arctic Ocean": {"color": "lime", "marker": "4",
                                 "description": "Volcanic island formed by the Mid-Atlantic Ridge"},
    "Atlantic Ocean": {"color": "gold", "marker": "8", "description": "Mid-ocean ridge and hotspot volcanoes"},
    "Antarctica": {"color": "silver", "marker": "P", "description": "Polar volcanoes including Mt. Erebus"}
}


# Main application
def main():
    st.title("🌋 Global Volcano Explorer")

    st.sidebar.header("Data Loading")
    # In a real application, you'd allow the user to upload a file
    # For this example, we'll assume the file is already available
    data_load_state = st.sidebar.text('Loading volcano data...')

    try:
        # Allow user to upload their own file or use a default path
        uploaded_file = st.sidebar.file_uploader("Upload volcanoes.csv file", type=['csv'])

        if uploaded_file is not None:
            # If user uploaded a file, use that
            df = load_volcano_data(uploaded_file)
        else:
            # Otherwise try to use local file
            df = load_volcano_data("volcanoes.csv")

        # Display column information
        st.sidebar.expander("Debug Info").write(f"Columns found: {df.columns.tolist()}")

        data_load_state.text('Loading volcano data... done!')

        # Show raw data if requested
        if st.sidebar.checkbox('Show raw data'):
            st.subheader('Raw Data')
            st.write(df.head(20))

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Volcano Map", "Statistical Analysis", "Eruption History", "Comparative Analysis"])

        with tab1:
            st.header("Volcano Distribution Map")
            st.write("Use the filters below to explore volcanoes around the world.")

            # [ST1] Dropdown for country selection
            countries = ["All"] + sorted(df['country'].dropna().unique().tolist())
            selected_country = st.selectbox("Select Country:", countries)

            # [ST2] Multiselect for volcano types
            volcano_types = sorted(df['primary_volcano_type'].dropna().unique().tolist())
            selected_types = st.multiselect("Select Volcano Types:", volcano_types,
                                            default=[volcano_types[0]] if volcano_types else [])

            # [ST3] Slider for elevation range
            min_elevation = float(df['elevation_(m)'].dropna().min()) if not df['elevation_(m)'].dropna().empty else 0.0
            max_elevation = float(df['elevation_(m)'].dropna().max()) if not df[
                'elevation_(m)'].dropna().empty else 5000.0

            elevation_range = st.slider(
                "Elevation Range (meters):",
                min_value=min_elevation,
                max_value=max_elevation,
                value=(min_elevation, max_elevation),
                step=100.0  # Keep this a float
            )

            # Filter data based on selections
            filtered_df = df.copy()
            if selected_country != "All":
                filtered_df = filtered_df[filtered_df['country'] == selected_country]

            if selected_types:
                filtered_df = filtered_df[filtered_df['primary_volcano_type'].isin(selected_types)]

            filtered_df = filtered_df[
                (filtered_df['elevation_(m)'] >= elevation_range[0]) &
                (filtered_df['elevation_(m)'] <= elevation_range[1]) |
                (filtered_df['elevation_(m)'].isna())  # Keep rows with missing elevation data
                ]

            # Create scatterplot map using matplotlib
            if not filtered_df.empty:
                st.subheader(f"Map showing {len(filtered_df)} volcanoes")

                fig, ax = plt.subplots(figsize=(10, 6))

                # [DA2] Sort data by elevation for bubble size
                filtered_df = filtered_df.sort_values('elevation_(m)')

                # [MAP] Create a detailed map with volcano locations
                scatter = ax.scatter(
                    filtered_df['longitude'],
                    filtered_df['latitude'],
                    c=filtered_df['elevation_(m)'],
                    cmap='magma',
                    alpha=0.7,
                    s=filtered_df['elevation_(m)'].apply(lambda x: max(20, min(300, x / 50))),  # Scale bubble size
                    edgecolors='white',
                    linewidths=0.5
                )

                # Add colorbar to show elevation scale
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Elevation (m)')

                # Add basic map features
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_facecolor('#f0f0f0')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Global Volcano Distribution')

                # Select a few prominent volcanoes to label
                if not filtered_df.empty:
                    # Filter out rows with NaN values in elevation
                    valid_elev_df = filtered_df.dropna(subset=['elevation_(m)'])
                    if not valid_elev_df.empty:
                        top_volcanoes = valid_elev_df.nlargest(5, 'elevation_(m)')
                        for _, row in top_volcanoes.iterrows():
                            ax.annotate(
                                row['volcano_name'],
                                (row['longitude'], row['latitude']),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                            )

                st.pyplot(fig)

                # Add a table of the most notable volcanoes in the filtered set
                st.subheader("Notable Volcanoes in Selection")

                # [DA3] Find top 5 largest volcanoes by elevation
                notable_volcanoes = filtered_df.dropna(subset=['elevation_(m)']).nlargest(5, 'elevation_(m)')

                # Display as a styled dataframe
                st.dataframe(
                    notable_volcanoes[['volcano_name', 'country', 'elevation_(m)', 'last_known_eruption']],
                    use_container_width=True
                )
            else:
                st.warning("No volcanoes match your selected criteria.")

        with tab2:
            st.header("Volcano Statistics by Region")

            # [ST4] Radio buttons for region selection with custom design
            st.markdown("""
            <style>
            div.row-widget.stRadio > div {
                flex-direction: row;
                flex-wrap: wrap;
            }
            div.row-widget.stRadio > div[role="radiogroup"] > label {
                margin-right: 2rem;
            }
            </style>
            """, unsafe_allow_html=True)

            regions = sorted(df['volcanic_region'].dropna().unique().tolist())
            if not regions:
                st.warning("No volcanic region data found in the dataset.")
                selected_region = "All"
            else:
                selected_region = st.radio("Select a volcanic region:", ["All"] + regions)

            region_df = df if selected_region == "All" else df[df['volcanic_region'] == selected_region]

            # [DA4] Filter data by one condition
            active_eruptions = region_df[region_df['last_known_eruption'].str.contains('CE', na=False)]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Volcano Types Distribution")

                # [VIZ1] Pie chart of volcano types
                if not region_df.empty and 'primary_volcano_type' in region_df.columns:
                    type_counts = region_df['primary_volcano_type'].value_counts().head(6)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    wedges, texts, autotexts = ax.pie(
                        type_counts,
                        labels=type_counts.index,
                        autopct='%1.1f%%',
                        textprops={'fontsize': 10},
                        colors=plt.cm.tab10.colors
                    )

                    plt.setp(texts, size=9, weight="bold")
                    plt.setp(autotexts, size=9, weight="bold")

                    ax.set_title(f'Volcano Types in {selected_region if selected_region != "All" else "All Regions"}')
                    st.pyplot(fig)
                else:
                    st.write("No data available for volcano types in this region.")

            with col2:
                st.subheader("Volcano Elevations")

                # [VIZ2] Histogram of elevations
                if not region_df.empty and 'elevation_(m)' in region_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 8))

                    elevations = region_df['elevation_(m)'].dropna()
                    ax.hist(elevations, bins=20, color='skyblue', edgecolor='black')

                    ax.set_xlabel('Elevation (meters)')
                    ax.set_ylabel('Number of Volcanoes')
                    ax.set_title(
                        f'Elevation Distribution in {selected_region if selected_region != "All" else "All Regions"}')

                    avg_elev = elevations.mean()
                    ax.axvline(x=avg_elev, color='red', linestyle='dashed', linewidth=2)
                    ax.text(avg_elev * 1.05, ax.get_ylim()[1] * 0.9, f'Avg: {avg_elev:.0f}m', color='red')

                    st.pyplot(fig)
                else:
                    st.write("No elevation data available for this region.")

            # [VIZ3] Bar chart of volcanic activity by region
            st.subheader("Recent Volcanic Activity")
            if not region_df.empty:
                if selected_region == "All":
                    top_countries = df['country'].value_counts().head(10)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(top_countries.index, top_countries.values, color='orange')

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                f'{height:.0f}', ha='center', va='bottom')

                    plt.xticks(rotation=45, ha='right')
                    ax.set_xlabel('Country')
                    ax.set_ylabel('Number of Volcanoes')
                    ax.set_title('Top 10 Countries by Number of Volcanoes')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    if 'Activity_Evidence' in region_df.columns:
                        activity_counts = region_df['Activity_Evidence'].value_counts()

                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(activity_counts.index, activity_counts.values, color='orange')

                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                    f'{height:.0f}', ha='center', va='bottom')

                        plt.xticks(rotation=45, ha='right')
                        ax.set_xlabel('Activity Evidence')
                        ax.set_ylabel('Number of Volcanoes')
                        ax.set_title(f'Volcanic Activity Evidence in {selected_region}')

            # [DA5] Create a filter with two conditions
            recent_eruptions = region_df[
                (region_df['last_known_eruption'].str.contains('CE', na=False)) &
                (region_df['activity_evidence'] == 'Historical')
                ] if not region_df.empty else pd.DataFrame()

            count = len(region_df)

            if 'elevation_m' in region_df.columns and pd.api.types.is_numeric_dtype(region_df['elevation_m']):
                avg_elev = region_df['elevation_m'].mean()
            else:
                avg_elev = 0

            if 'primary_volcano_type' in region_df.columns:
                common_type = region_df['primary_volcano_type'].mode()[0]
            else:
                common_type = "Unknown"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Volcanoes", f"{count}")
            col2.metric("Avg. Elevation", f"{avg_elev:.0f}m")
            col3.metric("Most Common Type", f"{common_type}")
            col4.metric("Recent Eruptions", f"{len(recent_eruptions)}")

        with tab3:
            st.header("Eruption History Analysis")

            def categorize_eruption(eruption_str):
                if pd.isna(eruption_str):
                    return "Unknown"
                elif "BCE" in str(eruption_str):
                    return "Ancient (BCE)"
                elif "CE" in str(eruption_str):
                    try:
                        year_str = str(eruption_str).split(" CE")[0]
                        year = int(year_str)
                        if year < 1500:
                            return "Historic (Pre-1500 CE)"
                        elif year < 1900:
                            return "Early Modern (1500-1899)"
                        elif year < 2000:
                            return "20th Century"
                        else:
                            return "21st Century"
                    except:
                        return "Other Historical"
                else:
                    if any(x in str(eruption_str).lower() for x in ["holocene", "pleistocene", "unknown", "uncertain"]):
                        return "Geologic Record"
                    else:
                        return "Other"

            df['eruption_category'] = [categorize_eruption(x) for x in df['last_known_eruption']]

            eruption_pivot = pd.pivot_table(
                df,
                index='volcanic_region',
                columns='eruption_category',
                values='volcano_name',
                aggfunc='count',
                fill_value=0
            )

            eruption_pivot['Total'] = eruption_pivot.sum(axis=1)
            eruption_pivot = eruption_pivot.sort_values('Total', ascending=False)

            display_pivot = eruption_pivot.drop(columns=['Total'])

            st.subheader("Eruption Timeline by Region")
            st.write("This table shows the distribution of last known eruptions across different time periods.")
            st.dataframe(display_pivot, use_container_width=True)

            st.subheader("Global Eruption Timeline")

            eruption_counts = df['eruption_category'].value_counts()
            order = [
                "Ancient (BCE)",
                "Historic (Pre-1500 CE)",
                "Early Modern (1500-1899)",
                "20th Century",
                "21st Century",
                "Geologic Record",
                "Unknown",
                "Other"
            ]
            ordered_counts = pd.Series({cat: eruption_counts.get(cat, 0) for cat in order if cat in eruption_counts})

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(
                ordered_counts.index,
                ordered_counts.values,
                color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(ordered_counts)))
            )

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{height:.0f}', ha='center', va='bottom')

            ax.set_xlabel('Eruption Period')
            ax.set_ylabel('Number of Volcanoes')
            ax.set_title('Distribution of Last Known Eruptions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            st.pyplot(fig)

            st.subheader("Recent Eruptions (21st Century)")
            recent = df[df['eruption_category'] == "21st Century"].sort_values('last_known_eruption', ascending=False)

            if not recent.empty:
                st.dataframe(
                    recent[['volcano_name', 'country', 'last_known_eruption', 'primary_volcano_type']],
                    use_container_width=True
                )
            else:
                st.write("No 21st century eruptions found in the dataset.")

        with tab4:
            st.header("Comparative Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # [DA7] Group data by region and calculate statistics
                region_stats = df.groupby('volcanic_region').agg({
                    'volcano_name': 'count',
                    'elevation_(m)': ['mean', 'max', 'min']
                })

                # Flatten the multi-index columns
                region_stats.columns = ['volcano_count', 'avg_elevation', 'max_elevation', 'min_elevation']
                region_stats = region_stats.reset_index()

                # Sort by count
                region_stats = region_stats.sort_values('volcano_count', ascending=False)

                st.subheader("Volcano Counts by Region")

                # [VIZ4] Table visualization
                st.dataframe(
                    region_stats,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                # Compare rock types across regions
                st.subheader("Dominant Rock Types")

                # [DA8] Iterate through rows to analyze rock types
                # Use iterrows for demonstration
                rock_type_data = {}

                for _, row in df.iterrows():
                    rock_type = row.get('dominant_rock_type', 'Unknown')
                    region = row.get('volcanic_region', 'Unknown')

                    if pd.isna(rock_type) or pd.isna(region):
                        continue

                    if region not in rock_type_data:
                        rock_type_data[region] = {}

                    if rock_type not in rock_type_data[region]:
                        rock_type_data[region][rock_type] = 0

                    rock_type_data[region][rock_type] += 1

                # Convert to DataFrame for easier manipulation
                rock_df_list = []
                for region, rocks in rock_type_data.items():
                    for rock, count in rocks.items():
                        rock_df_list.append({
                            'Region': region,
                            'Rock Type': rock,
                            'Count': count
                        })

                rock_df = pd.DataFrame(rock_df_list)

                # Get top 5 rock types overall
                top_rocks = rock_df.groupby('Rock Type')['Count'].sum().nlargest(5).index.tolist()

                # Filter to only these rock types
                filtered_rock_df = rock_df[rock_df['Rock Type'].isin(top_rocks)]

                # Group by region and rock type
                pivot_rock = filtered_rock_df.pivot_table(
                    index='Region',
                    columns='Rock Type',
                    values='Count',
                    fill_value=0
                )

                # Display the table
                st.dataframe(pivot_rock, use_container_width=True)

            # [DA9] Add a calculated column - volcano density by elevation tier
            st.subheader("Volcano Distribution by Elevation Tiers")

            # Create elevation tiers
            def elevation_tier(elevation):
                if elevation < 1000:
                    return "Low (<1000m)"
                elif elevation < 2000:
                    return "Medium (1000-2000m)"
                elif elevation < 4000:
                    return "High (2000-4000m)"
                else:
                    return "Very High (>4000m)"

            df['elevation_tier'] = df['elevation_(m)'].apply(elevation_tier)

            # Count volcanoes by elevation tier and region
            elevation_distribution = pd.crosstab(df['volcanic_region'], df['elevation_tier'])

            # Sort regions by total number of volcanoes
            elevation_distribution['Total'] = elevation_distribution.sum(axis=1)
            elevation_distribution = elevation_distribution.sort_values('Total', ascending=False).drop(
                columns=['Total'])

            # Create a stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 8))

            # Select top 10 regions by volcano count
            top_regions = df['volcanic_region'].value_counts().nlargest(10).index
            plot_data = elevation_distribution.loc[top_regions]

            # Create the stacked bar chart
            plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

            ax.set_xlabel('Volcanic Region')
            ax.set_ylabel('Number of Volcanoes')
            ax.set_title('Volcano Distribution by Elevation Tier Across Top 10 Regions')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Elevation Tier')
            plt.tight_layout()

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("Please ensure your 'volcanoes.csv' file is in the correct location and format.")


if __name__ == "__main__":
    main()