import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def haversine(lat1, lng1, lat2, lng2):
    R = 3958.8  # Earth's radius in miles
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlng / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_uszips_data(file_path):
    return pd.read_csv(file_path, usecols=['zip', 'lat', 'lng'])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script2.py <ParentZipsCSV>")
        exit(1)

    csv_file = sys.argv[1]
    try:
        parent_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        exit(1)

    total_parents = len(parent_df)
    print(f"This CSV contains {total_parents} parent ZIP codes.")
    display_input = input(f"How many of these {total_parents} do you want to display on the map? ").strip()
    try:
        display_count = int(display_input)
        if display_count < 1 or display_count > total_parents:
            raise ValueError
    except ValueError:
        print("Invalid number. Please choose a number between 1 and the total number of parents available.")
        exit(1)

    # Load all ZIP code coordinates to plot as black dots (optional)
    try:
        zip_data = load_uszips_data("uszips.csv")
    except FileNotFoundError:
        print("uszips.csv not found, will only plot parent ZIP codes.")
        zip_data = pd.DataFrame(columns=['zip', 'lat', 'lng'])

    # Subset the parent dataframe to the requested number
    display_df = parent_df.head(display_count)

    # For plotting the circles, we need lat/lng for each parent ZIP
    if zip_data.empty:
        print("No ZIP data available for plotting all ZIP codes or finding parent coords.")
        exit(1)

    zip_data_dict = zip_data.set_index('zip')[['lat', 'lng']].to_dict('index')

    # Map Setup
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    # Plot all ZIP codes as black dots if zip_data is available and not empty
    if not zip_data.empty:
        ax.scatter(zip_data['lng'], zip_data['lat'], s=1, color='black', transform=ccrs.PlateCarree(), alpha=0.5)

    theta = np.linspace(0, 2 * np.pi, 100)

    # Plot the chosen parent ZIP codes
    for idx, row in display_df.iterrows():
        pzip = int(row['parent_zip'])
        radius_miles = row['radius_miles']  # Use radius from the CSV
        pop_sum = row['population_sum']
        # Lookup lat/lng
        if pzip in zip_data_dict:
            plat = zip_data_dict[pzip]['lat']
            plng = zip_data_dict[pzip]['lng']
        else:
            # If not found, skip
            continue

        # Convert radius from miles to degrees
        radius_deg = radius_miles / 69.0

        # Compute circle coordinates
        circle_lats = plat + radius_deg * np.cos(theta)
        circle_lngs = plng + (radius_deg / np.cos(np.radians(plat))) * np.sin(theta)
        ax.plot(circle_lngs, circle_lats, color='red', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)

        # Place the parent index number at the center of the circle
        parent_index = int(row['index'])
        ax.text(plng, plat, str(parent_index), transform=ccrs.PlateCarree(),
                ha='center', va='center', color='red', fontsize=6, zorder=6)

    # Title and save the figure
    title = f"Top {display_count} Parent ZIP Codes (by Population)"
    plt.title(title)
    
    # Save the map as a PNG file with the same name as the input CSV
    output_file = csv_file.rsplit('.', 1)[0] + ".png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved as {output_file}")

    plt.show()
