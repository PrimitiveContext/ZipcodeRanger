import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def haversine(lat1, lng1, lat2, lng2):
    R = 3958.8  # Earth's radius in miles
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_uszips_data(file_path):
    df = pd.read_csv(file_path, usecols=[
        'zip', 'lat', 'lng', 'city', 'state_name', 'population', 'density', 'population_x_density'
    ])
    df['zip'] = df['zip'].apply(lambda x: str(x).zfill(5))  # Ensure 5-digit ZIP codes
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
    return df

def circle_intersection_area(R, d):
    """
    Compute the area of intersection of two identical circles of radius R 
    whose centers are separated by distance d.
    """
    if d >= 2*R:
        # Circles do not overlap
        return 0.0
    if d == 0:
        # Circles are coincident; intersection is area of one circle
        return math.pi * R**2
    
    # Formula for intersection area of two circles of same radius:
    # area = 2 * R² * arccos(d/(2R)) - (d/2)*√(4R² - d²)
    term = d / (2*R)
    area = 2 * R**2 * math.acos(term) - (d/2)*math.sqrt(4*R**2 - d**2)
    return area

def compute_coverage_for_all(zip_data, radius_miles):
    zips = zip_data['zip'].values
    lats = zip_data['lat'].values
    lngs = zip_data['lng'].values
    pops = zip_data['population'].values
    cities = zip_data['city'].values
    states = zip_data['state_name'].values

    sorted_df = zip_data.sort_values(by='zip')
    sorted_zips = sorted_df['zip'].values
    sorted_lats = sorted_df['lat'].values
    sorted_lngs = sorted_df['lng'].values
    sorted_pops = sorted_df['population'].values

    coverage_info = []
    for i in tqdm(range(len(zips)), desc=f"Computing coverage for radius={radius_miles} miles"):
        parent_zip = zips[i]
        parent_lat = lats[i]
        parent_lng = lngs[i]
        parent_city = cities[i]
        parent_state = states[i]

        lat_tolerance = radius_miles / 69.0
        cos_lat = math.cos(math.radians(parent_lat)) or 1e-9
        lng_tolerance = radius_miles / (69.0 * cos_lat)

        candidate_mask = (
            (sorted_lats >= parent_lat - lat_tolerance) &
            (sorted_lats <= parent_lat + lat_tolerance) &
            (sorted_lngs >= parent_lng - lng_tolerance) &
            (sorted_lngs <= parent_lng + lng_tolerance)
        )

        candidate_zips = sorted_zips[candidate_mask]
        candidate_lats = sorted_lats[candidate_mask]
        candidate_lngs = sorted_lngs[candidate_mask]
        candidate_pops = sorted_pops[candidate_mask]

        total_population = 0.0
        child_zipcodes = []

        for cz, clat, clng, cpop in zip(candidate_zips, candidate_lats, candidate_lngs, candidate_pops):
            dist = haversine(parent_lat, parent_lng, clat, clng)
            if dist <= radius_miles:
                child_zipcodes.append(str(cz).zfill(5))  # Pad child ZIP codes to 5 digits
                total_population += float(cpop or 0)

        coverage_info.append({
            'parent_zip': str(parent_zip).zfill(5),
            'parent_city': parent_city,
            'parent_state': parent_state,
            'parent_lat': parent_lat,
            'parent_lng': parent_lng,
            'child_zipcodes': child_zipcodes,
            'total_population': float(total_population)
        })

    return coverage_info

if __name__ == "__main__":
    file_path = "uszips.csv"
    print("Loading data from uszips.csv...")
    zip_data = load_uszips_data(file_path)
    print("Data loaded successfully!")

    radii_input = input("Enter the radii in miles (separated by spaces or commas): ").strip()
    radii_input = radii_input.replace(',', ' ')
    radii_str_list = radii_input.split()

    radii_list = []
    for r_str in radii_str_list:
        try:
            r_val = float(r_str)
            if r_val < 0:
                raise ValueError("Radius cannot be negative.")
            radii_list.append(r_val)
        except ValueError:
            print(f"Invalid radius: {r_str}. Skipping this value.")
            continue

    if not radii_list:
        print("No valid radii provided. Exiting.")
        exit(1)

    MAX_OVERLAP_RATIO = 0.1  # 10% overlap allowed max with any single existing parent
    for radius_miles in radii_list:
        print(f"\nComputing coverage for radius: {radius_miles} miles...")
        coverage_list = compute_coverage_for_all(zip_data, radius_miles)

        for entry in coverage_list:
            entry['total_population'] = float(entry['total_population'] or 0.0)

        # Sort by total_population descending
        coverage_list.sort(key=lambda x: x['total_population'], reverse=True)

        chosen_parents = []
        used_zipcodes = set()

        # Circle area for given radius
        circle_area = math.pi * (radius_miles ** 2)

        for entry in coverage_list:
            pzip = entry['parent_zip']

            # If this parent is already used, skip
            if pzip in used_zipcodes:
                continue

            new_child_zips = [cz for cz in entry['child_zipcodes'] if cz not in used_zipcodes]
            if not new_child_zips:
                continue

            # Check overlap constraint with existing parents
            candidate_lat = entry['parent_lat']
            candidate_lng = entry['parent_lng']

            # Compute overlap with each chosen parent
            acceptable = True
            for cp in chosen_parents:
                dist_centers = haversine(candidate_lat, candidate_lng, cp['parent_lat'], cp['parent_lng'])
                overlap_area = circle_intersection_area(radius_miles, dist_centers)
                if overlap_area > MAX_OVERLAP_RATIO * circle_area:
                    # More than 10% overlap with this existing parent
                    acceptable = False
                    break

            if not acceptable:
                continue

            # If acceptable, select this parent
            chosen_parents.append({
                'parent_zip': pzip,
                'parent_city': entry['parent_city'],
                'parent_state': entry['parent_state'],
                'parent_lat': entry['parent_lat'],
                'parent_lng': entry['parent_lng'],
                'child_zipcodes': new_child_zips,
                'total_population': entry['total_population']
            })

            used_zipcodes.add(pzip)
            used_zipcodes.update(new_child_zips)

        final_count = len(chosen_parents)
        csv_filename = f"{int(radius_miles)}Miles_{final_count}ParentZips.csv"

        rows = []
        for idx, p in enumerate(chosen_parents, start=1):
            rows.append({
                'index': idx,
                'parent_zip': p['parent_zip'],
                'state': p['parent_state'],
                'city': p['parent_city'],
                'population_sum': p['total_population'],
                'child_zip_count': len(p['child_zipcodes']),
                'child_zipcode_list': ", ".join(p['child_zipcodes']),
                'radius_miles': radius_miles
            })

        df_out = pd.DataFrame(rows)
        df_out.to_csv(csv_filename, index=False)
        print(f"CSV file created: {csv_filename}")

        if chosen_parents:
            population_sums = [entry['total_population'] for entry in chosen_parents]
            indices = list(range(len(population_sums)))
            cumulative_pop = np.cumsum(population_sums)
            total_pop = cumulative_pop[-1]
            slice_points = []

            for i in range(1, 11):  # Create slices at 10%, 20%, ..., 100%
                target_pop = total_pop * (i / 10)
                slice_index = next((j for j, v in enumerate(cumulative_pop) if v >= target_pop), None)
                if slice_index is not None:
                    slice_points.append((slice_index, i * 10))  # (index, percentile)

            # Create a figure with grid spec for graph and table
            fig = plt.figure(figsize=(14, 6))
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)

            # Graph plot
            ax_graph = fig.add_subplot(gs[0])
            ax_graph.plot(indices, population_sums, marker='o', linestyle='-', label=f'Coverage Curve (Radius: {radius_miles} miles)')

            ax_graph.set_ylim(0, max(population_sums)*1.2)

            for sp_index, percentile in slice_points:
                line_height = max(population_sums) * 1.1
                ax_graph.axvline(x=sp_index, color='red', linestyle='--')
                ax_graph.text(sp_index, line_height, f'{percentile}%', color='red', fontsize=10,
                              horizontalalignment='center', verticalalignment='bottom')

            ax_graph.set_xlabel('Index')
            ax_graph.set_ylabel('Population Sum')
            ax_graph.set_title(f'Parent ZIP Code Coverage Curve (Radius: {radius_miles} miles)')
            ax_graph.grid(True)
            ax_graph.legend()

            # Table plot
            ax_table = fig.add_subplot(gs[1])
            ax_table.axis('off')
            table_data = [[f"{percentile}%", sp_index] for sp_index, percentile in slice_points]
            column_labels = ["Percentile", "Index"]
            ax_table.table(cellText=table_data, colLabels=column_labels, loc='center')

            png_filename = f"Curve_{csv_filename.replace('.csv', '.png')}"
            plt.savefig(png_filename, format='png')
            print(f"Graph and table saved as: {png_filename}")
            plt.close()
