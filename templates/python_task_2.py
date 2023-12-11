import pandas as pd
import networkx as nx

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """

    G = nx.Graph()
    df['id_start'] = df['id_start'].astype(int)
    df['id_end'] = df['id_end'].astype(int)

    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])
    
    distance_matrix = pd.DataFrame(index=G.nodes, columns=G.nodes)
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 == node2:
                distance_matrix.at[node1, node2] = 0
            else:
                try:
                    distance_matrix.at[node1, node2] = nx.shortest_path_length(G, node1, node2, weight='weight')
                except nx.NetworkXNoPath:
                    distance_matrix.at[node1, node2] = float('inf')

    distance_matrix.index = distance_matrix.index.astype(int)
    pd.set_option('display.float_format', '{:.0f}'.format)
    return distance_matrix
df = pd.read_csv('datasets\dataset-3.csv')
result = calculate_distance_matrix(df)
print(result)


def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)

    unrolled_data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df
result_distance_matrix = calculate_distance_matrix(df)
unrolled_result = unroll_distance_matrix(result_distance_matrix)
print(unrolled_result)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_threshold = reference_avg_distance - (0.1 * reference_avg_distance)
    upper_threshold = reference_avg_distance + (0.1 * reference_avg_distance)

    similar_ids = df.groupby('id_start')['distance'].mean().reset_index()
    similar_ids = similar_ids[(similar_ids['distance'] >= lower_threshold) & (similar_ids['distance'] <= upper_threshold)]

    return similar_ids

df = pd.read_csv('datasets\dataset-3.csv') 
reference_id = df['id_start'].value_counts().idxmax()
result = find_ids_within_ten_percentage_threshold(df, reference_id)
print(result)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll'
        df[column_name] = df['distance'] * rate_coefficient

    return df
unrolled_df = unroll_distance_matrix(distance_matrix)  
toll_rate_df = calculate_toll_rate(unrolled_df)

result_distance_matrix = calculate_distance_matrix(df)
print("Result Distance Matrix:")
print(result_distance_matrix)
unrolled_result = unroll_distance_matrix(result_distance_matrix)
print("\nUnrolled Result:")
print(unrolled_result)



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)), (time(10, 0, 0), time(18, 0, 0)), (time(18, 0, 0), time(23, 59, 59))]
    weekend_time_ranges = [(time(0, 0, 0), time(23, 59, 59))]
    weekday_discount_factors = [0.8, 1.2, 0.8]
    result_df = pd.DataFrame()

    unique_pairs = df[['id_start', 'id_end']].drop_duplicates()
    for index, row in unique_pairs.iterrows():
        for day in range(7): 
            for start_time, end_time in weekday_time_ranges:
            
                start_datetime = datetime.combine(datetime.today(), start_time) + timedelta(days=day)
                end_datetime = datetime.combine(datetime.today(), end_time) + timedelta(days=day)
                
               
                mask = (df['id_start'] == row['id_start']) & (df['id_end'] == row['id_end']) & \
                       (df['start_time'] >= start_datetime.time()) & (df['end_time'] <= end_datetime.time())
                current_rows = df[mask].copy()
            
                if day < 5:  
                    for i, discount_factor in enumerate(weekday_discount_factors):
                        current_rows.loc[current_rows['start_time'].between(weekday_time_ranges[i][0], weekday_time_ranges[i][1]), 
                                         ['moto_toll', 'car_toll', 'rv_toll', 'bus_toll', 'truck_toll']] *= discount_factor
                else:
                    current_rows[['moto_toll', 'car_toll', 'rv_toll', 'bus_toll', 'truck_toll']] *= weekend_discount_factor

             
                current_rows['start_day'] = current_rows['end_day'] = (datetime.today() + timedelta(days=day)).strftime('%A')
                current_rows['start_time'] = start_time
                current_rows['end_time'] = end_time

                
                result_df = pd.concat([result_df, current_rows], ignore_index=True)

    return result_df
