import pandas as pd
df=pd.read_csv("datasets\dataset-1.csv")
def generate_car_matrix(df)->pd.DataFrame:

    
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    car_matrix=df.pivot(index="id_1",columns="id_2",values="car").fillna(0)

    return car_matrix


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    dict1=df['car'].value_counts().to_dict()    
    return dict1


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_mean=df["bus"].mean()
    bus_indexes=df[df["bus"] > 2 * bus_mean].index().tolist()

    return bus_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    route_avg=df.groupby("route")["truck"].mean()
    new_route=route_avg[route_avg > 7 ].list.tolist()
    new_route.sort()

    return new_route


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25).round(1)
    return modified_matrix
    

    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """

    df['startDateTime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S', errors='coerce')
    df['endDateTime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S', errors='coerce')

    df = df.dropna(subset=['startDateTime', 'endDateTime'])

    incorrect_timestamps = []

    unique_pairs = df[['id', 'id_2']].drop_duplicates()
    for index, row in unique_pairs.iterrows():

        current_rows = df[(df['id'] == row['id']) & (df['id_2'] == row['id_2'])]

        time_diffs = (current_rows['endDateTime'] - current_rows['startDateTime'])
        full_24_hours = all(diff == timedelta(hours=24) for diff in time_diffs)

        unique_days = set(current_rows['startDateTime'].dt.day_name()).union(set(current_rows['endDateTime'].dt.day_name()))
        all_days_covered = len(unique_days) == 7


        incorrect_timestamps.append({'id': row['id'], 'id_2': row['id_2'], 'result': not full_24_hours or not all_days_covered})

    result_df = pd.DataFrame(incorrect_timestamps).set_index(['id', 'id_2'])['result']

    return result_df

df = pd.read_csv("datasets\dataset-2.csv")

result = time_check(df)

print(result)

