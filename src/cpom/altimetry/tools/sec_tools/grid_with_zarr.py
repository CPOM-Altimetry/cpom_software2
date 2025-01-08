"""gridding test with zarr format"""

import numpy as np
import pandas as pd
import zarr


# Define grid function
def grid_to_dataframe(x, y, params, min_x, min_y, bin_size):
    """
    Grid the input data into bin_size x bin_size cells and store in a DataFrame.
    """
    grid_data = []
    for xi, yi, *vals in zip(x, y, *params.values()):
        # Determine which grid cell the point belongs to
        x_bin = int((xi - min_x) / bin_size)
        y_bin = int((yi - min_y) / bin_size)
        grid_data.append((x_bin, y_bin, xi, yi, *vals))

    # Create DataFrame
    columns = ["x_bin", "y_bin", "x", "y"] + list(params.keys())
    return pd.DataFrame(grid_data, columns=columns)


# Save the DataFrame to Zarr
def save_to_zarr(df, zarr_store, group_name="grid_data", mode="a"):
    """
    Save the gridded data to a Zarr store.
    If the Zarr store exists, append the data to the existing dataset.
    """
    # Open Zarr store
    z = zarr.open(zarr_store, mode=mode)  # 'a' append, 'w-' create, fail if exists

    # Check if the group exists; create if not
    if group_name not in z:
        group = z.create_group(group_name)
        group.create_dataset("x_bin", shape=(0,), chunks=(1000,), dtype="i4", data=[])
        group.create_dataset("y_bin", shape=(0,), chunks=(1000,), dtype="i4", data=[])
        group.create_dataset("x", shape=(0,), chunks=(1000,), dtype="f4", data=[])
        group.create_dataset("y", shape=(0,), chunks=(1000,), dtype="f4", data=[])
        for param in df.columns[4:]:
            group.create_dataset(param, shape=(0,), chunks=(1000,), dtype="f4", data=[])
    else:
        group = z[group_name]

    # Append data to the datasets
    for col in df.columns:
        group[col].append(df[col].values)


# Retrieve data from a specific grid cell
def get_values_from_cell(zarr_store, x_bin, y_bin, group_name="grid_data"):
    """
    Retrieve all values from the specified grid cell (x_bin, y_bin).
    """
    z = zarr.open(zarr_store, mode="r")
    group = z[group_name]

    # Load data into a DataFrame for easy filtering
    df = pd.DataFrame(
        {
            "x_bin": group["x_bin"][:],
            "y_bin": group["y_bin"][:],
            "x": group["x"][:],
            "y": group["y"][:],
        }
    )
    for param in group.keys():
        if param not in df.columns:
            df[param] = group[param][:]

    # Filter for the specific grid cell
    return df[(df["x_bin"] == x_bin) & (df["y_bin"] == y_bin)]


def main():
    """main function"""
    # Grid specifications
    min_x = -10
    min_y = -10
    bin_size = 2

    # Example data
    x = np.array([-9.8, -9.6, 4.6])
    y = np.array([-9.8, -9.6, 4.5])
    params = {
        "p1": np.array([1.0, 2.0, 3.0]),
        "p2": np.array([1.0, 1.0, 1.0]),
        "p3": np.array([2.0, 2.1, 0.8]),
    }

    # Grid the data
    df = grid_to_dataframe(x, y, params, min_x, min_y, bin_size)

    # Save to Zarr
    zarr_store = "gridded_data.zarr"
    save_to_zarr(df, zarr_store)

    # Add more data later
    x_new = np.array([1.2, -3.4])
    y_new = np.array([1.5, -3.0])
    params_new = {
        "p1": np.array([4.0, 5.0]),
        "p2": np.array([1.2, 1.3]),
        "p3": np.array([0.9, 1.1]),
    }
    new_df = grid_to_dataframe(x_new, y_new, params_new, min_x, min_y, bin_size)
    save_to_zarr(new_df, zarr_store)

    # Retrieve data from a specific cell
    x_bin_to_query = 0  # Example grid cell in x
    y_bin_to_query = 0  # Example grid cell in y
    cell_data = get_values_from_cell(zarr_store, x_bin_to_query, y_bin_to_query)

    print("Gridded Data:")
    print(pd.DataFrame(zarr.open(zarr_store, mode="r")["grid_data"]))  # All data
    print(f"\nData in cell ({x_bin_to_query}, {y_bin_to_query}):")
    print(cell_data)


# Example usage
if __name__ == "__main__":
    main()
