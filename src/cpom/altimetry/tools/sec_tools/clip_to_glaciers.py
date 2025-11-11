"""
cpom.altimetry.tools.sec_tools.clip.py

Purpose:
  Clip epoch data to glacier outlines from shapefile.
  Output clipped epoch data as parquet files.

  Optionally plot clipped data by epoch.
"""

import argparse
import json
import os
import sys
import time
from logging import Logger
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl

from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def get_shapefile(grid_area: GridArea, params: argparse.Namespace, logger: Logger) -> tuple:
    """
    Load a shapefile.
    Currently supports :
        Mouginot glacier outlines for Greenland.
        IMBIE2 basins for Greenland and Antarctica.
    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            shapefile (str): Shapefile option to load.
            [mouginot, green_basins_imbie2, ant_basins_imbie2]
        logger (Logger): Logger object.
    Returns:
        tuple: (shapefile path, selection column name)
    """

    if params.shapefile == "mouginot_glaciers":
        shapefile = (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
            "mouginot_subbasins/Greenland_Basins_PS_v1.4.2.shp"
        )
        selection_col = "NAME"
        shp = gpd.read_file(shapefile)
        shp = shp.to_crs(grid_area.crs_bng)
        return shp, selection_col
    if params.shapefile == "green_basins_imbie2":
        shapefile = (
            f"{os.environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/"
            "GRE_Basins_IMBIE2_v1.3/shpfiles/GRE_IceSheet_IMBIE2_v1.3.shp"
        )
        selection_col = "SUBREGION1"
        shp = gpd.read_file(shapefile)
        shp = shp.to_crs(grid_area.crs_bng)
        return shp, selection_col
    if params.shapefile == "ant_basins_imbie2":
        shapefile = (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/antarctica/'
            "rignot_2016_imbie2_ant_grounded_icesheet_basins/data/ANT_Basins_IMBIE2_v1.6.shp"
        )
        selection_col = "SUBREGION1"
        shp = gpd.read_file(shapefile)
        shp = shp.to_crs(grid_area.crs_bng)
        return shp, selection_col

    logger.info(
        "Shapefile option not recognised. Please choose from : [mouginot_glaciers,"
        " mouginot_basins]"
    )
    return None, None


def get_epoch_data(grid_area: GridArea, params: argparse.Namespace, logger: Logger) -> pl.DataFrame:
    """
    Load epoch averaged data, compute lat/lon coordinates for each bin.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            in_dir (str): Input directory containing epoch averaged parquet files.
        logger (Logger): Logger object.
    Returns:
        pl.DataFrame: Epoch averaged data with lat/lon coordinates.
    """
    in_file = Path(params.in_dir) / "*.parquet"
    logger.info(f"Loading epoch data from: {in_file}")
    epoch_data = pl.read_parquet(in_file)

    unique_cells = epoch_data.select(["x_bin", "y_bin"]).unique()
    lats, lons = grid_area.get_cellcentre_lat_lon_from_col_row(
        unique_cells.select(pl.col("x_bin")).to_numpy().flatten(),
        unique_cells.select(pl.col("y_bin")).to_numpy().flatten(),
    )

    in_data_with_coords = epoch_data.join(
        unique_cells.with_columns(
            [
                pl.Series("lat", lats),
                pl.Series("lon", lons),
            ]
        ),
        on=["x_bin", "y_bin"],
        how="left",
    )
    return in_data_with_coords


def clip_to_glaciers(grid_area: GridArea, params: argparse.Namespace, logger: Logger):
    """
    Clip epoch data to glacier outlines from shapefile.
    Write clipped data to parquet files.
    Optionally plot clipped data by epoch.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            selector_value (List[str]): Specific glacier names or 'all'.
            out_dir (str): Output directory for clipped data.
            plot_by_epoch (bool): Whether to plot clipped data by epoch.
        logger (Logger): Logger object.
    """
    # Load shapefile
    shp, selector = get_shapefile(grid_area, params, logger)
    epoch_data = get_epoch_data(grid_area, params, logger)
    data_gdf = gpd.GeoDataFrame(
        epoch_data.to_pandas(),
        geometry=gpd.points_from_xy(epoch_data["lon"], epoch_data["lat"]),
        crs=grid_area.crs_wgs,
    ).to_crs(grid_area.crs_bng)

    glacier_names = (
        set(shp[selector]) if params.selector_value == ["all"] else params.selector_value
    )
    for name in sorted(glacier_names):
        logger.info(f"Clipping to glacier: {name}")
        glacier_shape = shp[shp[selector] == name]
        clipped_gdf = gpd.sjoin(data_gdf, glacier_shape, how="inner", predicate="within")

        # Convert back to polars, but first clean up the dataframe
        # Remove geometry column and any extra columns from the spatial join
        clipped_pandas = clipped_gdf.drop(
            columns=["geometry", "index_right", "lat", "lon"], errors="ignore"
        )

        # Only keep columns that exist in the original epoch_data
        available_columns = [col for col in epoch_data.columns if col in clipped_pandas.columns]
        clipped_pandas_clean = clipped_pandas[available_columns]

        # Convert to polars and remove lat/lon columns if they exist
        clipped_data = pl.from_pandas(clipped_pandas_clean)

        os.makedirs(Path(params.out_dir) / name, exist_ok=True)
        if params.plot_by_epoch:
            plot_clipped_data(clipped_data, glacier_shape, params, name, grid_area)

        logger.info(
            f"Writing clipped data to: {Path(params.out_dir) / name / 'epoch_average.parquet'}"
        )
        pl.LazyFrame(clipped_data).sink_parquet(
            Path(params.out_dir) / name / "epoch_average.parquet"
        )


def plot_clipped_data(
    clipped_data: pl.DataFrame,
    glacier_shp: gpd.GeoDataFrame,
    params: argparse.Namespace,
    glacier_name: str,
    grid_area: GridArea,
):
    """
    Plot clipped data by epoch.
    Args:
        clipped_data (pl.DataFrame): Clipped epoch data.
        glacier_shp (gpd.GeoDataFrame): Shapefile GeoDataFrame.
        params (argparse.Namespace): Command line parameters.
            Includes:
            epoch_plotting_column (str): Column name for epoch to group by for plot.
            dh_plotting_column (str): Column name for dh to plot.
        glacier_name (str): Name of the glacier.
        grid_area (GridArea): CPOM GridArea object.
    """
    x, y = grid_area.get_cellcentre_x_y_from_col_row(clipped_data["x_bin"], clipped_data["y_bin"])
    clipped_data = clipped_data.with_columns(
        [
            pl.Series("x", x),
            pl.Series("y", y),
        ]
    )
    # Get unique epochs
    unique_epochs = clipped_data[params.epoch_plotting_column].unique()
    for epoch in sorted(unique_epochs):
        epoch_data = clipped_data.filter(pl.col(params.epoch_plotting_column) == epoch)
        _, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            epoch_data["x"],
            epoch_data["y"],
            c=epoch_data[params.dh_plotting_column],
            cmap="coolwarm",  # Reverse colormap to match reference colors
            vmin=-1,
            vmax=1,
            s=10,
            alpha=0.7,
        )

        plt.colorbar(scatter, ax=ax, label="dh (m)")

        # Overlay glacier boundaries
        glacier_shp.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(
            Path(params.out_dir) / glacier_name / f"clipped_{glacier_name}_epoch_{epoch}.png",
            dpi=300,
        )
        plt.close()


def get_metadata_json(params: argparse.Namespace, start_time, logger: Logger):
    """
    Generate metadata JSON for clipped data.
    Args:
        params (argparse.Namespace): Command line parameters.
            Includes:
            Command line parameters
            Processing time
    """
    meta_json_path = Path(params.out_dir) / "metadata.json"
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(params),
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
        logger.info("Wrote data_set metadata to %s", meta_json_path)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


def main(args):
    """
    Main function to parse arguments and run clipping tool.
    Steps:
    1. Parse command line arguments.
    2. Load grid metadata from JSON file.
    3. Create output directory and set up logging.
    4. Call clip_to_glaciers function to perform clipping and optional plotting.
    """
    parser = argparse.ArgumentParser(
        description="Clip altimetry data to glacier outlines from shapefile"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input gridded altimetry data" "e.g. from epoch_average.parquet",
    )
    parser.add_argument(
        "--out_dir",
        help="Path of output directory for clipped_epochs results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the grid metadata JSON file.",
        required=True,
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        required=True,
        help="Shapefile to use for clipping : " "Options : mouginot_glaciers, mouginot_basins",
        # "ant_basins_imbie2 , green_basins_imbie2",
    )
    parser.add_argument(
        "--selector_value",
        nargs="+",
        default=["all"],
        help="Select specific glaciers or 'all' for all glaciers.",
    )
    parser.add_argument(
        "--plot_by_epoch",
        action="store_false",
        help="Whether to plot the clipped data by epoch.",
    )
    parser.add_argument(
        "--epoch_plotting_column",
        type=str,
        default="epoch_number",
        help="Column name for epoch in the data to plot.",
    )
    parser.add_argument(
        "--dh_plotting_column",
        type=str,
        default="dh_ave",
        help="Column name for dh to plot.",
    )
    start_time = time.time()
    params = parser.parse_args(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
    )

    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    grid_area = GridArea(grid_meta["gridarea"], grid_meta["binsize"])

    clip_to_glaciers(grid_area, params, logger)

    get_metadata_json(params, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
