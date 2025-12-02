"""
cpom.altimetry.tools.sec_tools.clip_to_basins

Purpose:
    Clip epoch-averaged altimetry data to glacier or basin boundaries from shapefiles.

    Takes gridded elevation change data and spatially clips it to specific regions
    (glaciers, drainage basins, etc.) defined in shapefiles. Outputs clipped data
    as parquet files, optionally with plots showing the spatial distribution.

Supported Shapefiles:
    - mouginot_glaciers: Greenland glacier outlines (Mouginot et al.)
    - green_basins_imbie2: Greenland IMBIE2 drainage basins
    - ant_basins_imbie2: Antarctic IMBIE2 drainage basins (two-tier: Regions/Subregions)

Output:
    - Clipped data: <out_dir>/<basin>/epoch_average.parquet
    - Optional plots: <out_dir>/<basin>/clipped_<basin>_epoch_{number}.png
    - Metadata: <out_dir>/metadata.json
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


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for clipping altimetry data to basins.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: _description_
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
        help="Shapefile to use for clipping : "
        "Options : mouginot_glaciers, ant_basins_imbie2, green_basins_imbie2",
    )
    parser.add_argument(
        "--region_selector",
        nargs="+",
        default=["all"],
        help="Select regions to process. Use 'all' to process all available regions. "
        "Ignore for root level data",
    )
    parser.add_argument(
        "--subregion_selector",
        nargs="+",
        default=["all"],
        help="For two-tier shapefiles only (e.g., ant_basins_imbie2): "
        "Select specific subregions within each region "
        "(e.g., H-Hp, F-G, A-Ap) or 'all' for all subregions. "
        "Ignored for single-tier shapefiles.",
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

    return parser.parse_args(args)


def get_basin_geometry_from_shapefile(
    shp: gpd.GeoDataFrame,
    selector: tuple | str,
    basin_path: str,
) -> gpd.GeoDataFrame:
    """
    Extract geometry for a specific basin or subbasin from a shapefile.

    Filters a shapefile GeoDataFrame to return only the geometry for the specified
    basin. Handles both single-tier (direct basin names) and two-tier
    (region/subregion hierarchy) structures.

    Args:
        shp (gpd.GeoDataFrame): Full shapefile with all basin geometries.
        selector (tuple | str): Column name(s) for filtering.
            - Single-tier: "column_name" (e.g., "NAME")
            - Two-tier: ("region_column", "subregion_column") (e.g., ("Regions", "Subregion"))
        basin_path (str): Path identifying the basin.
            - Single-tier: "basin_name" (e.g., "Jakobshavn")
            - Two-tier: "region/subregion" (e.g., "West/H-Hp")
            - Special case: "region/none" for regions without subregions (e.g., Islands)

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame for the specified basin.
    """

    if isinstance(selector, tuple):
        # Two-tier structure: region/subregion
        region_col, subregion_col = selector
        if "/" in basin_path:
            region, subregion = basin_path.split("/", 1)
            region_shp = shp[shp[region_col] == region]
            # Handle regions without subregions (NaN values)
            if subregion.lower() == "none" or subregion == "":
                return region_shp[region_shp[subregion_col].isna()]
            return region_shp[region_shp[subregion_col] == subregion]
        # Just region, no subregion specified
        return shp[shp[region_col] == basin_path]
    # Single-tier structure
    return shp[shp[selector] == basin_path]


def get_shapefile(grid_area: GridArea, params: argparse.Namespace, logger: Logger | None) -> tuple:
    """
    Load and reproject a shapefile for glacier or basin boundaries.

    Loads one of the supported shapefiles and reprojects it to match the grid area's
    coordinate reference system (CRS).

    Supported shapefiles:
        - mouginot_glaciers: Greenland glacier outlines (Mouginot et al. v1.4.2)
        - green_basins_imbie2: Greenland IMBIE2 drainage basins (v1.3)
        - ant_basins_imbie2: Antarctic IMBIE2 drainage basins (Rignot 2016, v1.6)

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            shapefile (str): Shapefile option to load.
            [mouginot, green_basins_imbie2, ant_basins_imbie2]
        logger (Logger): Logger object.
    Returns:
        tuple: (GeoDataFrame, selection column name(s))
            For single-tier: (shp, "column_name")
            For two-tier: (shp, ("region_column", "subregion_column"))
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
        shp = gpd.read_file(shapefile)
        shp = shp.to_crs(grid_area.crs_bng)
        return shp, ("Regions", "Subregion")  # Two-tier selection

    logger.info(
        "Shapefile option not recognised. Please choose from : [mouginot_glaciers,"
        " mouginot_basins]"
    )
    return None, None


def get_epoch_data(grid_area: GridArea, params: argparse.Namespace, logger: Logger) -> pl.DataFrame:
    """
    Load epoch-averaged elevation change data and add geographic coordinates.

    Loads all parquet files from input directory and converts grid cell coordinates
    (x_bin, y_bin) to latitude/longitude using the grid area's coordinate system.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            - in_dir (str): Directory containing epoch_average*.parquet files
        logger (Logger): Logger for progress messages.

    Returns:
        pl.DataFrame: Epoch-averaged data with added 'lat' and 'lon' columns.
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


def clip_data_to_shape(
    data_gdf: gpd.GeoDataFrame,
    subregion_shape: gpd.GeoDataFrame,
    epoch_data: pl.DataFrame,
) -> pl.DataFrame:
    """Clip data to basin shape and convert back to polars DataFrame.

    Args:
        data_gdf (gpd.GeoDataFrame): GeoDataFrame of the data points with geometry.
        subregion_shape (gpd.GeoDataFrame): GeoDataFrame of the basin or subregion shape.
        epoch_data (pl.DataFrame): Original epoch-averaged data in polars DataFrame format.

    Returns:
        pl.DataFrame: Clipped data as a polars DataFrame.
    """
    clipped_gdf = gpd.sjoin(data_gdf, subregion_shape, how="inner", predicate="within")
    clipped_pandas = clipped_gdf.drop(
        columns=["geometry", "index_right", "lat", "lon"], errors="ignore"
    )
    available_columns = [col for col in epoch_data.columns if col in clipped_pandas.columns]
    return pl.from_pandas(clipped_pandas[available_columns])


def save_basin_data_and_plot(
    subregion_shape: gpd.GeoDataFrame,
    output_name: str,
    clipped_data: pl.DataFrame,
    params: argparse.Namespace,
    grid_area: GridArea,
):
    """
    Save clipped data and optionally generate plots for a single glacier/basin.

    Args:
        subregion_shape: GeoDataFrame for the glacier/basin geometry
        output_name: Name for output directory and files
        clipped_data: Clipped polars DataFrame for this basin
        params: Command line parameters (includes out_dir, plot_by_epoch, plotting columns)
        grid_area: GridArea object for coordinate conversion
    """
    output_dir = Path(params.out_dir) / output_name
    os.makedirs(output_dir, exist_ok=True)

    if params.plot_by_epoch:
        plot_clipped_data(clipped_data, subregion_shape, params, output_name, grid_area)

    pl.LazyFrame(clipped_data).sink_parquet(output_dir / "epoch_average.parquet")


# pylint: disable=R0914
def clip_to_basins(grid_area: GridArea, params: argparse.Namespace, logger: Logger):
    """
    Clip epoch-averaged elevation change data to glacier or basin boundaries.

    Main processing function that:
    1. Loads shapefile with basin boundaries
    2. Loads epoch-averaged elevation change data
    3. Performs spatial join to clip data to each basin
    4. Saves clipped data as parquet files
    5. Optionally generates plots for each epoch and basin

    Supports both single-tier (direct basin names) and two-tier (region/subregion)
    directory structures for output organization.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            - in_dir (str): Input directory with epoch data
            - out_dir (str): Output directory for clipped results
            - shapefile (str): Shapefile identifier
            - region_selector (list[str]): Regions/basins to process or ['all']
            - subregion_selector (list[str]): Subregions to process or ['all'] (two-tier only)
            - plot_by_epoch (bool): Whether to generate plots
            - epoch_plotting_column (str): Column for epoch grouping (if plotting)
            - dh_plotting_column (str): Column for elevation change values (if plotting)
        logger (Logger): Logger object.

    Output files:
        - <out_dir>/<basin>/epoch_average.parquet: Clipped elevation change data
        - <out_dir>/<basin>/clipped_<basin>_epoch_{n}.png: Optional plots (if plot_by_epoch=True)
    """

    # Load shapefile and epoch data
    shp, selector = get_shapefile(grid_area, params, logger)
    epoch_data = get_epoch_data(grid_area, params, logger)
    data_gdf = gpd.GeoDataFrame(
        epoch_data.to_pandas(),
        geometry=gpd.points_from_xy(epoch_data["lon"], epoch_data["lat"]),
        crs=grid_area.crs_wgs,
    ).to_crs(grid_area.crs_bng)

    if isinstance(selector, tuple):
        region_col, subregion_col = selector
    else:
        region_col = selector
        subregion_col = None

    for region in (
        set(shp[region_col]) if params.region_selector == ["all"] else params.region_selector
    ):
        logger.info(f"Processing region: {region}")
        if subregion_col is None:
            # Single-tier: process region directly
            region_shape = get_basin_geometry_from_shapefile(shp, selector, region)
            if not region_shape.empty:
                logger.info(f"Clipping to: {region}")
                clipped_data = clip_data_to_shape(data_gdf, region_shape, epoch_data)
                save_basin_data_and_plot(region_shape, region, clipped_data, params, grid_area)
                logger.info(f"Wrote: {Path(params.out_dir) / region / 'epoch_average.parquet'}")
        else:
            # Two-tier: iterate through subregions
            region_shp = shp[shp[region_col] == region]
            subregions = (
                set(region_shp[subregion_col])
                if params.subregion_selector == ["all"]
                else [s for s in params.subregion_selector if s in region_shp[subregion_col].values]
            )
            for subregion in subregions:
                # Handle basins without subregions (e.g., Islands)
                if subregion is None or (
                    isinstance(subregion, str) and subregion.lower() == "none"
                ):
                    output_name = region
                else:
                    output_name = f"{region}/{subregion}"

                subregion_shape = get_basin_geometry_from_shapefile(shp, selector, output_name)
                if not subregion_shape.empty:
                    logger.info(f"Clipping to: {output_name}")
                    clipped_data = clip_data_to_shape(data_gdf, subregion_shape, epoch_data)
                    save_basin_data_and_plot(
                        subregion_shape, output_name, clipped_data, params, grid_area
                    )
                    logger.info(
                        f"Wrote: {Path(params.out_dir) / output_name / 'epoch_average.parquet'}"
                    )


def plot_clipped_data(
    clipped_data: pl.DataFrame,
    glacier_shp: gpd.GeoDataFrame,
    params: argparse.Namespace,
    basin_name: str,
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
        basin_name (str): Name of the basin/subbasin .
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

        plot_path = Path(params.out_dir) / basin_name / f"clipped_{basin_name}_epoch_{epoch}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
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
    4. Call clip_to_basins function to perform clipping and optional plotting.
    """

    start_time = time.time()
    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
    )

    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    grid_area = GridArea(grid_meta["gridarea"], grid_meta["binsize"])

    clip_to_basins(grid_area, params, logger)

    get_metadata_json(params, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
