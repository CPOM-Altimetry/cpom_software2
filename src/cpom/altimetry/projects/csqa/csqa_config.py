"""cpom.altimetry.projects.csqa.csqa_config

Load and validate the CSQA global configuration (csqa_config.yaml) and parameter definitions
(csqa_parameters.yaml).

The config file used is, in order of precedence:
    - the path passed to load_config()
    - the CSQA_CONFIG environment variable
    - config/csqa_config.yaml in this package
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import yaml  # type: ignore[import-untyped]

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config", "csqa_config.yaml")

PARAM_TYPES = ("flag", "float")


@dataclass(frozen=True)
class AreaConfig:
    """An area for which plots and statistics are produced"""

    id: str
    long_name: str
    polarplot_area: str
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class ProductConfig:
    """An input product archive"""

    id: str
    long_name: str
    dirs: list[str]
    file_glob: str


@dataclass(frozen=True)
class FlagDef:
    """A flag value of a flag parameter"""

    value: int
    name: str
    color: str | None
    key: str  # name sanitized for use in file/column names, ie "lake_enclosed_sea"


@dataclass(frozen=True)
class VariantDef:
    """A variable providing one variant of a parameter (ie one retracker).
    Parameters without variants have a single variant with id ''."""

    id: str
    name: str
    variable: str


@dataclass(frozen=True)
class ParameterConfig:  # pylint: disable=too-many-instance-attributes
    """A monitored parameter"""

    id: str
    long_name: str
    description: str
    source: str
    type: str  # 'flag' or 'float'
    variants: list[VariantDef]
    variant_label: str
    modes: list[str]  # empty when there is no mode selection
    areas: list[str]
    flags: list[FlagDef] = field(default_factory=list)
    units: str = ""
    plot_range: tuple[float, float] | None = None
    cmap: str = "RdYlBu_r"
    dtype: str = "float32"

    @property
    def has_variants(self) -> bool:
        """True if the parameter has more than one variable variant (ie retrackers)"""
        return self.variants[0].id != ""

    @property
    def mode_options(self) -> list[str]:
        """mode selections to process: [''] when the parameter has no mode selection"""
        return self.modes if self.modes else [""]

    @property
    def variables(self) -> list[str]:
        """netCDF variables read for this parameter"""
        return [v.variable for v in self.variants]


@dataclass(frozen=True)
class CsqaConfig:  # pylint: disable=too-many-instance-attributes
    """CSQA global configuration"""

    config_file: str
    mission_start_date: datetime
    cycle_length_days: int
    baselines: list[str]
    products: dict[str, ProductConfig]
    stage_preference: list[str]
    default_lat: str
    default_lon: str
    output_dir: str
    areas: dict[str, AreaConfig]
    mode_variable: str
    mode_values: dict[str, int]
    mode_labels: dict[str, str]
    image_format: str
    dpi: int
    webp_quality: int
    max_points: int
    parameters: dict[str, ParameterConfig]


def sanitize_key(name: str) -> str:
    """Convert a display name to a lower case key containing only [a-z0-9_]

    Args:
        name (str): display name, ie 'Lake/Enclosed Sea'

    Returns:
        str: key, ie 'lake_enclosed_sea'
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _expand_path(path: str) -> str:
    """expand environment variables and ~ in a path"""
    return os.path.expanduser(os.path.expandvars(str(path)))


def _require(cfg: dict, key: str, context: str):
    """return cfg[key] or raise a ValueError naming the missing key"""
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"missing '{key}' in {context}")
    return cfg[key]


def _parse_parameter(pid: str, pcfg: dict, cfg_areas: dict, mode_labels: dict, products: dict):
    """parse and validate one parameter definition

    Args:
        pid (str): parameter id
        pcfg (dict): parameter definition from the yaml file
        cfg_areas (dict): configured areas
        mode_labels (dict): configured mode labels
        products (dict): configured products

    Returns:
        ParameterConfig
    """
    context = f"parameter '{pid}'"
    if not re.fullmatch(r"[a-z0-9_]+", pid):
        raise ValueError(f"{context}: id must only contain [a-z0-9_]")

    ptype = _require(pcfg, "type", context)
    if ptype not in PARAM_TYPES:
        raise ValueError(f"{context}: type must be one of {PARAM_TYPES}, not {ptype}")

    source = _require(pcfg, "source", context)
    if source not in products:
        raise ValueError(f"{context}: source {source} not in configured products")

    if "variants" in pcfg:
        vcfg = pcfg["variants"]
        variants = [
            VariantDef(
                id=str(_require(v, "id", context)),
                name=str(v.get("name", v["id"])),
                variable=str(_require(v, "variable", context)),
            )
            for v in _require(vcfg, "options", context)
        ]
        for variant in variants:
            if not re.fullmatch(r"[a-z0-9]+", variant.id):
                raise ValueError(f"{context}: variant id {variant.id} must only contain [a-z0-9]")
        variant_label = str(vcfg.get("label", "Variant"))
    else:
        variants = [
            VariantDef(
                id="",
                name=str(pcfg.get("long_name", pid)),
                variable=str(_require(pcfg, "variable", context)),
            )
        ]
        variant_label = ""

    modes = [str(m) for m in pcfg.get("modes", [])]
    for mode in modes:
        if mode not in mode_labels:
            raise ValueError(f"{context}: mode {mode} not in configured mode labels")

    areas = [str(a) for a in pcfg.get("areas", list(cfg_areas))]
    for area in areas:
        if area not in cfg_areas:
            raise ValueError(f"{context}: area {area} not in configured areas")

    flags = []
    if ptype == "flag":
        for flag in _require(pcfg, "flags", context):
            flags.append(
                FlagDef(
                    value=int(_require(flag, "value", context)),
                    name=str(_require(flag, "name", context)),
                    color=flag.get("color"),
                    key=sanitize_key(str(flag["name"])),
                )
            )
        if len({f.key for f in flags}) != len(flags):
            raise ValueError(f"{context}: flag names must be unique")

    plot_cfg = pcfg.get("plot", {}) or {}
    plot_range = None
    if plot_cfg.get("range") is not None:
        plot_range = (float(plot_cfg["range"][0]), float(plot_cfg["range"][1]))

    return ParameterConfig(
        id=pid,
        long_name=str(pcfg.get("long_name", pid)),
        description=" ".join(str(pcfg.get("description", "")).split()),
        source=source,
        type=ptype,
        variants=variants,
        variant_label=variant_label,
        modes=modes,
        areas=areas,
        flags=flags,
        units=str(pcfg.get("units", "")),
        plot_range=plot_range,
        cmap=str(plot_cfg.get("cmap", "RdYlBu_r")),
        dtype=str(pcfg.get("dtype", "float32")),
    )


def load_config(config_file: str | None = None) -> CsqaConfig:
    """Load the CSQA global config and parameter definitions

    Args:
        config_file (str|None): path of csqa_config.yaml. If None use $CSQA_CONFIG or the
                                package default

    Raises:
        ValueError: if the configuration is invalid

    Returns:
        CsqaConfig
    """
    if config_file is None:
        config_file = os.environ.get("CSQA_CONFIG", DEFAULT_CONFIG_FILE)
    config_file = os.path.abspath(_expand_path(config_file))

    with open(config_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    context = config_file
    cycles_cfg = _require(cfg, "cycles", context)
    mission_start = datetime.strptime(
        str(_require(cycles_cfg, "mission_start_date", context)), "%Y-%m-%d"
    )

    products = {}
    for prod_id, prod in _require(cfg, "products", context).items():
        products[prod_id] = ProductConfig(
            id=prod_id,
            long_name=str(prod.get("long_name", prod_id)),
            dirs=[_expand_path(d) for d in _require(prod, "dirs", f"product {prod_id}")],
            file_glob=str(prod.get("file_glob", "*.nc")),
        )

    areas = {}
    for area_id, area in _require(cfg, "areas", context).items():
        areas[area_id] = AreaConfig(
            id=area_id,
            long_name=str(area.get("long_name", area_id)),
            polarplot_area=str(_require(area, "polarplot_area", f"area {area_id}")),
            lat_min=float(area.get("lat_min", -90.0)),
            lat_max=float(area.get("lat_max", 90.0)),
        )

    modes_cfg = _require(cfg, "modes", context)
    mode_values = {str(k): int(v) for k, v in _require(modes_cfg, "values", "modes").items()}
    mode_labels = {str(k): str(v) for k, v in _require(modes_cfg, "labels", "modes").items()}
    for mode in mode_labels:
        if mode != "all" and mode not in mode_values:
            raise ValueError(f"mode {mode} has a label but no value in modes:values")

    params_file = _expand_path(_require(cfg, "parameters_file", context))
    if not os.path.isabs(params_file):
        params_file = os.path.join(os.path.dirname(config_file), params_file)
    with open(params_file, encoding="utf-8") as fh:
        params_cfg = yaml.safe_load(fh)

    parameters = {}
    for pid, pcfg in _require(params_cfg, "parameters", params_file).items():
        parameters[pid] = _parse_parameter(pid, pcfg, areas, mode_labels, products)

    plots_cfg = cfg.get("plots", {}) or {}
    default_coords = cfg.get("default_coordinates", {}) or {}

    return CsqaConfig(
        config_file=config_file,
        mission_start_date=mission_start,
        cycle_length_days=int(cycles_cfg.get("cycle_length_days", 30)),
        baselines=[str(b).upper() for b in cfg.get("baselines", [])],
        products=products,
        stage_preference=[str(s) for s in cfg.get("stage_preference", [])],
        default_lat=str(default_coords.get("lat", "lat_poca_20_ku")),
        default_lon=str(default_coords.get("lon", "lon_poca_20_ku")),
        output_dir=_expand_path(_require(cfg, "output_dir", context)),
        areas=areas,
        mode_variable=str(_require(modes_cfg, "variable", "modes")),
        mode_values=mode_values,
        mode_labels=mode_labels,
        image_format=str(plots_cfg.get("image_format", "webp")),
        dpi=int(plots_cfg.get("dpi", 85)),
        webp_quality=int(plots_cfg.get("webp_quality", 80)),
        max_points=int(plots_cfg.get("max_points", 2_000_000)),
        parameters=parameters,
    )
