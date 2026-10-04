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

from cpom.altimetry.projects.csqa.cycles import CycleCalendar
from cpom.altimetry.projects.csqa.derived import DERIVED_VARIABLES
from cpom.altimetry.projects.csqa.gridding import GRID_STATISTICS

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
    # cpom.gridding.gridareas.GridArea of the area's gridded maps ('' if the area has none)
    grid_area: str = ""


@dataclass(frozen=True)
class ProductConfig:
    """An input product archive"""

    id: str
    long_name: str
    dirs: list[str]
    file_glob: str


@dataclass(frozen=True)
class ModeSurface:
    """A selection of an acquisition mode over surface types (ie LRM over ice), used like a
    mode in a parameter's modes"""

    id: str  # ie 'lrm_ice'
    mode: str  # acquisition mode id, ie 'lrm'
    surfaces: tuple[str, ...]  # surface type ids, ie ('ice',)
    label: str  # ie 'LRM Ice'


@dataclass(frozen=True)
class PassSelection:
    """A selection of the ascending or descending passes (records with increasing or decreasing
    latitude), used like a mode in a parameter's modes"""

    id: str  # ie 'asc'
    ascending: bool
    label: str  # ie 'Ascending passes'


@dataclass(frozen=True)
class FlagDef:
    """A flag value of a flag parameter"""

    value: int
    name: str
    color: str | None
    key: str  # name sanitized for use in file/column names, ie "lake_enclosed_sea"


@dataclass(frozen=True)
class VariantDef:  # pylint: disable=too-many-instance-attributes
    """A variable providing one variant of a parameter (ie one retracker).
    Parameters without variants have a single variant with id ''."""

    id: str
    name: str
    variable: str
    # what the variant is in each acquisition mode (ie the retracker used), None where it is
    # not used in a mode. Empty if not configured
    mode_descriptions: dict[str, str | None] = field(default_factory=dict)
    # for a bit of a bit flag word: the bit's mask. Its values are 1 (set) or 0 (not set)
    bit_mask: int | None = None
    # for a bit: the bit's name in the variable's flag_meanings attribute
    bit_name: str = ""
    # colour scale range and colormap of this variant's maps, overriding the parameter's
    # (for variants with very different value ranges, ie geophysical corrections)
    plot_range: tuple[float, float] | None = None
    cmap: str | None = None
    # values are rejected (NaN) where this bit of a flag word variable is set, ie
    # freeboard_error of flag_prod_status_20_ku. reject_name: the bit's name in flag_meanings
    reject_variable: str = ""
    reject_mask: int | None = None
    reject_name: str = ""
    # a variant derived from several variables (ie the mispointing angle from the roll and
    # pitch angles): the name of a derived.DERIVED_VARIABLES function and its input variables.
    # The variant's variable is then its first input (giving its dimension and coordinates)
    derived: str = ""
    inputs: tuple[str, ...] = ()

    @property
    def display_variable(self) -> str:
        """the variable, or the derived variable's name"""
        return self.derived if self.derived else self.variable


@dataclass(frozen=True)
class ColourScale:
    """A colour scale of a float parameter's maps. Each colour scale has its own set of maps,
    of the same data. The first (default) scale's maps have no file name suffix, the others
    have the suffix _<id>"""

    id: str
    name: str
    range: tuple[float, float] | None
    cmap: str
    file_suffix: str
    log: bool = False  # logarithmic colour scale


@dataclass(frozen=True)
class GridStatistic:
    """A statistic of the measurements in each grid cell, mapped and summarised"""

    id: str  # a key of gridding.GRID_STATISTICS, ie 'median'
    name: str
    units: str
    range: tuple[float, float] | None  # colour scale range (None: range of the values)
    cmap: str
    log: bool = False  # logarithmic colour scale


@dataclass(frozen=True)
class GridConfig:
    """Gridded maps and statistics of a float parameter: the parameter's measurements in each
    area are gridded into cells of binsize_km, and statistics of the measurements in each cell
    are mapped. The gridded statistics of an area are statistics of its cell values"""

    binsize_km: float
    areas: list[str]
    modes: list[str]  # mode selections gridded ([''] when the parameter has no modes)
    statistics: list[GridStatistic]  # the first is the default
    min_count: int = 1  # minimum number of measurements in a cell

    @property
    def label(self) -> str:
        """display name, ie '10 km grid'"""
        return f"{self.binsize_km:g} km grid"

    def file_suffix(self, stat_id: str) -> str:
        """plot file name suffix of a statistic's maps, ie 'grid10km_median'"""
        return f"grid{self.binsize_km:g}km_{stat_id}".replace(".", "p")


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
    plot_range: tuple[float, float] | None = None  # of the default colour scale
    cmap: str = "RdYlBu_r"  # of the default colour scale
    dtype: str = "float32"
    colour_scales: list[ColourScale] = field(default_factory=list)  # default scale first
    map_modes: list[str] = field(default_factory=list)  # modes with maps (default all modes)
    default_variant: str = ""  # variant shown first in the portal (default the first)
    grid: GridConfig | None = None  # gridded maps and statistics (None: not gridded)
    # acquisition modes with valid values: values of other modes are rejected (ie LRM values
    # of parameters only computed in SAR and SARin modes). Empty: every mode
    valid_modes: list[str] = field(default_factory=list)
    # first product baseline containing the parameter (ie 'F'): earlier baselines are not
    # processed for it. Empty: every baseline
    first_baseline: str = ""
    # factor applied to the product values, ie 1000 for degrees -> millidegrees
    value_scale: float = 1.0

    def in_baseline(self, baseline: str) -> bool:
        """True if the parameter is processed for a product baseline"""
        return not self.first_baseline or baseline >= self.first_baseline

    @property
    def is_bit_flag(self) -> bool:
        """True if the parameter's variants are the bits of a flag word"""
        return self.variants[0].bit_mask is not None

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
        """netCDF variables of the parameter's variants (the first input of derived variants)"""
        return [v.variable for v in self.variants]


@dataclass(frozen=True)
class CsqaConfig:  # pylint: disable=too-many-instance-attributes
    """CSQA global configuration"""

    config_file: str
    mission_start_date: datetime
    cycle_length_days: int
    data_latency_days: float
    baselines: list[str]
    products: dict[str, ProductConfig]
    stage_preference: list[str]
    default_lat: str
    default_lon: str
    output_dir: str
    areas: dict[str, AreaConfig]
    mode_variable: str
    mode_values: dict[str, int]
    mode_labels: dict[str, str]  # of the modes, 'all' and the mode surface selections
    image_format: str
    dpi: int
    webp_quality: int
    max_points: int
    parameters: dict[str, ParameterConfig]
    surface_variable: str = ""  # surface type mask variable (for mode surface selections)
    surface_values: dict[str, int] = field(default_factory=dict)
    mode_surfaces: dict[str, ModeSurface] = field(default_factory=dict)
    pass_selections: dict[str, PassSelection] = field(default_factory=dict)

    def mode_text(self, param: ParameterConfig, mode: str) -> str:
        """a parameter's mode selection as text, ie 'All modes', 'SAR mode', 'LRM Ice',
        'Ascending passes', or 'All passes' for a parameter selecting passes ('' if none)"""
        if mode == "":
            return ""
        if mode == "all":
            others = [m for m in param.modes if m != "all"]
            if others and all(m in self.pass_selections for m in others):
                return "All passes"
            return self.mode_labels["all"]
        label = self.mode_labels.get(mode, mode)
        if mode in self.mode_surfaces or mode in self.pass_selections:
            return label
        return f"{label} mode"

    def calendar(self) -> CycleCalendar:
        """the cycle calendar of this configuration"""
        return CycleCalendar(
            self.mission_start_date, self.cycle_length_days, self.data_latency_days
        )


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


def _variant_plot_range(variant_cfg: dict) -> tuple[float, float] | None:
    """colour scale range of a variant's maps (plot: range: [min, max]), or None"""
    plot_range = (variant_cfg.get("plot") or {}).get("range")
    return (float(plot_range[0]), float(plot_range[1])) if plot_range else None


def _reject_bit(variant_cfg: dict, context: str) -> dict:
    """VariantDef fields of a variant's reject_bit: {variable, mask, name}, if configured"""
    reject = variant_cfg.get("reject_bit")
    if not reject:
        return {}
    mask = int(_require(reject, "mask", context))
    if mask <= 0 or mask & (mask - 1):
        raise ValueError(f"{context}: reject_bit mask {mask} is not a single bit")
    return {
        "reject_variable": str(_require(reject, "variable", context)),
        "reject_mask": mask,
        "reject_name": str(reject.get("name", "")),
    }


def _derived(variant_cfg: dict, context: str) -> dict:
    """VariantDef fields of a derived variant (derived and inputs), if configured"""
    derived = variant_cfg.get("derived")
    if not derived:
        return {}
    if derived not in DERIVED_VARIABLES:
        raise ValueError(f"{context}: unknown derived variable {derived}")
    inputs = tuple(str(x) for x in _require(variant_cfg, "inputs", context))
    if len(inputs) != DERIVED_VARIABLES[derived][0]:
        raise ValueError(
            f"{context}: {derived} needs {DERIVED_VARIABLES[derived][0]} input variables"
        )
    return {"derived": str(derived), "inputs": inputs}


def _variant_variable(variant_cfg: dict, context: str) -> str:
    """a variant's variable: its first input if it is derived"""
    if variant_cfg.get("derived"):
        return str((_require(variant_cfg, "inputs", context) or [""])[0])
    return str(_require(variant_cfg, "variable", context))


def _parse_grid(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    pcfg: dict,
    context: str,
    cfg_areas: dict,
    modes: list[str],
    units: str,
    default_scale: ColourScale,
) -> GridConfig | None:
    """parse and validate a parameter's grid definition (None if it has none)"""
    gcfg = pcfg.get("grid")
    if not gcfg:
        return None
    context = f"{context} grid"
    binsize_km = float(_require(gcfg, "binsize_km", context))
    if binsize_km <= 0:
        raise ValueError(f"{context}: binsize_km must be > 0")
    areas = [str(a) for a in _require(gcfg, "areas", context)]
    for area in areas:
        if area not in cfg_areas or not cfg_areas[area].grid_area:
            raise ValueError(f"{context}: area {area} is not configured with a grid_area")
    grid_modes = [str(m) for m in gcfg.get("modes", modes[:1] or [""])]
    for mode in grid_modes:
        if mode not in (modes or [""]):
            raise ValueError(f"{context}: mode {mode} is not one of the parameter's modes")
    statistics = []
    for stat in _require(gcfg, "statistics", context):
        stat_id = str(_require(stat, "id", context))
        if stat_id not in GRID_STATISTICS:
            raise ValueError(f"{context}: unknown statistic {stat_id}")
        # values of the parameter by default use the parameter's default colour scale
        is_value = stat_id not in ("count", "std")
        stat_range = stat.get("range", default_scale.range if is_value else None)
        statistics.append(
            GridStatistic(
                id=stat_id,
                name=str(stat.get("name", GRID_STATISTICS[stat_id])),
                units="" if stat_id == "count" else units,
                range=(float(stat_range[0]), float(stat_range[1])) if stat_range else None,
                cmap=str(stat.get("cmap", default_scale.cmap)),
                log=bool(stat.get("log", default_scale.log if is_value else False)),
            )
        )
        if statistics[-1].log and not (stat_range and float(stat_range[0]) > 0):
            raise ValueError(f"{context}: log colour scale of {stat_id} needs a range above 0")
    if not statistics or len({s.id for s in statistics}) != len(statistics):
        raise ValueError(f"{context}: statistics must be unique and not empty")
    return GridConfig(
        binsize_km=binsize_km,
        areas=areas,
        modes=grid_modes,
        statistics=statistics,
        min_count=int(gcfg.get("min_count", 1)),
    )


def _parse_parameter(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    pid: str,
    pcfg: dict,
    cfg_areas: dict,
    mode_labels: dict,
    products: dict,
    acquisition_modes: set[str],
):
    """parse and validate one parameter definition

    Args:
        pid (str): parameter id
        pcfg (dict): parameter definition from the yaml file
        cfg_areas (dict): configured areas
        mode_labels (dict): configured mode selection labels ('all', modes and mode surfaces)
        products (dict): configured products
        acquisition_modes (set[str]): configured acquisition mode ids (ie lrm, sar, sarin)

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

    flags_cfg = pcfg.get("flags")
    if "bits" in pcfg:
        # a bit flag word: each bit is a variant with values 0 (not set) and 1 (set)
        if ptype != "flag":
            raise ValueError(f"{context}: bits need type flag")
        bcfg = pcfg["bits"]
        variable = str(_require(pcfg, "variable", context))
        variants = []
        for bit in _require(bcfg, "options", context):
            mask = int(_require(bit, "mask", context))
            if mask <= 0 or mask & (mask - 1):
                raise ValueError(f"{context}: bit mask {mask} is not a single bit")
            variants.append(
                VariantDef(
                    id=f"b{mask.bit_length() - 1}",
                    name=str(bit.get("label", bit.get("name", mask))),
                    variable=variable,
                    bit_mask=mask,
                    bit_name=str(bit.get("name", "")),
                )
            )
        if len({v.id for v in variants}) != len(variants):
            raise ValueError(f"{context}: bit masks must be unique")
        variant_label = str(bcfg.get("label", "Flag bit"))
        flags_cfg = flags_cfg or [
            {"value": 0, "name": "Not set", "color": "#b8bfc8"},
            {"value": 1, "name": "Set", "color": "#d55e00"},
        ]
    elif "variants" in pcfg:
        vcfg = pcfg["variants"]
        variants = [
            VariantDef(
                id=str(_require(v, "id", context)),
                name=str(v.get("name", v["id"])),
                variable=_variant_variable(v, context),
                mode_descriptions={
                    str(mode): (None if desc is None else str(desc))
                    for mode, desc in (v.get("mode_descriptions") or {}).items()
                },
                plot_range=_variant_plot_range(v),
                cmap=(v.get("plot") or {}).get("cmap"),
                **_reject_bit(v, context),
                **_derived(v, context),
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
                **_reject_bit(pcfg, context),
            )
        ]
        variant_label = ""

    modes = [str(m) for m in pcfg.get("modes", [])]
    for mode in modes:
        if mode not in mode_labels:
            raise ValueError(f"{context}: mode {mode} not in configured mode labels")
    for variant in variants:
        for mode in variant.mode_descriptions:
            if mode not in acquisition_modes:
                raise ValueError(
                    f"{context}: variant {variant.id} mode_descriptions has unknown mode {mode}"
                )

    areas = [str(a) for a in pcfg.get("areas", list(cfg_areas))]
    for area in areas:
        if area not in cfg_areas:
            raise ValueError(f"{context}: area {area} not in configured areas")

    flags = []
    if ptype == "flag":
        if flags_cfg is None:
            raise ValueError(f"missing 'flags' in {context}")
        for flag in flags_cfg:
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
    cmap = str(plot_cfg.get("cmap", "RdYlBu_r"))
    plot_range = None
    if plot_cfg.get("range") is not None:
        plot_range = (float(plot_cfg["range"][0]), float(plot_cfg["range"][1]))

    # alternative colour scales, each producing its own maps (the first is the default)
    colour_scales = []
    for i, scale in enumerate(plot_cfg.get("scales") or []):
        scale_id = str(_require(scale, "id", context))
        if not re.fullmatch(r"[a-z0-9]+", scale_id):
            raise ValueError(f"{context}: colour scale id {scale_id} must only contain [a-z0-9]")
        scale_range = scale.get("range")
        colour_scales.append(
            ColourScale(
                id=scale_id,
                name=str(scale.get("name", scale_id)),
                range=(float(scale_range[0]), float(scale_range[1])) if scale_range else None,
                cmap=str(scale.get("cmap", cmap)),
                file_suffix="" if i == 0 else scale_id,
                log=bool(scale.get("log", False)),
            )
        )
        if colour_scales[-1].log and not (scale_range and float(scale_range[0]) > 0):
            raise ValueError(f"{context}: log colour scale {scale_id} needs a range above 0")
    if len({s.id for s in colour_scales}) != len(colour_scales):
        raise ValueError(f"{context}: colour scale ids must be unique")
    if colour_scales:
        plot_range, cmap = colour_scales[0].range, colour_scales[0].cmap
    else:
        colour_scales = [ColourScale("", "", plot_range, cmap, "")]

    # modes with maps: all mode selections by default ([''] when there is no mode selection)
    map_modes = [str(m) for m in pcfg.get("map_modes", modes or [""])]
    for mode in map_modes:
        if mode not in (modes or [""]):
            raise ValueError(f"{context}: map mode {mode} is not one of the parameter's modes")
    valid_modes = [str(m) for m in pcfg.get("valid_modes", [])]
    for mode in valid_modes:
        if mode not in acquisition_modes:
            raise ValueError(f"{context}: valid mode {mode} is not an acquisition mode")
    first_baseline = str(pcfg.get("first_baseline", "")).upper()
    if first_baseline and not re.fullmatch(r"[A-Z]", first_baseline):
        raise ValueError(f"{context}: first_baseline must be a baseline letter")
    value_scale = float(pcfg.get("value_scale", 1.0))
    if value_scale == 0:
        raise ValueError(f"{context}: value_scale must not be 0")
    units = str(pcfg.get("units", ""))
    grid = _parse_grid(pcfg, context, cfg_areas, modes, units, colour_scales[0])
    if grid is not None and (ptype != "float" or variants[0].bit_mask is not None):
        raise ValueError(f"{context}: only float parameters can be gridded")
    default_variant = str(pcfg.get("default_variant", variants[0].id))
    if default_variant not in [v.id for v in variants]:
        raise ValueError(f"{context}: default_variant {default_variant} is not a variant id")

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
        units=units,
        plot_range=plot_range,
        cmap=cmap,
        dtype=str(pcfg.get("dtype", "float32")),
        colour_scales=colour_scales,
        map_modes=map_modes,
        default_variant=default_variant,
        grid=grid,
        valid_modes=valid_modes,
        first_baseline=first_baseline,
        value_scale=value_scale,
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
            grid_area=str(area.get("grid_area", "")),
        )

    modes_cfg = _require(cfg, "modes", context)
    mode_values = {str(k): int(v) for k, v in _require(modes_cfg, "values", "modes").items()}
    mode_labels = {str(k): str(v) for k, v in _require(modes_cfg, "labels", "modes").items()}
    for mode in mode_labels:
        if mode != "all" and mode not in mode_values:
            raise ValueError(f"mode {mode} has a label but no value in modes:values")

    # selections of a mode over surface types (ie LRM over ice)
    surfaces_cfg = cfg.get("surface_types") or {}
    surface_values = {str(k): int(v) for k, v in (surfaces_cfg.get("values") or {}).items()}
    mode_surfaces = {}
    for sel_id, sel in (cfg.get("mode_surfaces") or {}).items():
        sel_context = f"mode_surfaces {sel_id}"
        mode = str(_require(sel, "mode", sel_context))
        surfaces = sel.get("surfaces", sel.get("surface"))
        surfaces = tuple(str(x) for x in ([surfaces] if isinstance(surfaces, str) else surfaces))
        if not re.fullmatch(r"[a-z0-9_]+", str(sel_id)) or sel_id in mode_labels:
            raise ValueError(f"{sel_context}: id must be [a-z0-9_] and not a mode id")
        if mode not in mode_values:
            raise ValueError(f"{sel_context}: unknown mode {mode}")
        if not surfaces or any(x not in surface_values for x in surfaces):
            raise ValueError(f"{sel_context}: surfaces must be configured surface_types")
        mode_surfaces[str(sel_id)] = ModeSurface(
            str(sel_id), mode, surfaces, str(sel.get("label", sel_id))
        )
    if mode_surfaces and not surfaces_cfg.get("variable"):
        raise ValueError("mode_surfaces need surface_types:variable")

    # selections of the ascending or descending passes
    pass_selections = {}
    for sel_id, sel in (cfg.get("pass_selections") or {}).items():
        sel_context = f"pass_selections {sel_id}"
        direction = str(_require(sel, "direction", sel_context))
        if direction not in ("ascending", "descending"):
            raise ValueError(f"{sel_context}: direction must be ascending or descending")
        if (
            not re.fullmatch(r"[a-z0-9_]+", str(sel_id))
            or sel_id in mode_labels
            or sel_id in mode_surfaces
        ):
            raise ValueError(f"{sel_context}: id must be [a-z0-9_] and not a mode id")
        pass_selections[str(sel_id)] = PassSelection(
            str(sel_id), direction == "ascending", str(sel.get("label", sel_id))
        )
    all_labels = {
        **mode_labels,
        **{m.id: m.label for m in mode_surfaces.values()},
        **{p.id: p.label for p in pass_selections.values()},
    }

    params_file = _expand_path(_require(cfg, "parameters_file", context))
    if not os.path.isabs(params_file):
        params_file = os.path.join(os.path.dirname(config_file), params_file)
    with open(params_file, encoding="utf-8") as fh:
        params_cfg = yaml.safe_load(fh)

    parameters = {}
    for pid, pcfg in _require(params_cfg, "parameters", params_file).items():
        parameters[pid] = _parse_parameter(pid, pcfg, areas, all_labels, products, set(mode_values))
    for param in parameters.values():
        # gridded statistics timeseries are saved as <param>_grid.csv
        if param.grid is not None and f"{param.id}_grid" in parameters:
            raise ValueError(f"parameter id {param.id}_grid clashes with {param.id}'s grid")

    plots_cfg = cfg.get("plots", {}) or {}
    default_coords = cfg.get("default_coordinates", {}) or {}

    return CsqaConfig(
        config_file=config_file,
        mission_start_date=mission_start,
        cycle_length_days=int(cycles_cfg.get("cycle_length_days", 30)),
        data_latency_days=float(cycles_cfg.get("data_latency_days", 0)),
        baselines=[str(b).upper() for b in cfg.get("baselines", [])],
        products=products,
        stage_preference=[str(s) for s in cfg.get("stage_preference", [])],
        default_lat=str(default_coords.get("lat", "lat_poca_20_ku")),
        default_lon=str(default_coords.get("lon", "lon_poca_20_ku")),
        output_dir=_expand_path(_require(cfg, "output_dir", context)),
        areas=areas,
        mode_variable=str(_require(modes_cfg, "variable", "modes")),
        mode_values=mode_values,
        mode_labels=all_labels,
        image_format=str(plots_cfg.get("image_format", "webp")),
        dpi=int(plots_cfg.get("dpi", 85)),
        webp_quality=int(plots_cfg.get("webp_quality", 80)),
        max_points=int(plots_cfg.get("max_points", 2_000_000)),
        parameters=parameters,
        surface_variable=str(surfaces_cfg.get("variable", "")),
        surface_values=surface_values,
        mode_surfaces=mode_surfaces,
        pass_selections=pass_selections,
    )
