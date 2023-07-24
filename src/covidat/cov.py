import argparse
import locale
import logging
import re
import threading
import typing
from collections import Counter
from contextlib import contextmanager, nullcontext
from cProfile import label
from datetime import date, timedelta
from itertools import count
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.axis
import matplotlib.colors
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from IPython.display import display

from .covdata import *
from .util import COLLECTROOT

logger = logging.getLogger(__name__)

un_cmap = "cet_rainbow4"
un_cmap_r = "cet_rainbow4_r"
un_l_cmap = "YlOrRd"
un_l_cmap_r = "YlOrRd_r"
div_cmap = "icefire"
div_l_cmap = "RdYlBu_r"


class NoDataError(KeyError):
    pass


INZ_TICKS = [
    0.5,
    1,
    5,
    10,
    20,
    35,
    50,
    100,
    200,
    300,
    400,
    700,
    1000,
    1500,
    2000,
    3000,
    4000,
]

LOCALE_LOCK = threading.Lock()

MARKER_SEQ = (
    "o",
    "X",
    (4, 0, 45),
    "P",
    (4, 0, 0),
    (4, 1, 0),
    "^",
    (4, 1, 45),
    "v",
)

# with plt.rc_context({'axes.prop_cycle': cycler(color=sns.color_palette("rainbow", n_colors=10))}):


def with_palette(palette, n_colors=10):
    return plt.rc_context({"axes.prop_cycle": cycler(color=sns.color_palette(palette, n_colors=n_colors))})


def with_age_palette():
    return with_palette("Paired", n_colors=10)


def labelend2(
    ax,
    mms,
    ycol,
    ymax=None,
    cats="Bundesland",
    x="Datum",
    shorten=lambda c: SHORTNAME_BY_BUNDESLAND[c],
    colorize=None,
    ends=(False, True),
    **kwargs,
):
    annotargs = {"va": "center"} | kwargs
    for i, bl in enumerate(mms[cats].unique()):
        # print(bl)
        ys = mms[mms[cats] == bl][ycol]
        lastidx = ys.last_valid_index()
        if lastidx is not None:
            if colorize:
                annotargs["color"] = colorize[i]
            txt = shorten(bl)
            # print(bl, cov.SHORTNAME_BY_BUNDESLAND[bl], (mms["Datum"].iloc[-1], ys.iloc[-1]))
            if ends[1]:
                ax.annotate(
                    txt,
                    (mms[x].loc[lastidx], ys.loc[lastidx]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    **annotargs,
                )
            # print(bl, cov.SHORTNAME_BY_BUNDESLAND[bl], (mms["Datum"].iloc[-1], ys.iloc[-1]))
            firstidx = ys.first_valid_index()
            if ends[0] and (firstidx != lastidx or not ends[1]):
                ax.annotate(
                    txt,
                    (mms[x].loc[firstidx], ys.loc[firstidx]),
                    xytext=(-5, 0),
                    textcoords="offset points",
                    ha="right",
                    **annotargs,
                )


def _sortedlabels(ax, mms0, by, cat="Bundesland", x="Datum", fmtval=None):
    hls = list(zip(*ax.get_legend_handles_labels()))
    label_order = filterlatest(mms0, x).sort_values(by=by, ascending=False)[cat].unique()
    hls = [(handle, label) for handle, label in hls if label in label_order]
    hls.sort(key=lambda hl: np.where(label_order == hl[1]))
    if fmtval:
        for i, (art, bl) in enumerate(hls):
            val = mms0.loc[mms0[cat] == bl, by].iloc[-1]
            hls[i] = (art, f"{bl}: {fmtval(val)}")
    return zip(*hls)


def sortedlabels(ax, mms0, by, cat="Bundesland", x="Datum", fmtval=None):
    hls = list(zip(*ax.get_legend_handles_labels()))
    label_order = mms0.loc[~pd.isna(mms0[by])].groupby(cat).agg({by: "last"}).sort_values(by=by, ascending=False).index
    hls = [(handle, label) for handle, label in hls if label in label_order]
    hls.sort(key=lambda hl: np.where(label_order == hl[1]))
    if fmtval:
        for i, (art, bl) in enumerate(hls):
            # vals =  mms0.loc[mms0[cat] == bl, by]
            val = mms0.loc[mms0[cat] == bl, by].dropna().iloc[-1]
            hls[i] = (art, f"{bl}: {fmtval(val)}")
    return zip(*hls)


def sum_rows(df, cname, csval, agg="sum"):
    dfsum = df.groupby(level=[n for n in df.index.names if n != cname]).agg(agg)
    dfsum[cname] = csval
    dfsum.set_index(cname, append=True, inplace=True, verify_integrity=True)
    return dfsum.reorder_levels(df.index.names)


def add_sum_rows(df, cname, csval, agg="sum"):
    return pd.concat([df, sum_rows(df, cname, csval, agg)])


@contextmanager
def setlocale(name):
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)


ARCHIVE_ROOT = COLLECTROOT / "covid/coronaDAT/archive"
ARCHIVE_PATCH_ROOT = COLLECTROOT / "covid/ages_all"
DATE_FMT = "%Y%m%d"


def plt_mdiff1(ax, fs_oo, ages_old_oo, vcol, ndays, logview, rwidth, sharey=False, color=None):
    newseries = fs_oo[vcol].rolling(rwidth).sum()
    if rwidth == 1:
        avgseries = fs_oo[vcol].rolling(7).mean()
    if ndays is not None:
        newseries = newseries.iloc[-ndays:]
    else:
        ndays = len(newseries)
    oldseries = (fs_oo[vcol] * 0).add(ages_old_oo[vcol], fill_value=0).rolling(rwidth).sum().iloc[-ndays:]

    def subquery(s):
        # return s[(pd.to_datetime("2020-05-01") <= s.index) & (s.index < pd.to_datetime("2020-11-01"))]
        return s

    oldseries = subquery(oldseries)
    newseries = subquery(newseries)
    if sharey:
        oldseries = (oldseries / fs_oo["AnzEinwohner"]) * 100_000
        newseries = (newseries / fs_oo["AnzEinwohner"]) * 100_000
    # oldseries = oldseries.loc[oldseries.index >= newseries.index[0]]
    if not logview:
        diffseries = newseries.sub(oldseries, fill_value=0)
        # display(fs_oo.iloc[0]["Bundesland"], diffseries.tail(3), newseries.tail(3))
    multiday = (fs_oo.index[-1] - ages_old_oo.index[-1]).days >= 2
    if multiday:
        ax.axvspan(
            ages_old_oo.index[-1] + timedelta(0.5),
            fs_oo.index[-1] + timedelta(0.5),
            in_layout=False,
            zorder=3,
            color=(1, 1, 0),
            alpha=0.1,
            lw=0,
        )
        ax.axvline(
            ages_old_oo.index[-1] + timedelta(0.5),
            color="lightgrey",
            ls=":",
            lw=1,
            snap=True,
            aa=False,
        )
    newstamp = (
        ("Stand " + fs_oo.iloc[-1]["FileDate"].strftime("%a %x")) if "FileDate" in fs_oo.columns else "Letzter Stand"
    )
    oldstamp = "Stand " + ages_old_oo.iloc[-1]["FileDate"].strftime("%a %x")
    if logview:
        ax.plot(newseries, label=newstamp)
        ax.plot(oldseries, label=oldstamp)
    else:
        newlabel = "Neu bis " + newstamp
        minuslabel = "Nach unten korrigiert"
        if rwidth > 1:
            ax.stackplot(
                newseries.index,
                oldseries,
                diffseries.where(diffseries > 0).fillna(0),
                diffseries.where(diffseries < 0).fillna(0),
                colors=[color or "k", (1, 0, 0), "skyblue"],
                lw=0,
                labels=[oldstamp, newlabel, minuslabel],
            )
        else:
            ax.bar(oldseries.index, oldseries, color=color or "k", lw=0, label=oldstamp)
            newbs = ax.bar(
                newseries.index,
                newseries.sub(oldseries, fill_value=0).where(newseries > oldseries),
                bottom=oldseries,
                color=(1, 0, 0),
                lw=0,
                label=newlabel,
            )
            ax.bar(
                newseries.index,
                newseries.sub(oldseries, fill_value=0).where(newseries < oldseries),
                bottom=oldseries,
                color="skyblue",
                lw=0,
                label=minuslabel,
            )
            if not sharey:
                ax.set_ylim(top=max(oldseries.max(), newseries.max()) * 1.05)
            ax.plot(avgseries.iloc[-ndays - 1 :], ls="--", color="lightgrey")
            # lastval = newseries.iloc[-1]
            # ax.annotate(format(lastval, ".0f"), (newbs[-1].get_x() + newbs[-1].get_width(), lastval * 0.95), color="k", rotation=90,
            #    fontsize="xx-small", ha="right", va="top", rotation_mode="anchor")


def plt_mdiff(
    fs,
    ages_old,
    catcol,
    vcol="AnzahlTot",
    name="COVID-Tote",
    sharey=True,
    rwidth=14,
    ndays=60,
    logview=False,
    color=None,
):
    # sharey=True
    fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharex=True, sharey=sharey)
    # display(g.axes)
    # style=dict(lw=0.5, mew=0, marker=".", markersize=5)
    style = dict(lw=0.4)
    maxy = 0
    for k, ax in zip(fs[catcol].unique(), axs.flat):
        # print(k)
        fs_oo = fs[fs[catcol] == k].set_index("Datum")
        maxy_oo = fs_oo[vcol].iloc[-ndays:].rolling(rwidth).sum().max()

        if ax in axs.T[0]:
            ax.set_ylabel(name + "/100.000" if sharey else name)
        if ax in axs.T[0] or not sharey:
            # ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator())
            if logview:
                set_logscale(ax)
                ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[0.3, 1], numticks=5))
                # print("setscale", k)
            else:
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator("auto", integer=True, min_n_ticks=min(4, maxy_oo))
                )
            if not sharey:
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))

        if not sharey:
            ax.tick_params(pad=-5, axis="y")
            if maxy_oo >= 10_000:
                ax.tick_params(axis="y", labelsize="x-small" if maxy_oo >= 100_000 else "small")
        ages_old_oo = ages_old[ages_old[catcol] == k].set_index("Datum")
        diffsum = fs_oo[vcol + "Sum"].iloc[-1] - ages_old_oo[vcol + "Sum"].iloc[-1]
        ax.set_title(f"{k} ({diffsum:+n})", y=0.94)
        plt_mdiff1(ax, fs_oo, ages_old_oo, vcol, ndays, logview, rwidth, sharey, color=color)
        maxy = max(fs_oo[vcol].iloc[-ndays:].rolling(rwidth).sum().max(), maxy)
        if ax in axs[-1]:
            set_date_opts(
                ax,
                fs_oo.index if not ndays else fs_oo.index[-ndays:],
                showyear=ndays > 180,
            )
            if ndays > 350:
                ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
            # ax.set_xlim(left=pd.to_datetime("2021-05-01"), right=pd.to_datetime("2021-9-30"))
        # ax.set_ylim(top=2)
        # break

        if ax is axs.flat[0]:
            fig.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    print(maxy, sharey)
    fig.subplots_adjust(wspace=0.03 if sharey else (0.11 if maxy >= 10_000 else 0.25))
    if ndays >= 90:
        fig.autofmt_xdate()
    fig.suptitle(f"{name}, {rwidth}-Tage-Summen", y=0.95)
    return fig, axs


# https://github.com/mwaskom/seaborn/blob/091f4c0e4f3580a8060de5596fa64c1ff9454dc5/seaborn/matrix.py#L192-L247
def adjust_cmap(plot_data, cmap=None, vmin=None, vmax=None, center=None, robust=False):
    """Use some heuristics to set good defaults for colorbar and range."""

    calc_data = plot_data.astype(float)  # Mask support removed
    if vmin is None:
        if robust:
            vmin = np.nanpercentile(calc_data, 2)
        else:
            vmin = np.nanmin(calc_data)
    if vmax is None:
        if robust:
            vmax = np.nanpercentile(calc_data, 98)
        else:
            vmax = np.nanmax(calc_data)
    # print("range:", (vmin, vmax))
    # Choose default colormaps if not provided
    if cmap is None:
        if center is None:
            cmap = sns.color_palette(un_cmap, as_cmap=True)
        else:
            cmap = sns.color_palette(div_cmap, as_cmap=True)
    elif isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    elif isinstance(cmap, list):
        cmap = matplotlib.colors.ListedColormap(cmap)

    # Recenter a divergent colormap
    if center is not None:
        # Copy bad values
        # in mpl<3.2 only masked values are honored with "bad" color spec
        # (see https://github.com/matplotlib/matplotlib/pull/14257)
        bad = cmap(np.ma.masked_invalid([np.nan]))[0]

        # under/over values are set for sure when cmap extremes
        # do not map to the same color as +-inf
        under = cmap(-np.inf)
        over = cmap(np.inf)
        under_set = under != cmap(0)
        over_set = over != cmap(cmap.N - 1)

        normlize = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=center)
        # cmin, ccenter, cmax = normlize([vmin, center, vmax])
        # print("crange:", (cmin, center, cmax))
        ncolors = int(max(256, (vmax - center) / (center - vmin) * 256))
        # print("nc", ncolors)
        cc = normlize(np.linspace(vmin, vmax, ncolors))
        # print(cmap)
        cmap = mpl.colors.ListedColormap(cmap(cc))
        cmap.set_bad(bad)
        if under_set:
            cmap.set_under(under)
        if over_set:
            cmap.set_over(over)
    return cmap


Bundesland = pd.CategoricalDtype(
    categories=[
        # Sonderkategorie
        "Österreich",
        # Westösterreich (AT3)
        "Vorarlberg",
        "Tirol",
        "Salzburg",
        "Oberösterreich",
        # Ostösterreich (AT1)
        "Niederösterreich",
        "Wien",
        "Burgenland",
        # Südösterreich (AT2)
        "Steiermark",
        "Kärnten",
    ],
    ordered=False,
)

ICU_LIM_GREEN_BY_STATE = {
    "Österreich": 200,
    "Burgenland": 5,
    "Kärnten": 20,
    "Niederösterreich": 35,
    "Oberösterreich": 25,
    "Salzburg": 15,
    "Steiermark": 30,
    "Tirol": 20,
    "Vorarlberg": 5,
    "Wien": 50,
}


class StackBarPlotter:
    def __init__(self, ax: plt.Axes, xs, allow_negative=False, areamode=False, **kwargs):
        self.ax = ax
        self.xs = xs
        self.kwargs = kwargs
        self.sum_below = np.zeros(len(xs))
        self.areamode = areamode
        if allow_negative:
            self.sum_above = np.zeros(len(xs))
        else:
            self.sum_above = None

    def _doplot(self, xs, ys, bots, kwargs):
        if self.areamode:
            self.ax.fill_between(xs, bots, bots + ys, **self.kwargs, **kwargs)
        else:
            self.ax.bar(
                xs,
                ys,
                bottom=bots,
                **self.kwargs,
                **kwargs,
            )

    def __call__(self, ys, **kwargs):
        if self.sum_above is not None:
            nolabels = kwargs.copy()
            nolabels["label"] = "_nolegend"
            positive = ys >= 0
            if positive.any():
                self._doplot(
                    self.xs[positive],
                    ys[positive],
                    self.sum_below[positive],
                    kwargs,
                )
                self.sum_below[positive] += ys[positive]
            if not positive.all():
                self._doplot(
                    self.xs[~positive],
                    ys[~positive],
                    self.sum_above[~positive],
                    nolabels,
                )
            self.sum_above[~positive] += ys[~positive]
        else:
            self._doplot(self.xs, ys, self.sum_below, kwargs)
            self.sum_below += ys


def plt_age_sums_ax(  # TODO: Ugly!
    ax: plt.Axes,
    data: pd.DataFrame,
    n_days: int,
    col: str = "inz",
    mode: str = "direct",
    weightcol: str = None,
    color_args: dict = None,
):
    return plt_cat_sums_ax(
        ax,
        data.query("Altersgruppe != 'Alle'"),
        data.query("Altersgruppe == 'Alle'"),
        catcol="Altersgruppe",
        n_days=n_days,
        col=col,
        mode=mode,
        weightcol=weightcol,
        color_args=color_args,
    )


def plt_cat_sums_ax(  # TODO: Ugly!
    ax: plt.Axes,
    data_cat: pd.DataFrame,
    data_agg: pd.DataFrame,
    *,
    catcol: str,
    n_days: int,
    col: str = "inz",
    mode: str = "direct",
    weightcol: str = None,
    color_args: dict = None,
):
    # n_days = (agd_sums_at.iloc[-1]["Datum"] - agd_sums_at.iloc[0]["Datum"]).days
    # n_days = 500
    # print(n_days)
    if n_days == 0:
        n_days = (data_agg.iloc[-1]["Datum"] - data_agg.iloc[0]["Datum"]).days + 1
    # print(repr(n_days))
    cutoff = data_agg["Datum"].iloc[-1] - np.timedelta64(n_days, "D")
    data_cat = data_cat[data_cat["Datum"] > cutoff]
    data_agg = data_agg[data_agg["Datum"] > cutoff].set_index("Datum").sort_index()
    labels = data_cat[catcol].unique()
    if weightcol:
        weights = data_agg[weightcol]
    colors = sns.color_palette(n_colors=len(labels), **(color_args or {}))
    dates = matplotlib.dates.date2num(data_agg.index.get_level_values("Datum"))
    if mode == "ratio":
        sums = data_cat.groupby("Datum")[col].sum()
    # print(agd_ids)
    if n_days < 100:
        pltbar = StackBarPlotter(ax, dates, width=0.6, linewidth=0)
    else:
        pltbar = StackBarPlotter(ax, dates, areamode=True)
    for label, color in zip(labels, colors):
        df = data_cat[data_cat[catcol] == label].set_index("Datum").sort_index()
        hs = df[col]
        if weightcol:
            hs = (hs * df[weightcol] / weights).replace(np.nan, 0)
        if mode == "direct":
            pass
        elif mode == "ratio":
            hs = (hs / sums).replace(np.nan, 0)
        else:
            raise ValueError("Unknown mode: " + repr(mode))
        pltbar(hs.to_numpy(), color=color, label=label)
        # print(sum_below, ag_id)
    # ax.legend(title="Altersgruppe", loc="center left")
    set_date_opts(ax, data_agg.index)
    if mode == "ratio":
        ax.set_ylim(bottom=0, top=1)


def ag_def_legend(fig):
    fig.legend(title="Altersgruppe", loc="center left", bbox_to_anchor=(-0.02, 0.5))


def get_data_path(dt: date) -> Path:
    dirname = dt.strftime(DATE_FMT)
    if (ARCHIVE_PATCH_ROOT / dirname).exists():
        return ARCHIVE_PATCH_ROOT / dirname
    return ARCHIVE_ROOT / dirname / "data"


def remove_labels_after(ax: plt.Axes, label: str, **kwargs):
    artists, labels = ax.get_legend_handles_labels()
    bev_idx = labels.index(label)
    artists, labels = artists[:bev_idx], labels[:bev_idx]
    # ax.set_legend_handles_labels(artists, labels)
    ax.legend(artists, labels, **kwargs)


# For use with https://github.com/statistikat/coronaDAT/
def get_zip_name(dt: date, pattern: Union[str, typing.Iterable[str]]) -> Path:
    if not isinstance(pattern, str):
        for pattern0 in pattern:
            try:
                return get_zip_name(dt, pattern0)
            except NoDataError:
                continue
        raise NoDataError(dt)
    paths = sorted(get_data_path(dt).glob(pattern))
    if not paths:
        raise NoDataError(dt)
    # if len(paths) > 1:
    #    logger.warning("More than one ZIP for dt=%s: %s", dt, paths)
    return paths[-1]


def loadall(
    fname: str,
    limit: Optional[int] = None,
    normalize=lambda df: df,
    csv_args: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    daydata_list = []  # type: list[pd.DataFrame]
    today = date(2023, 6, 30)  # date.today()
    for days_ago in count() if limit is None else range(limit):
        daydate = today - timedelta(days_ago)
        try:
            dd = load_day(fname, daydate, csv_args=csv_args)
        except NoDataError:
            logger.warning("Missing data for %s", daydate)
            if days_ago > 5:
                break
            continue
        dd["FileDate"] = pd.to_datetime(daydate)
        dd = normalize(dd)
        # print(dd.dtypes)
        # print(normalize.__code__.co_filename, normalize.__code__.co_firstlineno)
        daydata_list.append(dd)
    return pd.concat(daydata_list, ignore_index=True)


def get_day_zip_name(
    daydate: Optional[date] = None,
    pattern: Optional[Union[str, typing.Iterable[str]]] = None,
    try_earlier: Optional[bool] = None,
    warn_yesterday=True,
) -> Path:
    pattern = pattern or ("*_orig_csv_ages", "*_orig_csv_ages.zip", "*_*_orig_csv.zip")
    try_earlier = True if try_earlier is None and daydate is None else try_earlier
    today = date.today()
    try:
        return get_zip_name(daydate or today, pattern)
    except NoDataError:
        if not try_earlier:
            raise
        logger.log(
            logging.WARNING if warn_yesterday else logging.DEBUG,
            "No data for today, trying yesterday: %s",
            pattern,
        )
        try:
            return get_zip_name(today - timedelta(1), pattern)
        except NoDataError:
            logger.warning("No data for yesterday either, trying day before: %s", pattern)
            return get_zip_name(today - timedelta(2), pattern)


def load_day(
    fname: str,
    daydate: Optional[date] = None,
    pattern: Optional[str] = None,
    csv_args: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    zp = get_day_zip_name(daydate, pattern)
    with ZipFile(zp) if zp.is_file() else nullcontext(zp) as zf:
        try:
            dayfile = (zf / fname).open(encoding="utf-8") if isinstance(zf, Path) else zf.open(fname)
        except KeyError as e:
            raise NoDataError(daydate) from e
        with dayfile:
            return typing.cast(pd.DataFrame, pd.read_csv(dayfile, sep=";", **(csv_args or {})))


def simplify_agedata(ad: pd.DataFrame) -> pd.DataFrame:
    ad = ad.groupby(["Bundesland", "Altersgruppe", "Datum"]).sum(min_count=1)
    ad["AnzAktiv_i"] = ad["AnzAktiv"] / ad["AnzEinwohner"]
    ad["Anz_di"] = ad["Anz_d"] / ad["AnzEinwohner"]
    ad["Anz_di7"] = ad["Anz_di"].rolling(7).mean()
    ad["Anz_dI7"] = ad["Anz_di7"] * 100_000
    ad["AnzAktiv_i7"] = ad["AnzAktiv_i"].rolling(7).mean()
    ad["AnzAktiv_i14"] = ad["AnzAktiv_i"].rolling(14).mean()
    ad[["AnzAktiv_I7", "AnzAktiv_I14"]] = ad[["AnzAktiv_i7", "AnzAktiv_i14"]] * 100_000
    for i, level in enumerate(ad.index.levels):
        ad["_" + level.name] = ad.index.get_level_values(i)

    return ad


def parseutc_at(col, format=AGES_TIME_FMT):
    return pd.to_datetime(col, utc=True, format=format).dt.tz_convert("Europe/Vienna")


def parsedtonly(col, format=AGES_DATE_FMT):
    return pd.to_datetime(col, utc=True, format=format).tz_localize(None).normalize()


def mangle_eimpf(eimpf) -> pd.DataFrame:
    igw = (
        eimpf.drop(["Bevölkerung", "BundeslandID"], "columns")
        .melt(ignore_index=False)
        .query("variable.str.startswith('Gruppe_')")
    )
    igw["age"] = igw["variable"].str.split("_", expand=True).iloc[:, 1]
    igw.drop("variable", "columns")
    igw["age"].replace(">84", "84+", inplace=True)
    igw["age"].replace("<25", "25 & jünger", inplace=True)
    ig = igw.groupby(["Name", "age", "Datum"]).sum()
    ig.sort_index(inplace=True)
    ig["d"] = ig.diff()
    ig.reset_index("age", inplace=True)
    mask = ig["age"] != ig["age"].shift()
    ig.loc[mask, "d"] = ig.loc[mask, "value"]
    ig.set_index("age", append=True, inplace=True)
    ig.reorder_levels(["Name", "age", "Datum"])
    dsums = ig.groupby(level=("Name", "Datum")).sum()
    ig = ig.join(dsums, on=["Name", "Datum"], rsuffix="_sum")
    ig_wk = ig.groupby(
        [
            pd.Grouper(level="Name"),
            pd.Grouper(level="age"),
            pd.Grouper(level="Datum", freq="W"),
        ]
    ).sum()
    ig["d_rel"] = ig["d"] / ig["d_sum"]
    ig_wk["d_rel"] = ig_wk["d"] / ig_wk["d_sum"]
    return eimpf, ig, ig_wk


def load_eimpf() -> pd.DataFrame:
    eimpf = pd.read_csv(
        "timeline-eimpfpass.csv",
        sep=";",
        parse_dates=["Datum"],
        date_parser=lambda col: pd.to_datetime(col, utc=True),
        index_col=["Name", "Datum"],
    )
    igw = (
        eimpf.drop(["Bevölkerung", "BundeslandID"], "columns")
        .melt(ignore_index=False)
        .query("variable.str.startswith('Gruppe_')")
    )
    igw["age"] = igw["variable"].str.split("_", expand=True).iloc[:, 1]
    igw.drop("variable", "columns")
    igw["age"].replace(">84", "84+", inplace=True)
    igw["age"].replace("<25", "25 & jünger", inplace=True)
    ig = igw.groupby(["Name", "age", "Datum"]).sum()
    ig.sort_index(inplace=True)
    ig["d"] = ig.diff()
    ig.reset_index("age", inplace=True)
    mask = ig["age"] != ig["age"].shift()
    ig.loc[mask, "d"] = ig.loc[mask, "value"]
    ig.set_index("age", append=True, inplace=True)
    ig.reorder_levels(["Name", "age", "Datum"])
    dsums = ig.groupby(level=("Name", "Datum")).sum()
    ig = ig.join(dsums, on=["Name", "Datum"], rsuffix="_sum")
    ig_wk = ig.groupby(
        [
            pd.Grouper(level="Name"),
            pd.Grouper(level="age"),
            pd.Grouper(level="Datum", freq="W"),
        ]
    ).sum()
    ig["d_rel"] = ig["d"] / ig["d_sum"]
    ig_wk["d_rel"] = ig_wk["d"] / ig_wk["d_sum"]
    return eimpf, ig, ig_wk


def set_week_labels(fig: plt.Figure, ax: plt.Axes):
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=matplotlib.dates.MO, interval=1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("KW%W"))
    fig.autofmt_xdate()


@contextmanager
def calc_shifted(  # TODO: Read https://pandas.pydata.org/docs/user_guide/groupby.html and refactor to use index more if useful
    ds: pd.DataFrame,
    by: Union[str, List[str]],
    periods: int = 1,
    *,
    newcols: Optional[List[str]] = None,
    fill=np.nan,
):
    ds.sort_values(by=[by, "Datum"] if isinstance(by, str) else by + ["Datum"], inplace=True)
    shifted = ds.shift(periods)
    cols = ds.columns.copy()
    yield shifted
    autonewcols = ds.columns.difference(cols)
    if newcols is None:
        newcols = autonewcols
    elif not autonewcols.isin(newcols).all():
        raise ValueError("Created columns that were not specified: " + repr(autonewcols))
    if isinstance(by, str):
        mask = ds[by] != shifted[by]
    else:
        mask = np.zeros(len(ds), dtype=bool)
        for col in by:
            mask |= ds[col] != shifted[col]
    ds.loc[mask, newcols] = fill


def load_bezirke(daydate: typing.Optional[date] = None) -> pd.DataFrame:
    #  df = add_date(loadall("CovidFaelle_GKZ.csv"), "Datum")
    df = add_date(load_day("CovidFaelle_Timeline_GKZ.csv", daydate), "Time", format=AGES_TIME_FMT)
    df.rename(columns={"AnzahlTotTaeglich": "AnzahlTot"}, inplace=True)
    df["inz"] = df["AnzahlFaelle7Tage"] / df["AnzEinwohner"] * 100_000
    return df


def enrich_bez(bez_orig, fs):
    bez = (
        pd.concat(
            [
                bez_orig.set_index(["Datum", "Bezirk"]),
                fs.query("Bundesland != 'Wien'")
                .rename(columns={"Bundesland": "Bezirk"})
                .set_index(["Datum", "Bezirk"]),
            ]
        )
    ).reset_index()
    bez["InfektionenPro100"] = bez["AnzahlFaelleSum"] / bez["AnzEinwohner"] * 100
    bez["BundeslandID"] = bez["GKZ"] // 100
    bez["AnzahlFaelleSumProEW"] = bez["AnzahlFaelleSum"] / bez["AnzEinwohner"]
    bez["AnzahlFaelleProEW"] = bez["AnzahlFaelle"] / bez["AnzEinwohner"]
    bez.loc[np.isfinite(bez["GKZ"]), "Bundesland"] = bez["BundeslandID"].map(BUNDESLAND_BY_ID)
    bez.loc[~np.isfinite(bez["GKZ"]), "Bundesland"] = bez["Bezirk"]

    with calc_shifted(bez, "Bezirk", 7, newcols=["inz_a7", "dead"]):
        bez["inz_a7"] = bez["inz"].rolling(7).mean()
        bez["dead"] = calc_inz(bez, "AnzahlTot")
    with calc_shifted(bez, "Bezirk", 14, newcols=["dead14"]):
        bez["dead14"] = bez["AnzahlTot"].rolling(14).sum() / bez["AnzEinwohner"] * 100_000
    with calc_shifted(bez, "Bezirk", 28, newcols=["dead28"]):
        bez["dead28"] = bez["AnzahlTot"].rolling(28).sum() / bez["AnzEinwohner"] * 100_000
    with calc_shifted(bez, "Bezirk", 91, newcols=["dead91"]):
        bez["dead91"] = bez["AnzahlTot"].rolling(91).sum() / bez["AnzEinwohner"] * 100_000
    enrich_inz(bez, "InfektionenPro100")
    enrich_inz(bez, "inz_a7")
    enrich_inz(bez)
    return bez


def plot_bez(df: pd.DataFrame, which: Sequence[str], **kwargs):
    return sns.lineplot(
        data=df[df["Bezirk"].str.fullmatch("|".join(re.escape(s) for s in which))],
        x="Datum",
        y="inz",
        hue="Bezirk",
        hue_order=which,
        marker=".",
        linewidth=1.1,
        **kwargs,
    )


a4_dims = (11.7, 8.27)
a3_dims = (a4_dims[1] * 2, a4_dims[0])
a2_dims = (a3_dims[1] * 2, a3_dims[0])
a5_dims = (a4_dims[1], a4_dims[0] / 2)


def render_bez(df: pd.DataFrame, which: Sequence[str], figname: str):
    fig, axs = plt.subplots(nrows=2, figsize=a4_dims, num=figname)
    plot_bez(df, which, ax=axs[0], legend=False)
    plot_bez(df, which, ax=axs[1])
    axs[0].set_ylim(bottom=0)
    ymax = axs[0].get_ylim()[1]  # type: float
    axs[1].set_yscale("log", base=2)
    axs[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[1].set_yticks([1, 2, 3, 5, 10, 20, 35, 50])
    for ax in axs:
        lineargs = dict(color="grey", linewidth=1)
        # ax.axvline(x=date(2021, 7, 1), **lineargs)
        # ax.axvline(x=date(2021, 6, 10), **lineargs)
        # ax.axvline(x=date(2021, 6, 26), linewidth=1, color="r")
        for y in (5, 10, 20, 50, 100, 200, 300, 400):
            if y >= ymax:
                break
            ax.axhline(y=y, zorder=-1, **lineargs)
    fig.legend(loc="lower left")
    axs[1].get_legend().remove()
    return fig


kfz = None


def bezname_to_kfz(bezname: str) -> str:
    global kfz  # pylint:disable=global-statement
    if kfz is None:
        kfz = pd.read_csv(
            "kennzeichen_at.csv",
            sep=";",
            names=["ID", "Bezirk", "Bundesland"],
            index_col="Bezirk",
            quotechar='"',
        )
    return kfz.loc[bezname]["ID"]


AGE_FROM_MAP = {
    "<5": 0,
    "5-14": 5,
    "15-24": 15,
    "25-34": 25,
    "35-44": 35,
    "45-54": 45,
    "55-64": 55,
    "65-74": 65,
    "75-84": 75,
    ">84": 85,
}

AGE_TO_MAP = {
    "<5": 4,
    "5-14": 14,
    "15-24": 24,
    "25-34": 34,
    "35-44": 44,
    "45-54": 54,
    "55-64": 64,
    "65-74": 74,
    "75-84": 84,
    ">84": np.iinfo(np.uint8).max,
}


def enrich_ag(ag_o, agefromto=True, parsedate=True, bev: typing.Optional[pd.Series] = None):
    catcols = ["BundeslandID", "AltersgruppeID", "Geschlecht"]
    faelle_ag = norm_df(ag_o, datecol="Time", format=AGES_TIME_FMT) if parsedate else ag_o
    faelle_ag["AnzEinwohnerFixed"] = faelle_ag["AnzEinwohner"]
    if bev is not None:
        faelle_ag.set_index(["Datum"] + catcols, inplace=True)
        # display("f", faelle_ag.index.dtypes)
        # display("b", bev.index.dtypes)
        faelle_ag["AnzEinwohner"] = bev
        faelle_ag.reset_index(inplace=True)
    if "AnzahlFaelleSum" not in faelle_ag.columns:
        faelle_ag = faelle_ag.rename(columns={"Anzahl": "AnzahlFaelleSum", "AnzahlTot": "AnzahlTotSum"})
    if agefromto:
        faelle_ag["AgeFrom"] = faelle_ag["Altersgruppe"].map(AGE_FROM_MAP).astype(np.uint8)
        faelle_ag["AgeTo"] = faelle_ag["Altersgruppe"].map(AGE_TO_MAP).astype(np.uint8)
    if "FileDate" in ag_o.columns:
        catcols.append("FileDate")
    with calc_shifted(faelle_ag, catcols, newcols=["AnzahlFaelle"]) as shifted:
        faelle_ag["AnzahlFaelle"] = faelle_ag["AnzahlFaelleSum"] - shifted["AnzahlFaelleSum"]
    with calc_shifted(faelle_ag, catcols, 7) as shifted:
        faelle_ag["AnzahlFaelle7Tage"] = faelle_ag["AnzahlFaelleSum"] - shifted["AnzahlFaelleSum"]
        faelle_ag["inz"] = calc_inz(faelle_ag)
    faelle_ag["Gruppe"] = faelle_ag["Bundesland"].str.cat([faelle_ag["Altersgruppe"], faelle_ag["Geschlecht"]], sep=" ")
    return faelle_ag


def load_ag(daydate: typing.Optional[date] = None, bev: typing.Optional[pd.Series] = None) -> pd.DataFrame:
    return enrich_ag(load_day("CovidFaelle_Altersgruppe.csv", daydate), bev=bev)


def enrich_inz(
    df: pd.DataFrame,
    inzcol: str = "inz",
    catcol: Union[str, List[str]] = "Bezirk",
    dailycol: str = "AnzahlFaelle",
):
    for d in (1, 3, 7, 14):
        sfx = str(d) if d != 1 else ""
        gcol = f"{inzcol}_g{sfx}"
        dgcol = f"{inzcol}_dg{sfx}"
        dcol = f"{inzcol}_d{sfx}"
        with calc_shifted(df, catcol, d, newcols=[gcol, dcol, dgcol]) as shifted:
            df[gcol] = df[inzcol] / shifted[inzcol]
            df[dcol] = df[inzcol] - shifted[inzcol]
            df[dgcol] = df[dailycol] / shifted[dailycol]
        df[f"{inzcol}_prev{sfx}"] = df[inzcol] - df[dcol]


def g_to_pct(g):
    result = format(g * 100 - 100, "+.3n").replace("-", "‒") + "%"
    return "±0%" if result == "+0%" else result


def set_pctdiff_formatter(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: g_to_pct(v)))


def plt_manychange(
    df: pd.DataFrame,
    catcol: str = "Bezirk",
    inzcol: str = "inz",
    weightcol: str = "AnzEinwohner",
):
    cmap = plt.get_cmap(div_l_cmap)

    def calc_colors(
        gs: pd.Series,
    ):
        cnorm = matplotlib.colors.TwoSlopeNorm(
            vmin=min(0.99, gs.quantile(0.05)),
            vcenter=1,
            vmax=max(1.01, gs.quantile(0.95)),
        )
        return cmap(cnorm(gs))

    colors = calc_colors(df[f"{inzcol}_g7"])
    colors1 = calc_colors(df[f"{inzcol}_g"])

    def plt_rising(col, d_col, **scatterargs):
        rising = df[d_col] >= 0
        sc_rising = ax.scatter(y=df.loc[rising, catcol], x=df.loc[rising, col], marker="4", **scatterargs)
        sc_falling = ax.scatter(y=df.loc[~rising, catcol], x=df.loc[~rising, col], marker="3", **scatterargs)
        return sc_rising, sc_falling

    ax = df.plot(
        x=catcol,
        y=f"{inzcol}_d7",
        figsize=(8, 8),
        kind="barh",
        left=df[f"{inzcol}_prev7"],
        color=colors,
        legend=False,
        zorder=-0.5,
        linewidth=0,
    )
    # ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    # ax.grid(axis="x", which="minor", lw=0.3)
    # ax.set_axisbelow(True)

    # ax.barh(
    #     y=df[catcol],
    #     x=df[f"d_{inzcol}"],
    #     left=df[f"{inzcol}_prev"],
    #     color=colors,
    #     height=1,
    #     zorder=-0.5
    # )
    df.plot(
        ax=ax,
        x=catcol,
        y=f"{inzcol}_d",
        kind="barh",
        left=df[f"{inzcol}_prev"],
        color=colors1,
        legend=False,
        zorder=-0.5,
        width=0.2,
        linewidth=0,
    )

    ax.scatter(y=df[catcol], x=df[inzcol], marker="|", color=colors)
    hpp, _ = plt_rising(f"{inzcol}_prev14", f"{inzcol}_d14", color="y")
    hp = ax.scatter(y=df[catcol], x=df[f"{inzcol}_prev7"], marker="|", color="k")
    hyr, _ = plt_rising(f"{inzcol}_prev", f"{inzcol}_d", color="k")

    ax.tick_params(axis="x", top=False, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_label_position("top")
    ax.grid(False)
    ax.grid(axis="x")

    # ax.grid(axis="y", linestyle="dotted")
    ewnorm = matplotlib.colors.Normalize(
        vmin=df[weightcol].quantile(0.0),
        vmax=df[weightcol].quantile(0.8, interpolation="lower"),
    )
    ewcmap = sns.color_palette("Greys", as_cmap=True)
    for _, catrow in df.iterrows():
        ax.axhline(
            y=catrow[catcol],
            zorder=-1,
            color=ewcmap(ewnorm(catrow[weightcol])),
            linewidth=0.5,
        )
    xlim = ax.get_xlim()
    ax.set_xlim((df[["inz", "inz_prev", "inz_prev7", "inz_prev14"]].min().min(), xlim[1]))
    # ax.set_yticklabels(ha="left")
    curinz = ax.plot(df[inzcol], df[catcol], color="k", alpha=0.25)

    aa7 = None
    if (inzcol + "_a7") in df.columns:
        aa7 = ax.scatter(df[inzcol + "_a7"], df[catcol], marker=".", color="k", s=10)
    # ax.plot(df[f"{inzcol}_prev"], df[catcol], color="k", alpha=0.2, linestyle="--")
    # ax.plot(df[f"{inzcol}_prev7"], df[catcol], color="k", alpha=0.2, linestyle=":")
    # ax.plot(df[f"{inzcol}_prev14"], df[catcol], color="k", alpha=0.1, linestyle="-.")

    # empty_artist = matplotlib.patches.Rectangle((0, 0), 1, 1, visible=False)
    handles = [hpp, hp, hyr, curinz[0]]
    labels = ["Vor 2 Wochen", "Vor 1 Woche", "Gestern", "Aktuell"]
    if aa7:
        handles.append(aa7)
        labels.append("7-Tage-Schnitt")
    ax.legend(handles, labels, loc="lower right")
    ax.set_ylabel(None)

    return ax


def calc_inz(df: pd.DataFrame, dailycol: str = "AnzahlFaelle", ewcol: str = "AnzEinwohner"):
    # The order here is important to not get a (tiny) negative result.
    # Numerical instability is a pain...
    return df[dailycol].rolling(7).sum() / df[ewcol] * 100_000


def enrich_hosp_data(faelle: pd.DataFrame) -> None:
    faelle["FZHospAlle"] = faelle["FZICU"] + faelle["FZHosp"]
    icu_caps = faelle["Bundesland"].replace(ICU_LIM_GREEN_BY_STATE)
    faelle["ICU_green_use"] = faelle["FZICU"] / icu_caps

    with calc_shifted(faelle, "Bundesland", 3):
        faelle["FZHospAlle_a3"] = faelle["FZHospAlle"].rolling(3).mean()
        faelle["FZICU_a3"] = faelle["FZICU"].rolling(3).mean()
        faelle["FZICU_m3"] = faelle["FZICU"].rolling(3).max()
        faelle["FZHosp_a3"] = faelle["FZHosp"].rolling(3).mean()
    faelle["ICU_green_use_m3"] = faelle["FZICU_m3"] / icu_caps
    faelle["ICU_green_use_a3"] = faelle["FZICU_a3"] / icu_caps
    faelle["hosp"] = faelle["FZHospAlle"] / faelle["AnzEinwohner"] * 100_000
    faelle["nhosp"] = faelle["FZHosp"] / faelle["AnzEinwohner"] * 100_000
    faelle["icu"] = faelle["FZICU"] / faelle["AnzEinwohner"] * 100_000


def load_faelle(daydate: typing.Optional[date] = None, bev: typing.Optional[pd.Series] = None) -> pd.DataFrame:
    faelle_ag = load_ag(daydate, bev)
    faelle_ag["AGCoarse"] = "<25"
    faelle_ag.loc[faelle_ag["AgeFrom"] >= 25, "AGCoarse"] = "25-54"
    faelle_ag.loc[faelle_ag["AgeFrom"] > 54, "AGCoarse"] = ">54"
    faelle_ag_u55 = faelle_ag[faelle_ag["AgeFrom"] < 55].copy()
    faelle_ag_u55["AGCoarse"] = "<55"
    faelle_ag = pd.concat([faelle_ag, faelle_ag_u55])

    ag_coarse = (
        faelle_ag.drop(columns="AltersgruppeID")
        .groupby(by=["AGCoarse", "Datum", "Bundesland", "BundeslandID"], as_index=False)
        .sum()
    )
    with calc_shifted(ag_coarse, ["Bundesland", "AGCoarse"], 7, newcols=["inz"]):
        ag_coarse["inz"] = calc_inz(ag_coarse)
    idxs = ["Datum", "BundeslandID", "Bundesland"]
    ag_coarse.set_index(["AGCoarse"] + idxs, inplace=True)
    faelle = add_date(load_day("CovidFaelle_Timeline.csv", daydate), "Time", format=AGES_TIME_FMT)
    faelle["AnzEinwohnerFixed"] = faelle["AnzEinwohner"]
    if bev is not None:
        faelle.set_index(["Datum", "BundeslandID"], inplace=True)
        faelle["AnzEinwohner"] = bev.groupby(["Datum", "BundeslandID"]).sum()
        faelle.reset_index(inplace=True)
    faelle.rename(columns={"AnzahlTotTaeglich": "AnzahlTot"}, inplace=True)
    # breakpoint()
    faelle = (
        faelle.set_index(idxs)
        .join(
            add_date(
                load_day("CovidFallzahlen.csv", daydate),
                "Meldedat",
                format=AGES_DATE_FMT,
            )
            .replace("Alle", "Österreich")
            .set_index(idxs)
        )
        .join(ag_coarse.xs(">54")[["inz"]].rename(columns={"inz": "ag55_inz"}))
        .join(ag_coarse.xs("<55")[["inz"]].rename(columns={"inz": "agU55_inz"}))
        .join(ag_coarse.xs("<25")[["inz"]].rename(columns={"inz": "agU25_inz"}))
        .join(ag_coarse.xs("25-54")[["inz"]].rename(columns={"inz": "ag25to54_inz"}))
    )
    faelle.reset_index(inplace=True)
    enrich_hosp_data(faelle)
    with calc_shifted(faelle, "Bundesland", 7, newcols=["inz", "dead"]):
        faelle["inz"] = calc_inz(faelle)
        faelle["dead"] = (faelle["AnzahlTot"] / faelle["AnzEinwohner"]).rolling(7).sum() * 100_000
    with calc_shifted(faelle, "Bundesland", 28):
        faelle["dead28"] = (faelle["AnzahlTot"] / faelle["AnzEinwohner"]).rolling(28).sum() * 100_000
    return faelle


def load_faelle_at(daydate: typing.Optional[date] = None) -> pd.DataFrame:
    return load_faelle(daydate).query("Bundesland == 'Österreich'")


bglinestyle_nc = dict(zorder=-1, linewidth=0.7)
bglinestyle = dict(color="grey", **bglinestyle_nc)


def plot_with_hline(ds: pd.DataFrame, ax: plt.Axes, col: str, color: str, label: str, **kwargs):
    ls = dict(bglinestyle)
    ls["color"] = color
    ds.plot(ax=ax, color=color, label=label, y=col, **kwargs)
    valy = ds.iloc[-10:][col][~np.isnan(ds[col])]
    if len(valy) > 0:
        ax.axhline(y=valy.iloc[-1], linestyle="--", **ls)


NOTABLE_DATES = {
    date(2020, 3, 16): (0, True, "LD 1"),
    date(2020, 4, 14): (0, False, "LD 1 Ende"),
    date(2020, 5, 1): (1, False, "Lockerungs-VO"),
    # Schulöffnung siehe
    # https://www.bmbwf.gv.at/Themen/schule/beratung/corona/corona_info/corona_etappenplan.html
    date(2020, 5, 4): (3, False, "Schulöffnung: Abschlusskl."),
    date(2020, 5, 15): (0, False, "Gastroöffnung"),
    date(2020, 5, 18): (2, False, "Schulöffnung: Unterst."),
    date(2020, 6, 3): (3, False, "Schulöffnung voll"),
    date(2020, 6, 15): (2, False, "Lockerung MNS, Gastro"),
    date(2020, 7, 1): (1, False, "Lockerung Sport, Event, MNS"),
    date(2020, 7, 11): (0, True, "Sommerferien"),
    date(2020, 7, 24): (2, True, "MNS neu"),
    date(2020, 9, 4): (0, False, "Sommerferien Ende"),
    date(2020, 9, 14): (2, True, "MNS Verschärfung"),
    date(2020, 9, 21): (2, True, "Eventbeschränkungen"),
    date(2020, 10, 25): (2, True, "Gastrobeschränkung"),
    date(2020, 11, 3): (0, True, "Gastroschließung, Ausgangsbeschränkung"),
    date(2020, 11, 17): (1, True, "Handels- & Schulschließung"),
    date(2020, 12, 7): (1, False, "Handels- & tw. Schulöffnung"),
    date(2020, 12, 26): (1, True, "Handels- & Schulschließung 2"),
    date(2021, 1, 25): (2, True, "FFP2"),
    date(2021, 2, 7): (2, False, "Schulöffnung"),
    date(2021, 2, 8): (0, False, "Handelsöffnung, Ende Kontaktbesch."),
    date(2021, 2, 12): (2, True, "Ausreisetest Tirol"),
    date(2021, 3, 10): (2, False, "Ausreisetest Tirol Ende"),
    date(2021, 3, 15): (2, False, "Öffnung Sport + Vorarlberg"),
    date(2021, 4, 1): (0, True, "Handels- & Schulschließung Ost"),
    date(2021, 4, 12): (2, False, "Tw. Schulöffung Ost"),
    date(2021, 5, 2): (1, False, "Handels- & Schulöffnung Ost"),
    date(2021, 5, 19): (0, False, "Gastroöffnung + 3G"),
    date(2021, 6, 10): (2, False, "Lockerung Events"),
    date(2021, 7, 1): (0, False, "Nachtgastroöffnung, Lockerungen"),
    date(2021, 7, 22): (2, True, "Nachtgastro: 2G"),
    date(2021, 9, 6): (0, False, "Sommerferien Ende"),
}

BEZ_GKZ_GEO_ORDER = (
    802,
    803,
    804,
    801,  # Vorarlberg
    708,
    706,
    702,
    703,
    701,
    709,
    705,
    704,
    707,  # Tirol
    506,
    504,
    505,
    502,
    501,
    503,  # Salzburg
    # Oberösterreich
    404,  # Innviertel
    412,
    414,
    408,
    413,  # Mühlviertel
    416,
    406,
    411,  # OÖ Zentralraum
    401,
    410,
    405,
    403,
    418,
    417,
    407,  # Traunviertel
    409,
    415,
    402,  # Stery-Kirchdorf
    305,
    303,
    320,
    315,  # AT121
    325,
    309,
    322,
    311,
    313,
    301,  # AT124 Waldviertel
    302,
    319,  # AT123
    314,
    318,
    323,
    304,
    306,
    317,  # AT122
    321,
    312,
    310,
    316,
    308,
    307,  # AT126 + AT125 (teile von 310, 316, 308)
    900,
    107,
    102,
    103,
    101,
    106,
    108,
    109,
    104,
    105,
    623,
    622,
    617,  # AT224
    621,
    611,  # AT223
    612,  # AT222 Liezen
    614,
    620,  # AT226 Westliche Obersteiermark (Mur*)
    606,
    601,  # AT221 Graz+Umgebung
    610,
    603,
    616,  # AT225 West- und Südsteiermark
    209,
    208,
    205,  # AT213 Unterkärnten
    201,
    204,
    210,
    207,  # AT211 Klagenfurt-Villach
    202,
    203,
    206,  # AT212 Oberkärnten
)
assert len(set(BEZ_GKZ_GEO_ORDER)) == len(BEZ_GKZ_GEO_ORDER), str(Counter(BEZ_GKZ_GEO_ORDER).most_common(1))


def set_logscale(ax, reduced=False, axis="y"):
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log", base=2)
    laxis = ax.yaxis if axis == "y" else ax.xaxis
    laxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[0.3, 1] if reduced else [0.2, 0.3, 0.5, 1]))
    fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    laxis.set_major_formatter(fmt)


def plot_detail_ax(faelle: pd.DataFrame, ax: plt.Axes):
    faelle = faelle.replace(np.inf, np.nan)
    latest = faelle.index.get_level_values("Datum")[-1]
    first = faelle.index.get_level_values("Datum")[0]
    growth_col = "inz_g7"
    plot_with_hline(
        faelle.where(faelle["AnzahlFaelle7Tage"] >= 150),
        ax,
        growth_col,
        "blue",
        "Inzidenzanstieg/Woche (linke Skala)",
        marker=",",
        mfc="white",
    )
    # ax.plot(faelle.index[-1:], faelle.iloc[-1:][growth_col], color="blue", marker=".", lw=0, markersize=3)
    ax.set_ylabel("Faktor Inzidenz ggü. 7 Tage vorher")
    # faelle[["g_inz7a"]].plot(ax=ax, color="blue")
    ax2 = ax.twinx()
    # ax2.set_yscale("log", base=2)
    ax2.set_ylabel("Pro 100.000 Personen")
    set_logscale(ax2)
    # ax2.set_yticks(INZ_TICKS)

    # pred = pd.to_datetime(latest + timedelta(14))
    # faelle.loc[pred, "inz"] = faelle.iloc[-1]["inz"] * faelle.iloc[-1]["g_inz7"] ** 2

    maxy = None
    if "Abwassersignal" in faelle.columns or "Abwasser_y" in faelle.columns:
        # faelle["i14"] = (faelle["AnzahlFaelle"].rolling(14).sum()) / faelle["AnzEinwohner"] * 100_000
        plot_with_hline(faelle, ax2, "inz", "darkgreen", "7-Tage-Inzidenz")
        maxy = faelle["inz"].max()

        if "Abwassersignal" in faelle.columns:
            abwcol = "Abwassersignal"
            if False:  # Annahme: Abwassersignal ist roh, d.h. nicht auf Abdeckung & EW normiert
                msk_erw = faelle.index >= pd.to_datetime("2023-02-01")

                # "Rund 52%" https://web.archive.org/web/20230110152232/https://abwassermonitoring.at/dashboard/
                faelle.loc[~msk_erw, "Abwassersignal"] = faelle["Abwassersignal"] / (faelle["AnzEinwohner"] * 0.515)

                # "Mehr als 58%" https://abwassermonitoring.at/dashboard/
                faelle.loc[msk_erw, "Abwassersignal"] = faelle["Abwassersignal"] / (faelle["AnzEinwohner"] * 0.582)

                faelle["Abwassersignal"] *= 7  # "Daumen mal Pi"
            else:  # Annahme: Signal ist bereits auf abgedeckte Bev. normiert
                # "Daumen mal Pi"
                # x2 wäre ggf. angemessen wegen DZ im Jänner 2022 http://www.dexhelpp.at/de/immunisierungsgrad/
                faelle["Abwassersignal"] /= 650_000  # /= (650_000 / 2)
        else:  # Abwasser_y
            abwcol = "Abwasser_y"
            syncarea = faelle.query("Datum >= '2022-09-01' and Datum <= '2022-11-01'")
            maxabwi = syncarea["Abwasser_y"].idxmax()
            maxinz = (
                syncarea.loc[
                    (syncarea.index >= maxabwi - timedelta(14)) & (syncarea.index <= maxabwi + timedelta(14)), "inz"
                ].max()
                * 2
            )  # Guesstimate
            adj = maxinz / syncarea.loc[maxabwi, "Abwasser_y"]
            faelle["Abwasser_y"] = faelle["Abwasser_y"].where(faelle.index >= pd.to_datetime("2022-09-01"))
            faelle["Abwasser_y"] *= adj

        # display(faelle[np.isfinite(faelle[abwcol])][abwcol].iloc[-1])
        plot_with_hline(
            faelle[np.isfinite(faelle[abwcol])],
            ax2,
            abwcol,
            "hotpink",
            "Abwassersignal (Inzidenz-justiert, ohne Einheit)",
        )
        maxy = max(maxy, faelle[abwcol].max())
    elif "ag55_inz" in faelle.columns:
        plot_with_hline(faelle, ax2, "agU55_inz", "springgreen", "7-Tage-Inzidenz Altersgruppe <55")
        plot_with_hline(faelle, ax2, "ag55_inz", "yellowgreen", "7-Tage-Inzidenz Altersgruppe 55+")
        maxy = faelle[["agU55_inz", "ag55_inz"]].max().max()
    else:
        plot_with_hline(faelle, ax2, "inz", "darkgreen", "7-Tage-Inzidenz")
        maxy = faelle["inz"].max()
    if "nhosp" in faelle.columns:
        if "HospPostCov" in faelle.columns:
            phosp = faelle["HospPostCov"].resample("1D").interpolate() / faelle["AnzEinwohner"] * 100_000
            plot_with_hline(
                faelle.query("Datum >= '2022-11-01'"), ax2, "nhosp", "wheat", "_Normalstationsbelegung ohne PostCov"
            )
            faelle["nhosp"] = faelle["nhosp"].add(
                phosp.where(faelle.index >= pd.to_datetime("2022-11-02")), fill_value=0
            )
            extralbl = " (hell: Zählweise ab 2.11.22)"
        else:
            extralbl = ""
        plot_with_hline(faelle, ax2, "nhosp", "darkorange", "Normalstationsbelegung" + extralbl)
    if "icu" in faelle.columns:
        plot_with_hline(faelle, ax2, "icu", "darkred", "ICU-Belegung")
    if "dead" in faelle.columns:
        lastls = "--" if faelle["dead"].last_valid_index().year > 2021 else "-"
        ax2.plot(faelle.iloc[-8:]["dead"], color="black", alpha=0.3, ls=lastls)
        plot_with_hline(faelle.query("Datum < '2022-01-01'"), ax2, "dead", "black", "7-Tage-Summe Tote")
        plot_with_hline(faelle.query("Datum >= '2022-01-01'"), ax2, "dead", "black", "_7-Tage-Summe Tote vol.", ls="--")

    ax.grid(axis="x", which="major")
    # ax.axvline(x=latest, linewidth=1, color="r")
    ymax: float = max(
        min(4, faelle.iloc[150:][growth_col].max() * 1.1),
        faelle.iloc[-7:][growth_col].max() * 1.02,
    )
    one_year_earlier = latest - np.timedelta64(365, "D")
    if first < one_year_earlier and False:
        ax.axvline(x=one_year_earlier, linestyle="--", color="k")
        ax.text(one_year_earlier, ymax * 0.99, one_year_earlier.strftime(" %x"), va="top")
    # ax.set_yticks([0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 10])
    ax.axhline(y=1, color="blue", linewidth=0.5, alpha=0.8)
    ax.set_ylim(top=ymax, bottom=max(0, faelle.iloc[60:][growth_col].min() * 0.95))
    # ax.grid(axis="y", color="lightblue", zorder=-1, linewidth=0.5)
    # ax2.grid(axis="y", color="lightgrey", zorder=-1)
    ax2.set_ylim(bottom=max(0.1, ax2.get_ylim()[0]), top=maxy * 1.1)
    # print(faelle)
    # print(ax.get_ylim())

    # important_dates = {dt: desc for dt, desc in NOTABLE_DATES.items() if desc[0] <= 0}
    # for dt, desc in important_dates.items():
    #     _, tightening, note = desc
    #     ax.axvline(x=dt, zorder=-1, linestyle=":", color="r" if tightening else "green")
    #     ax.text(dt, ymax, note,  fontsize=4) #rotation=45, verticalalignment="top")

    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))

    prop = {"size": 10}
    # ax.legend(loc="upper left", prop=props)
    # ax2.legend(loc="lower right", bbox_to_anchor=(0.77, 0), prop=props)
    # fig.legend(loc="upper left", props=props, ncols=2)
    legend_info = [x1 + x2 for x1, x2 in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    ax2.legend(
        *legend_info,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        prop=prop,
        frameon=False,
    )
    ax.get_legend().remove()

    xax: matplotlib.axis.XAxis = ax.xaxis
    # xax.set_minor_locator(matplotlib.dates.WeekdayLocator())
    # xax.set_minor_formatter(matplotlib.dates.DateFormatter("%W"))
    xax.set_major_locator(matplotlib.dates.MonthLocator())
    # xax.set_minor_locator(
    # matplotlib.dates.WeekdayLocator(byweekday=matplotlib.dates.MO, interval=1)
    # )
    xax.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    # xax.set_minor_formatter(matplotlib.dates.DateFormatter("%d."))
    # xax.set_major_formatter(matplotlib.dates.DateFormatter("\n%b"))
    # fig.autofmt_xdate(which="minor", rotation=70)
    # ax.tick_params(axis="x", which="minor", pad=0, labelsize=6)
    # ax.tick_params(axis="x", which="major", pad=10)
    ax.set_xlim((first, latest + timedelta(2)))
    ax2.set_xlim((first, latest + timedelta(2)))

    return ax2


def plot_detail(faelle: pd.DataFrame, figname: str):
    fig, ax = plt.subplots(figsize=(16 * 0.7, 9 * 0.7))
    plot_detail_ax(faelle, ax)
    fig.autofmt_xdate(which="major")
    fig.tight_layout()
    return fig


BUNDESLAND_BY_ID = {
    1: "Burgenland",
    2: "Kärnten",
    3: "Niederösterreich",
    4: "Oberösterreich",
    5: "Salzburg",
    6: "Steiermark",
    7: "Tirol",
    8: "Vorarlberg",
    9: "Wien",
    10: "Österreich",
}


def load_gem_impfungen(daydate: Optional[date] = None):
    gis = add_date(
        typing.cast(
            pd.DataFrame,
            pd.read_csv(
                get_day_zip_name(
                    daydate,
                    "*_impfdaten_orig_csv_bmsgpk.zip",
                    try_earlier=daydate is None,
                    warn_yesterday=False,
                ),
                sep=";",
                dayfirst=True,
            ),
        ),
        "Datum",
    )
    gis["Gemeindecode"] = gis["Gemeindecode"].astype(str)
    gis.set_index("Gemeindecode", inplace=True)
    gis["BundeslandID"] = gis.index.get_level_values("Gemeindecode").str[0].astype(int)
    gis["Bundesland"] = gis["BundeslandID"].astype(int).replace(BUNDESLAND_BY_ID)
    gis["BezirkID"] = gis.index.get_level_values("Gemeindecode").str[0:3].astype(int)
    return gis


def group_impfungen_by_bez(gis: pd.DataFrame) -> pd.DataFrame:
    bis = gis.groupby(["Bundesland", "BundeslandID", "BezirkID", "Datum"]).sum()

    # Wien braucht eine Speziallösung da es in den AGES-Daten nicht
    # in Bezirke eingeteilt ist, in den Impfdaten (Gemeindeebene) aber schon
    bis.loc[("Wien", 9, 900, bis.index.get_level_values("Datum")[-1])] = bis.xs("Wien").sum()
    bis.reset_index(["Bundesland", "BundeslandID"], inplace=True)

    for pctcol in bis.columns:
        if "Pro100" in pctcol:
            bis[pctcol] = bis[pctcol.replace("Pro100", "")] / bis["Bevölkerung"] * 100
    return bis


def load_wahl():
    ws = typing.cast(
        pd.DataFrame,
        pd.read_csv("wahl_20191007_163653.csv", sep=";", encoding="cp1252"),
    )
    ws.drop(columns=ws.columns[-1], inplace=True)
    ws.rename(columns={ws.columns[0]: "GebietID"}, inplace=True)
    ws["GebietID"] = ws["GebietID"].str.lstrip("G")
    ws.set_index("GebietID", inplace=True)

    return ws


def filterlatest(ds: pd.DataFrame, x="Datum") -> pd.DataFrame:
    return ds[ds[x] == ds[x].iloc[-1]]


def print_predictions(faelle_at: pd.DataFrame, col: str, name: str, target: int = 400, days=14):
    ds = faelle_at[col]
    today = faelle_at.index.get_level_values("Datum")[-1]
    cur_inz = ds.iloc[-1]
    print("{:40} (zuletzt {:6.1f}) in {:2.0f} Tagen / {:4.0f} erreicht am:".format(name, cur_inz, days, target))
    logdiff = np.log(target) - np.log(cur_inz)

    def indays(n):
        return (today + np.timedelta64(int(np.ceil(n)), "D")).strftime("%x")

    def print_based_on_distance(label: str, distance: int, skip_recent: int = 1):
        cur_inz = ds.iloc[-1 - skip_recent]
        prev_inz = ds.iloc[-1 - skip_recent - distance]
        g_inz = cur_inz / prev_inz
        print(
            "  Bei {:30} ({:6.1f} * {:1.2f}): {:4.1f} / {}".format(
                label,
                prev_inz,
                g_inz,
                cur_inz * g_inz ** (days / distance),
                indays(logdiff / np.log(g_inz) * distance),
            )
        )

    print_based_on_distance("heutigem Anstieg", distance=1, skip_recent=0)
    print_based_on_distance("Anstieg vorgestern auf gestern", distance=1)
    print_based_on_distance("aktuellem Wochenanstieg", distance=7)
    print_based_on_distance("aktuellem 2-Wochenanstieg", distance=14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-predict", action="store_true")
    args = parser.parse_args()

    if not args.only_predict:
        sns.set_style("whitegrid", {"axes.grid": False, "legend.fontsize": 8})
        df = load_bezirke()
        enrich_inz(df)
        bez_recent = df[df["Datum"] >= pd.to_datetime(date.today() - timedelta(45))]
        bez_today = filterlatest(bez_recent)
        top = bez_today.sort_values(by="inz", ascending=False).iloc[:8]["Bezirk"]
        render_bez(bez_recent, top, "Hochinzidenz").savefig("df-top.png")
        hot = bez_today.sort_values(by="inz_g7", ascending=False).query("inz > 10").iloc[:8]["Bezirk"]
        render_bez(bez_recent, hot, "Meiststeigend Woche & > 10").savefig("df-hot.png")
        bez_of_interest = ("Wien", "Graz(Stadt)", "Linz(Stadt)")
        render_bez(bez_recent, bez_of_interest, "Auswahl").savefig("df-selected.png")

        plot_detail(
            df.query("Bezirk == 'Wels(Stadt)'").set_index("Datum"),
            "Steigerungsrate Wels",
        ).savefig("wels-detail.png")
        plot_detail(
            df.query("Bezirk == 'Imst'").set_index("Datum"),
            "Steigerungsrate Imst",
        ).savefig("imst-detail.png")

    faelle = load_faelle().set_index("Datum")
    enrich_inz(faelle, catcol="BundeslandID")

    faelle_at = faelle.query("Bundesland == 'Österreich'").copy()
    with setlocale(""):
        print_predictions(faelle_at, "ag55_inz", "Inzidenz 55+")
        print_predictions(faelle_at, "agU55_inz", "Inzidenz U55")
        print_predictions(faelle_at, "inz", "Inzidenz (Ges.)")
        print_predictions(faelle_at, "inz", "Inzidenz (Ges.)", days=18, target=600)
        faelle_at["AnzahlFaelle_a7"] = faelle_at["AnzahlFaelle"].rolling(7).mean()
        print_predictions(faelle_at, "AnzahlFaelle_a7", "Neuinfektion", days=14, target=40_000)
        print_predictions(
            faelle_at,
            "FZHospAlle_a3",
            "Belege Betten (ges.) im 3-Tage-Schnitt",
            target=2500,
        )
        print_predictions(
            faelle_at,
            "FZICU_a3",
            "Belegte Betten (ICU) i.3TS. -- orange Zone",
            target=600,
            days=14,
        )
        print_predictions(
            faelle_at,
            "FZICU_a3",
            "Belegte Betten (ICU) i.3TS. -- gelbe Zone",
            target=200,
            days=7,
        )

    if args.only_predict:
        return

    plot_detail(faelle_at, "Pandemieverlauf AT").savefig("timeline-at.png")

    fig, axs = plt.subplots(
        3,
        3,
        sharex=True,
        sharey=True,
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
        num="Pandemieverlauf/Bundesland",
        figsize=a2_dims,
        dpi=200,
        tight_layout=True,
    )

    ax: plt.Axes
    for idx, bundesland, ax in zip(range(9), BUNDESLAND_BY_ID.values(), axs.flat):
        y = idx % 3
        ax.set_title(bundesland)
        ax2 = plot_detail_ax(faelle[faelle["Bundesland"] == bundesland], ax)
        if ax.get_legend():
            ax.get_legend().remove()
        if ax2.get_legend():
            ax2.get_legend().remove()
        ax.label_outer()
        if y != 2:
            ax2.set_yticklabels([])
            ax2.set_ylabel(None)
        if y != 0:
            ax.set_ylabel(None)
        ax.set_ylim(top=4, bottom=0.4)
        ax2.set_ylim(bottom=0.1, top=2500)
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="upper left")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.autofmt_xdate(which="major")
    fig.savefig("timeline-bundesland.png", pad_inches=0, bbox_inches="tight")
    # fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1)
    plt.close(fig)
    del fig
    plt.show()


def set_percent_opts(ax: plt.Axes, freq=None, decimals=0, xmax=1):
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=xmax, decimals=decimals))
    if freq is not None:
        ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(freq))


def set_date_opts(ax: plt.Axes, xs: pd.Series = None, showyear=None, showday=None, week_always=False, autoyear=False):
    n = (
        ax.get_xlim()[1] - ax.get_xlim()[0]
        if xs is None
        else (matplotlib.dates.date2num(xs.max()) - matplotlib.dates.date2num(xs.min()))
    )
    # print("n=",n)

    weekly_mode = n <= 120
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY) if weekly_mode else matplotlib.dates.MonthLocator()
    )
    if n >= 330 and showyear is None:
        showyear = True
    if showyear and autoyear:
        # print("autoyear!")
        ax.figure.autofmt_xdate()
    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%d.%m" + (".%y" if showyear else ""))
        if weekly_mode
        else matplotlib.dates.DateFormatter("%b %y" if showyear else "%b")
    )
    if not weekly_mode and week_always:
        ax.xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY))
        ax.grid(which="major", axis="x", lw=1, zorder=-100)
        ax.grid(which="minor", axis="x", lw=0.5, zorder=1)
    if showday or (n <= 43 and showday is None):
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
        ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%a"))
        ax.xaxis.set_tick_params(which="major", pad=15)
        ax.xaxis.set_tick_params(which="minor", labelsize="x-small")
    ax.tick_params(axis="x", bottom=True, labelbottom=True)
    margin = 0.9
    if xs is not None and isinstance(xs.min(), date):
        ax.set_xlim(
            left=matplotlib.dates.date2num(xs.min()) - 0.5,
            right=matplotlib.dates.date2num(xs.max()) + 0.5,
        )


def plt_cat_dists(
    data_cat: pd.DataFrame,
    data_agg: pd.DataFrame,
    bundesland: str,
    catcol: str,
    stamp: str,
    *,
    datefrom=None,
    shortrange=40,
):
    ncats = data_cat[catcol].nunique()
    hugemode = ncats > 11
    if datefrom is None:
        pltagg = data_agg
    else:
        pltagg = data_agg[data_agg["Datum"] >= datefrom]

    pltdates = pltagg["Datum"]

    def plt_cumpct(title, data_agg, data_cat, labelend=True, continous=True):
        fig = plt.Figure()
        ax = fig.subplots()
        ax.set_title(f"{bundesland}: {title} je {catcol}{stamp}")
        styleargs = {} if continous else dict(ds="steps-mid", solid_joinstyle="miter")
        sns.lineplot(
            ax=ax,
            data=data_agg,
            x="Datum",
            y="AnzahlFaelleSumProEW",
            color="grey",
            ls="--",
            label="Gesamt: " + format(data_agg.iloc[-1]["AnzahlFaelleSumProEW"], ".0%"),
            **styleargs,
        )
        sns.lineplot(
            ax=ax,
            data=data_cat,
            x="Datum",
            y="AnzahlFaelleSumProEW",
            hue=catcol,
            size="AnzEinwohner",
            **styleargs,
        )
        if labelend:
            labelend2(ax, data_cat, "AnzahlFaelleSumProEW", cats=catcol, shorten=lambda c: c)
        artists, labels = ax.get_legend_handles_labels()
        bev_idx = labels.index("AnzEinwohner")
        artists, labels = artists[:bev_idx], labels[:bev_idx]
        bev_index = labels.index(catcol)
        del artists[bev_index], labels[bev_index]
        for i, label in enumerate(labels):
            vals = data_cat[data_cat[catcol] == label]
            if len(vals) <= 0:
                continue
            labels[i] += f": {vals.loc[vals.last_valid_index()]['AnzahlFaelleSumProEW']:.0%}"
        # ax.set_legend_handles_labels(artists, labels)
        ax.legend(artists, labels)
        ax.set_ylabel(title)
        # ax.legend(loc="upper left")
        set_percent_opts(ax, decimals=1)

        ax.set_ylim(bottom=0)
        set_date_opts(ax, pltdates, showyear=True)
        ax.set_xlim(left=max(date(2020, 9, 1), pltdates.iloc[0].date()))
        fig.autofmt_xdate()
        ax.tick_params(bottom=False)
        stampit(fig)
        return ax

    if False:
        display(plt_cumpct("Kumulative Inzidenz", data_agg, data_cat).figure)

    def norm100agg(df):
        df = df.copy()
        df["AnzahlFaelleSumProEW"] = df["AnzahlFaelleSum"] / df["AnzahlFaelleSum"].iloc[-1]
        return df

    def norm100(df):
        df = df.copy()
        for cat in df[catcol].unique():
            mask = df[catcol] == cat
            df.loc[mask, "AnzahlFaelleSumProEW"] = (
                df.loc[mask, "AnzahlFaelleSum"] / df.loc[mask, "AnzahlFaelleSum"].iloc[-1]
            )
        return df

    if False:
        ax = plt_cumpct(
            "Zeitliche Verteilung der Infektionen",
            norm100agg(data_agg),
            norm100(data_cat),
            labelend=False,
        )
        ax.set_ylabel("Kumulative Inzidenz in Prozent der Infizierten")
        set_percent_opts(ax, decimals=0, freq=0.1)
        ax.set_ylim(0, 1.01)
        display(ax.figure)

        data_agg0 = data_agg.copy()
        data_agg0["AnzahlFaelleSum"] = data_agg0["AnzahlTotSum"]
        data_cat0 = data_cat.copy()
        data_cat0["AnzahlFaelleSum"] = data_cat0["AnzahlTotSum"]

        ax = plt_cumpct(
            "Zeitliche Verteilung der Toten",
            norm100agg(data_agg0),
            norm100(data_cat0),
            labelend=False,
            continous=False,
        )
        ax.set_ylabel("Verstorbene in Prozent der bisher Verstorbenen")
        set_percent_opts(ax, decimals=0, freq=0.1)
        ax.set_ylim(-0.005, 1.005)
        display(ax.figure)

    fig = plt.Figure(figsize=(16, 9))
    ax = fig.subplots()
    ax.set_title(f"{bundesland}: Inzidenz im Gesamtverlauf je {catcol}{stamp}")
    # ax.set_ylim(bottom=0, top=agd_sums_at['inz'].max() * 1.05)
    # ax.set_xlim(left=agd_sums_at['Datum'].iloc[0], right=agd_sums_at['Datum'].iloc[-1])
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=len(pltdates),
        col="inz",
        weightcol="AnzEinwohner",
        catcol=catcol,
    )
    set_date_opts(ax, pltdates)
    ax.set_ylim(bottom=0)
    ax.legend(title=catcol, loc="upper left")
    stampit(fig)
    display(fig)

    fig = plt.Figure(figsize=(15, 4.95))
    ax = fig.subplots()
    # ax.set_title(f"{bundesland}: Inzidenz im Gesamtverlauf nach Altersgruppe" + stamp)
    # ax.set_ylim(bottom=0, top=agd_sums_at['inz'].max() * 1.05)
    # ax.set_xlim(left=agd_sums_at['Datum'].iloc[0], right=agd_sums_at['Datum'].iloc[-1])
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=len(pltdates),
        col="inz",
        weightcol="AnzEinwohner",
        color_args=dict(palette="flare_r"),
        catcol=catcol,
    )
    # ax.legend(title="Altersgruppe", loc="upper left")
    # ax.get_legend().remove()
    set_date_opts(ax, pltdates)
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    sns.despine(fig, bottom=True, left=True)
    display(fig)

    inzmin = max(0, pltagg["inz"].min() * 0.95)

    # with plt.rc_context({'axes.prop_cycle': cycler(color=sns.color_palette("rainbow", n_colors=10))}):
    fig = plt.Figure(figsize=(16, 9))
    ax = fig.subplots()
    fig.suptitle(f"{bundesland}: Anteil Fälle pro Woche je {catcol}{stamp}")
    ax.set_axisbelow(False)
    ax.set_ylabel("Anteil an neuen Fällen")
    ax2 = ax.twinx()
    ax2.set_ylabel("Gesamtinzidenz")
    # ax2.set_yscale("log", base=2)
    ax2.grid(False)
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.set_yticks(INZ_TICKS)
    ax2.plot(pltagg["Datum"], pltagg["inz"], color="k", label="Inzidenz")
    ax2.set_ylim(bottom=inzmin)
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=len(pltdates),
        col="AnzahlFaelle7Tage",
        mode="ratio",
        catcol=catcol,
    )
    set_percent_opts(ax, freq=0.1)
    set_date_opts(ax, pltdates)
    ncol = 7 if hugemode else 6
    fontsize = 9 if hugemode else None
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=fontsize,
        ncol=ncol,
        frameon=False,
    )
    stampit(fig)
    display(fig)

    fig = plt.Figure(figsize=(16, 9))
    ax = fig.subplots()
    fig.suptitle(f"{bundesland}: Inzidenzverhältnisse je {catcol}{stamp}")
    ax.set_axisbelow(False)
    ax.set_ylabel("Anteil an Inzidenzsumme")
    ax2 = ax.twinx()
    ax2.set_ylabel("Gesamtinzidenz")
    # ax2.set_yscale("log", base=2)
    ax2.grid(False)
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.set_yticks(INZ_TICKS)
    ax2.plot(pltagg["Datum"], pltagg["inz"], color="k", label="Inzidenz")
    ax2.set_ylim(bottom=inzmin)
    set_percent_opts(ax, decimals=1 if ncats != 10 else 0)
    ax.get_yaxis().set_major_locator(matplotlib.ticker.LinearLocator(ncats + 1))
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=len(pltdates),
        col="inz",
        mode="ratio",
        catcol=catcol,
    )
    set_date_opts(ax, pltdates)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
    )
    stampit(fig)
    display(fig)

    if False:
        fig = plt.Figure()
        ax = fig.subplots()
        cov.plt_age_sums_ax(ax, agd_sums_at, shortrange, "AnzahlFaelle")
        ax.set_title(f"{bundesland}: Fälle der letzten Tage je {catcol}{stamp}", fontsize=10)
        cov.ag_def_legend(fig)
        display(fig)

    def add_casenums(ax: plt.Axes):
        ax2 = ax.twinx()
        ax2.plot(
            data_agg["Datum"].iloc[-shortrange:],
            data_agg["AnzahlFaelle"].iloc[-shortrange:],
            label="Neue Fälle",
            drawstyle="steps-mid",
            # marker="_",
            # alpha=0.5,
            color="k",
            linewidth=1,
        )
        ax2.plot(
            data_agg["Datum"].iloc[-shortrange:],
            data_agg["AnzahlFaelle"].rolling(7).mean().iloc[-shortrange:],
            label="Wochenschnitt",
            # drawstyle="steps-mid",
            # marker="_",
            # alpha=0.5,
            ls="--",
            color=(0.25, 0.25, 0.25),
            linewidth=1,
        )
        ax2.set_ylim(
            bottom=0,
            top=np.ceil((data_agg["AnzahlFaelle"].iloc[-shortrange:].max() + 100) / 100.0) * 100,
        )
        ax2.set_ylabel("Anzahl neuer Fälle")
        ax2.grid(False)
        # ax.set_axisbelow(False)
        # ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax2.set_yticks(INZ_TICKS)
        ax2.get_yaxis().set_major_locator(matplotlib.ticker.LinearLocator(11))
        ax2.tick_params(axis="x", bottom=False, labelbottom=False)

    fig = plt.Figure(figsize=(10, 5))
    fig.suptitle(f"{bundesland}: Fallanteil pro Tag je {catcol}{stamp}", y=1.01, size=11)
    ax = fig.subplots()
    ax.set_ylabel("Anteil an neuen Fällen")
    add_casenums(ax)
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=shortrange,
        col="AnzahlFaelle",
        mode="ratio",
        catcol=catcol,
    )
    # set_date_opts(ax, pltdates)
    set_percent_opts(ax, freq=0.1)
    ax.set_ylim(bottom=0, top=1)
    # ax.set_title("Anteil neuer Fälle pro Tag nach Altersgruppe")
    # ax.legend(loc="center left", bbox_to_anchor=(-0.05, 0.5))
    fontsize2 = 7 if hugemode else 9
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        fontsize=fontsize2,
        frameon=False,
    )
    ax.tick_params(top=True)
    stampit(fig)
    display(fig)

    fig = plt.Figure(figsize=(10, 5))
    fig.suptitle(f"{bundesland}: Fälle pro Tag je {catcol}{stamp}", y=1.01, size=11)
    ax = fig.subplots()
    ax.set_ylabel("Anzahl neuer Fälle")
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=shortrange,
        col="AnzahlFaelle",
        # mode="ratio",
        catcol=catcol,
    )
    # set_date_opts(ax, pltdates)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        fontsize=fontsize2,
        frameon=False,
    )
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which="minor", axis="y", lw=0.5)
    stampit(fig)
    display(fig)

    fig = plt.Figure(figsize=(10, 5))
    fig.suptitle(f"{bundesland}: Verhältnisse Tagesinzidenz je {catcol}{stamp}", y=1.01, size=11)
    ax = fig.subplots()
    ax.set_ylabel("Anteil an Inzidenzsumme")
    add_casenums(ax)
    plt_cat_sums_ax(
        ax,
        data_cat,
        data_agg,
        n_days=shortrange,
        col="AnzahlFaelleProEW",
        mode="ratio",
        catcol=catcol,
    )
    # set_date_opts(ax, pltdates)
    set_percent_opts(ax, decimals=1 if ncats != 10 else 0)
    ax.get_yaxis().set_major_locator(matplotlib.ticker.LinearLocator(ncats + 1))
    ax.set_ylim(bottom=0, top=1)
    ax.tick_params(top=True)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        fontsize=fontsize2,
        frameon=False,
    )
    stampit(fig)
    display(fig)


DS_AGES = "AGES"
DS_BMG = "BMSGPK"
DS_STAT = "statistik.at"
DS_BOTH = ", ".join((DS_AGES, DS_BMG))


def stampit(fig: plt.Figure, dsource: str = DS_AGES):
    # return
    fig.text(
        0,
        0.1,
        f"Rohdatenquelle: {dsource} | Darstellung & Bearbeitung: @zeitferne",
        va="top",
        in_layout=False,
        size=6,
        color="grey",
        transform=fig.dpi_scale_trans,
    )


if __name__ == "__main__":
    main()
