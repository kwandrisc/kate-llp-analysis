import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from math import *
import pathlib
import pickle
import argparse
from tqdm import tqdm
import math
from scipy import optimize
from collections import OrderedDict
import pandas as pd

import pyLCIO
from pyLCIO import UTIL, EVENT
import ROOT

dir = "/ospool/uc-shared/project/futurecolliders/wandriscok/reco/nu_background/"
windows = ["loose"]
#bib_options = ["10_bib", "bib"]
bib_options = ["bib"]
#windows = ["loose", "medium", "tight"]
windows_ordered = ["tight", "medium", "loose"]
CACHE = pathlib.Path("cache/cut_prop_loose.pkl")
file_ranges = {
    "10_bib": (0, 2500),
    "bib": (0, 10)
}
Bfield = 3.57
speedoflight = 299792458/1000000  # mm/ns

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap"
}
bib_name = {
    "10_bib": "10% BIB",
    "bib": "100% BIB"
}
chi2_cut = 3
track_req_names = ["vb", "ib", "ob"]

parser = argparse.ArgumentParser()
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()
rebuild = args.rebuild

def build_rel_nav(event):
    nav = {
        "VXDBarrel": UTIL.LCRelationNavigator(event.getCollection("VXDBarrelHitsRelations")),
        "VXDEndcap": UTIL.LCRelationNavigator(event.getCollection("VXDEndcapHitsRelations")),
        "ITBarrel" : UTIL.LCRelationNavigator(event.getCollection("ITBarrelHitsRelations")),
        "ITEndcap" : UTIL.LCRelationNavigator(event.getCollection("ITEndcapHitsRelations")),
        "OTBarrel" : UTIL.LCRelationNavigator(event.getCollection("OTBarrelHitsRelations")),
        "OTEndcap" : UTIL.LCRelationNavigator(event.getCollection("OTEndcapHitsRelations")),
        }
    enc = event.getCollection("ITBarrelHits").getParameters().getStringVal(pyLCIO.EVENT.LCIO.CellIDEncoding)
    nav["_ENCODING"] = enc
    nav["_DECODER"] = UTIL.BitField64(enc)
    return nav

def linearfunc(p, x):
    # p[0] = velocity [mm/ns], p[1] = intercept [mm]
    return p[0] * x + p[1]

def residual(p, function_type, times, pos, spatial_unc):
    # weighted residuals
    return (function_type(p, times) - pos) / spatial_unc

guess_velo = 299.8

def fit_rms(p, function_type, times, pos, spatial_unc):
    x = np.asarray(times, dtype=float)
    y = np.asarray(pos, dtype=float)
    s = np.asarray(spatial_unc, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    x, y, s = x[m], y[m], s[m]
    if x.size == 0:
        return np.nan, np.nan

    yhat = function_type(p, x)
    r = yhat - y                
    rms_unw = float(np.sqrt(np.mean(r*r)))
    rw = r / s                  
    rms_w = float(np.sqrt(np.mean(rw*rw)))
    return rms_unw, rms_w

def time_rms_from_fit(v, t, r, time_unc, b=0.0):
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    st = np.asarray(time_unc, float)

    m = np.isfinite(t) & np.isfinite(r) & np.isfinite(st) & (st > 0)
    t, r, st = t[m], r[m], st[m]
    if t.size < 3 or (not np.isfinite(v)) or abs(v) < 1e-12:
        return np.nan, np.nan

    t_pred = (r - b) / v
    dt = t - t_pred

    uw_rms_t = float(np.sqrt(np.mean(dt * dt)))          
    w_rms_t = float(np.sqrt(np.mean((dt / st) ** 2))) 
    return uw_rms_t, w_rms_t

def reco_velo(function_type, times, pos, spatial_unc):
    x = np.asarray(times, dtype=float)
    y = np.asarray(pos, dtype=float)
    s = np.asarray(spatial_unc, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    x, y, s = x[m], y[m], s[m]

    if x.size < 3 or np.allclose(x, x.mean()):
        return np.nan, np.nan, np.nan, np.nan

    p0 = np.array([guess_velo, 0.0])

    fit = optimize.least_squares(
        residual, p0,
        args=(function_type, x, y, s),
        jac='2-point'
    )
    p = fit.x  
    try:
        J = fit.jac
        dof = max(1, x.size - p.size)
        chi2 = np.sum(((function_type(p, x) - y) / s) ** 2)
        sigma2 = chi2 / dof
        cov = np.linalg.inv(J.T @ J) * sigma2
        v_err = float(np.sqrt(cov[0, 0]))
        rms_unw, rms_w = fit_rms(p, function_type, x, y, s)

    except Exception:
        v_err = np.nan
        rms_unw, rms_w = np.nan, np.nan

    return float(p[0]), v_err, rms_unw, rms_w

def linearfunc_no_intercept(v, x):
    return v * x
def residual_no_intercept(v, times, pos, spatial_unc, time_unc):
    vv = float(np.atleast_1d(v)[0])
    s_eff = np.sqrt(np.asarray(spatial_unc, float)**2 + (vv * np.asarray(time_unc, float))**2)
    return (linearfunc_no_intercept(vv, times) - pos) / s_eff

def reco_velo_no_intercept(times, pos, spatial_unc, time_unc):
    x = np.asarray(times, dtype=float)
    y = np.asarray(pos, dtype=float)
    s = np.asarray(spatial_unc, dtype=float)
    st = np.asarray(time_unc, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(st) & (st > 0)
    x, y, s, st = x[m], y[m], s[m], st[m]

    if x.size < 3 or np.allclose(x, x.mean()):
        return np.nan, np.nan, np.nan, np.nan

    v0 = np.array([guess_velo])

    def residual0(v, times, pos, spatial_unc):
        vv = float(np.atleast_1d(v)[0])
        return (vv * times - pos) / spatial_unc

    fit = optimize.least_squares(
        residual0,
        v0,
        args=(x, y, s),
        jac="2-point"
    )

    v = float(fit.x[0])

    uw_rms_t, w_rms_t = time_rms_from_fit(v, x, y, st, b=0.0)

    try:
        r = (v * x - y)
        J = fit.jac
        dof = max(1, x.size - 1)
        chi2 = np.sum((r / s) ** 2)
        sigma2 = chi2 / dof
        cov = np.linalg.inv(J.T @ J) * sigma2
        v_err = float(np.sqrt(cov[0, 0]))
    except Exception:
        v_err = np.nan

    return v, v_err, uw_rms_t, w_rms_t


stats = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        stats = pickle.load(f)

if stats is None:
    stats = {
        window: {
            req: {
                "10_bib": {
                    "eta_nocut": [],
                    "rchi2_nocut": [],
                    "w_rms": []
                },
                "bib": {
                    "eta_nocut": [],
                    "rchi2_nocut": [],
                    "w_rms": []
                }
            }
            for req in track_req_names
        }
        for window in windows
    }
    
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        print(f"Analyzing {window} window...")
        for option in bib_options:
            total_track = 0
            track_vb = 0
            track_ib = 0
            track_ob = 0
            over_c = 0
            print(f"Analyzing {option}...")
            start, stop = file_ranges[option]
            for ifile in tqdm(range(start, stop)):
                file_name = f"nu_background_reco{ifile}.slcio"
                file_path = os.path.join(dir, window, option, file_name)

                if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
                    print(f"Skipping bad file: reco{ifile}")
                    continue

                try:
                    reader.open(file_path)
                except Exception as e:
                    print(f"LCIO failed to open reco{ifile}")
                    continue

                try:
                    for event in reader:
                        all_collections = event.getCollectionNames() 
                        track_collection = event.getCollection("SiTracks") if "SiTracks" in all_collections else None 
                        if not track_collection:
                            print("issue 1")
                            continue
                        test_hit_coll = event.getCollection("VXDBarrelHits")
                        if test_hit_coll is None:
                            continue
                        encoding = test_hit_coll.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                        decoder = UTIL.BitField64(encoding)
                        
                        for itrack, track in enumerate(track_collection):
                            total_track += 1

                            chi2 = track.getChi2()
                            ndf = track.getNdf()
                            reduced_chi2 = chi2 / ndf
                            track_hits = track.getTrackerHits()
                            
                            reco_pT = 0.3 * Bfield / fabs(track.getOmega() * 1000.)

                            vb_hits = 0
                            ib_hits = 0
                            ob_hits = 0
                            
                            track_times = []
                            track_pos = []
                            spatial_unc = []
                            time_unc = []

                            for hit in track_hits:
                                decoder.setValue(int(hit.getCellID0()))
                                system = decoder["system"].value()
                                layer = decoder["layer"].value()
                                if system in (1,2):
                                    vb_hits += 0.5
                                    spatial_unc.append(0.005)
                                    time_unc.append(0.03)
                                elif system in (3,4):
                                    ib_hits += 1
                                    spatial_unc.append(0.007)
                                    time_unc.append(0.06)
                                elif system in (5,6):
                                    ob_hits += 1
                                    spatial_unc.append(0.007)
                                    time_unc.append(0.06)

                                hit_time = hit.getTime()
                                x = hit.getPosition()[0]
                                y = hit.getPosition()[1]
                                z = hit.getPosition()[2]
                                hit_pos = np.sqrt(x**2 + y**2 + z**2)
                                tof = hit_pos/speedoflight

                                resolution = 0.03
                                if system > 2:
                                    resolution = 0.06

                                corrected_t = hit.getTime() + tof

                                track_times.append(corrected_t)
                                track_pos.append(hit_pos)

                            v_fit, v_err, uw_rms, w_rms = reco_velo_no_intercept(track_times, track_pos, spatial_unc, time_unc)

                            tan_lambda = track.getTanLambda()
                            eta = np.arcsinh(tan_lambda)
                            
                            total_hits = vb_hits + ib_hits + ob_hits

                            if vb_hits >= 3:
                                stats[window]["vb"][option]["eta_nocut"].append(eta)
                                stats[window]["vb"][option]["rchi2_nocut"].append(reduced_chi2)
                                stats[window]["vb"][option]["w_rms"].append(w_rms)
                                track_vb += 1
                            if vb_hits >= 3 and ib_hits >= 2:
                                stats[window]["ib"][option]["eta_nocut"].append(eta)
                                stats[window]["ib"][option]["rchi2_nocut"].append(reduced_chi2)
                                stats[window]["ib"][option]["w_rms"].append(w_rms)
                                track_ib += 1
                            if vb_hits >= 3 and ib_hits >= 2 and ob_hits >= 2:
                                stats[window]["ob"][option]["eta_nocut"].append(eta)
                                stats[window]["ob"][option]["rchi2_nocut"].append(reduced_chi2)
                                stats[window]["ob"][option]["w_rms"].append(w_rms)
                                track_ob += 1

                except Exception as e:
                    print(f"Crash while reading {file_path}: {e}")
                finally:
                    reader.close()

            print(f"Finished {window} / {option}")

            print(f"{window} window, {option} option, {file_ranges[option][1]} files stats:")
            print(f"Number of total tracks: {total_track}")
    

    #print(stats)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Writing cache to {CACHE}")
    print("Saved cache successfully.")

title_map = {
        "eta_nocut": "Eta values no cuts",
        "rchi2_nocut": r"Reduced $\chi^2$ values no cuts",
        "w_rms": "Weighted RMS For Residuals"
    }

xlabel_map = {
        "eta_nocut": "Eta",
        "rchi2_nocut": r"Reduced $\chi^2$", 
        "w_rms": "Weighted RMS For Residuals"
    }

bib_map = {
    "10_bib": "10% bib",
    "bib": "100% bib"
}



def get_feature_arrays_req(feature, window, option):
    return [
        np.asarray(window_to_stats[window][window][req][option][feature], float)
        for req in track_req_names
    ]


def get_bins_and_array(arr, feature, x_lim):
    arr = arr[np.isfinite(arr)]

    if feature == "w_rms":
        xmin, xmax = x_lim
        bin_width = 0.5

        bins = np.arange(xmin, xmax + bin_width, bin_width)

        # overflow handling
        arr_plot = arr.copy()
        arr_plot[arr_plot > xmax] = xmax - 0.5 * bin_width

        return arr_plot, bins

    else:
        return arr, 30


def plot_feature_byreq(feature, window, option, x_lim=None):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5), constrained_layout=True)
    
    titles = [
        "$\geq$3 VB Hits",
        "$\geq$3 VB, $\geq$2 IB Hits",
        "$\geq$3 VB, $\geq$2 IB, $\geq$2 OB Hits"
    ]

    feature_arrays = get_feature_arrays_req(feature, window, option)

    for req, arr in zip(track_req_names, feature_arrays):
        arr_clean = arr[np.isfinite(arr)]
        # if arr_clean.size > 0:
        #     # print(f"{feature} ({req}) raw min:", np.min(arr_clean),
        #     #     "raw max:", np.max(arr_clean))
    
    if x_lim is not None:
        feature_arrays = [
            arr[(arr >= x_lim[0]) & (arr <= x_lim[1])]
            for arr in feature_arrays
        ]
    
    for ax, arr, title in zip(axes, feature_arrays, titles):
        #print(arr)
        
        if arr.size != 0:
            weights = np.full(arr.size, 100.0 / arr.size)
        else:
            continue
        
        arr_plot, bins = get_bins_and_array(arr, feature, x_lim)

        if arr_plot.size != 0:
            weights = np.full(arr_plot.size, 100.0 / arr_plot.size)
        else:
            continue

        ax.hist(
            arr_plot,
            bins=bins,
            weights=weights,
            histtype="stepfilled",
            color="grey",
            alpha=0.30,
            edgecolor="black",
            linewidth=2.0,
            label="BIB background"
        )

        if feature == "eta_nocut":
            ax.axvline(x=0.8, color="r", linestyle="--", linewidth=2, label="Cut")
            ax.axvline(x=-0.8, color="r", linestyle="--", linewidth=2, label="_nolegend_")
        elif feature == "rchi2_nocut":
            ax.axvline(x=3, color="r", linestyle="--", linewidth=2, label="Cut")
        else:
            ax.axvline(x=1.6, color="r", linestyle="--", linewidth=2, label="Cut")

        ax.set_title(title, fontsize=18)
        ax.set_xlabel(xlabel_map[feature], fontsize=18)
        ax.tick_params(axis="both", length=5, width=1, labelsize=15)
        ax.legend(fontsize=13, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.2)

        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.text(
            0.02, 0.98,
            "Muon Collider",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            style="italic",
        )

        ax.text(
            0.02, 0.91,
            "MuColl_v1",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=12
        )

    axes[0].set_ylabel("Normalized Counts (%)", fontsize=18)

    fig.suptitle(
        f"{title_map.get(feature, feature)} | {window} window, {bib_name[option]}",
        fontsize=18
    )

    pdf.savefig(fig)
    plt.close(fig)

LOOSE_CACHE = pathlib.Path("cache/cut_prop_loose.pkl")
MED_CACHE = pathlib.Path("cache/cut_prop_medium.pkl")
TIGHT_CACHE = pathlib.Path("cache/cut_prop_tight.pkl")

with LOOSE_CACHE.open("rb") as f:
    loose_stats = pickle.load(f)

with MED_CACHE.open("rb") as f:
    med_stats = pickle.load(f)

with TIGHT_CACHE.open("rb") as f:
    tight_stats = pickle.load(f)


window_to_stats = {
    "tight": tight_stats,
    "medium": med_stats,
    "loose": loose_stats
}

def get_feature_arrays_window(feature, req, option):
    return [
        np.asarray(window_to_stats[window][window][req][option][feature], float)
        for window in windows_ordered
    ]


def plot_feature_bywindow(feature, req, option, x_lim=None):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5), constrained_layout=True)
    
    titles = [
        "Tight Window",
        "Medium Window",
        "Loose Window"
    ]

    feature_arrays = get_feature_arrays_window(feature, req, option)

    for window, arr in zip(windows_ordered, feature_arrays):
        arr_clean = arr[np.isfinite(arr)]
        # if arr_clean.size > 0:
        #     print(f"{feature} ({window}) raw min:", np.min(arr_clean),
        #         "raw max:", np.max(arr_clean))
    
    if x_lim is not None:
        feature_arrays = [
            arr[(arr >= x_lim[0]) & (arr <= x_lim[1])]
            for arr in feature_arrays
        ]
    
    for ax, arr, title, window in zip(axes, feature_arrays, titles, windows_ordered):
        #print(arr)
        
        if arr.size != 0:
            weights = np.full(arr.size, 100.0 / arr.size)
        else:
            continue
        
        arr_plot, bins = get_bins_and_array(arr, feature, x_lim)

        if arr_plot.size != 0:
            weights = np.full(arr_plot.size, 100.0 / arr_plot.size)
        else:
            continue

        if feature == "w_rms":
            if window == "tight":
                bins = 20
            elif window == "medium":
                bins = 25
            elif window == "loose":
                bins = 40

        ax.hist(
            arr_plot,
            bins=bins,
            weights=weights,
            histtype="stepfilled",
            color="grey",
            alpha=0.30,
            edgecolor="black",
            linewidth=2.0,
            label="BIB background"
        )

        if feature == "eta_nocut":
            ax.axvline(x=0.8, color="r", linestyle="--", linewidth=2, label="Cut")
            ax.axvline(x=-0.8, color="r", linestyle="--", linewidth=2, label="_nolegend_")
        elif feature == "rchi2_nocut":
            ax.axvline(x=3, color="r", linestyle="--", linewidth=2, label="Cut")
        else:
            ax.axvline(x=1.6, color="r", linestyle="--", linewidth=2, label="Cut")

        ax.set_title(title, fontsize=18)
        ax.set_xlabel(xlabel_map[feature], fontsize=18)
        ax.tick_params(axis="both", length=5, width=1, labelsize=15)
        ax.legend(fontsize=13, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.2)

        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.text(
            0.02, 0.98,
            "Muon Collider",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            style="italic",
        )

        ax.text(
            0.02, 0.91,
            "MuColl_v1",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=12
        )

    axes[0].set_ylabel("Normalized Counts (%)", fontsize=18)

    fig.suptitle(
        f"{title_map.get(feature, feature)} | req: {req}, {bib_name[option]}",
        fontsize=20
    )

    pdf.savefig(fig)
    plt.close(fig)



with PdfPages('pdf/eta_chi2.pdf') as pdf:
    features = ["eta_nocut", "rchi2_nocut", "w_rms"]
    for option in bib_options:
        for window in windows_ordered:
            plot_feature_byreq("eta_nocut", window, option, (-3, 3))
            plot_feature_byreq("rchi2_nocut", window, option, (0, 5))
            plot_feature_byreq("w_rms", window, option, (0, 15))

        for req in track_req_names:
            plot_feature_bywindow("eta_nocut", req, option, (-3, 3))
            plot_feature_bywindow("rchi2_nocut", req, option, (0, 5))
            plot_feature_bywindow("w_rms", req, option, (0, 15))
    print(f"Saved plots to eta_chi2.pdf") 



