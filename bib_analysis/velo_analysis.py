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
bib_options = ["10_bib"]
#windows = ["loose", "tight"]
CACHE = pathlib.Path("cache/bib_velo_timeresiduals.pkl")
SAVE_EVERY = 50
file_ranges = {
    "10_bib": (0, 2500),
    "bib": (4, 10)
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
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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

    t_pred = y / v
    dt = x - t_pred

    pulls = dt / st
    max_pull = np.max(np.abs(pulls))
    n_outliers = np.sum(np.abs(pulls) > 3)
    frac_outliers = n_outliers / len(pulls)

    uw_rms_t = float(np.sqrt(np.mean(dt**2)))
    w_rms_t = float(np.sqrt(np.mean(pulls**2)))

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

    return v, v_err, uw_rms_t, w_rms_t, max_pull, n_outliers, frac_outliers, pulls

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
                    "velocity": [],
                    "w_rms": [],
                    "uw_rms": [],
                    "max_pull": [],
                    "n_outliers": [],
                    "frac_outliers": [],
                    "pulls": []
                },
                "bib": {
                    "velocity": [],
                    "w_rms": [],
                    "uw_rms": [],
                    "max_pull": [],
                    "n_outliers": [],
                    "frac_outliers": [],
                    "pulls": []
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
            track_pass_chi2 = 0
            track_pass_eta = 0
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

                            # reduced chi2 cut
                            if reduced_chi2 > chi2_cut:
                                continue

                            track_pass_chi2 += 1

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

                            v_fit, v_err, uw_rms, w_rms, max_pull, n_outliers, frac_outliers, pulls = reco_velo_no_intercept(track_times, track_pos, spatial_unc, time_unc)

                            # making eta cut
                            tan_lambda = track.getTanLambda()
                            eta = np.arcsinh(tan_lambda)
                            if abs(eta) > 0.8:
                                continue
                            track_pass_eta += 1
                            
                            total_hits = vb_hits + ib_hits + ob_hits
                            

                            if vb_hits >= 3:
                                stats[window]["vb"][option]["velocity"].append(v_fit)
                                stats[window]["vb"][option]["w_rms"].append(w_rms)
                                stats[window]["vb"][option]["uw_rms"].append(uw_rms)
                                stats[window]["vb"][option]["max_pull"].append(max_pull)
                                stats[window]["vb"][option]["n_outliers"].append(n_outliers)
                                stats[window]["vb"][option]["frac_outliers"].append(frac_outliers)
                                stats[window]["vb"][option]["pulls"].append(pulls)
                                track_vb += 1
                            if vb_hits >= 3 and ib_hits >= 2:
                                stats[window]["ib"][option]["velocity"].append(v_fit)
                                stats[window]["ib"][option]["w_rms"].append(w_rms)
                                stats[window]["ib"][option]["uw_rms"].append(uw_rms)
                                stats[window]["ib"][option]["max_pull"].append(max_pull)
                                stats[window]["ib"][option]["n_outliers"].append(n_outliers)
                                stats[window]["ib"][option]["frac_outliers"].append(frac_outliers)
                                stats[window]["ib"][option]["pulls"].append(pulls)
                                track_ib += 1
                            if vb_hits >= 3 and ib_hits >= 2 and ob_hits >= 2:
                                stats[window]["ob"][option]["velocity"].append(v_fit)
                                stats[window]["ob"][option]["w_rms"].append(w_rms)
                                stats[window]["ob"][option]["uw_rms"].append(uw_rms)
                                stats[window]["ob"][option]["max_pull"].append(max_pull)
                                stats[window]["ob"][option]["n_outliers"].append(n_outliers)
                                stats[window]["ob"][option]["frac_outliers"].append(frac_outliers)
                                stats[window]["ob"][option]["pulls"].append(pulls)
                                track_ob += 1

                except Exception as e:
                    print(f"Crash while reading {file_path}: {e}")
                finally:
                    reader.close()

                if (ifile - start + 1) % SAVE_EVERY == 0:
                    with CACHE.open("wb") as f:
                        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Checkpoint saved at file {ifile}")

            print(f"Finished {window} / {option}")

            vb_percent = (track_vb / track_pass_eta) * 100
            ib_percent = (track_ib / track_pass_eta) * 100
            ob_percent = (track_ob / track_pass_eta) * 100
            overc_percent = (over_c / track_pass_eta) * 100

            print(f"{window} window, {option} option stats:")
            print(f"Number of total tracks: {total_track}")
            print(f"Number of tracks passing chi2: {track_pass_chi2}")
            print(f"Number of tracks passing eta: {track_pass_eta}")
            print(f"Vertex cut: {track_vb} / {track_pass_eta} -> {vb_percent:.2f}%")
            print(f"Inner cut: {track_ib} / {track_pass_eta} -> {ib_percent:.2f}%")
            print(f"Outer cut: {track_ob} / {track_pass_eta} -> {ob_percent:.2f}%")
    

    #print(stats)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Writing cache to {CACHE}")
    print("Saved cache successfully.")

title_map = {
        "velocity": "Reconstructed Velocity (intercept fit & spacial unc)",
        "w_rms": "RMS of Weighted Residuals (intercept fit & spacial unc)",
        "uw_rms": "RMS of Unweighted Residuals (intercept fit & spacial unc)",
        "max_pull": "Per-track Max Residual / Time Uncertainty Distribution",
        "n_outliers": "Per-track Number of >3 sigma hits Distribution",
        "frac_outliers": "Per-track Fraction of >3 sigma hits Distribution",
        "pulls": "Per-hit Residual / Time Uncertaintiy Distribution"
    }

xlabel_map = {
        "velocity": "Velocity [mm/ns]",
        "w_rms": "RMS of Weighted Residuals",
        "uw_rms": "RMS of Unweighted Residuals",
        "max_pull": "Max Residual / Time Uncertainties",
        "n_outliers": "Number of >3 sigma hits",
        "frac_outliers": "Fraction of >3 sigma hits",
        "pulls": "Residual / time uncertainty"
    }

bib_map = {
    "10_bib": "10% bib",
    "bib": "100% bib"
}


def get_feature_arrays(feature, window, option):
    arrays = []
    for req in track_req_names:
        data = stats[window][req][option][feature]

        if feature == "pulls":
            flat = np.concatenate(data) if len(data) > 0 else np.array([])
            arrays.append(np.asarray(flat, float))
        else:
            arrays.append(np.asarray(data, float))

    
    for req, arr in zip(track_req_names, arrays):

        if req == "ob":
            print("--------------------------")
            print(f"Feature: {feature}, {req}")

            arr = arr[np.isfinite(arr)]

            if arr.size > 0:
                min_arr = np.min(arr)
                max_arr = np.max(arr)

            print(f"Min value: {np.round(min_arr, 2)}, Max value: {np.round(max_arr, 2)}")

    return arrays


def plot_feature(feature, window, option, x_lim=None):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5))
    
    titles = [
        "$\geq$3 VB Hits",
        "$\geq$3 VB, $\geq$2 IB Hits",
        "$\geq$3 VB, $\geq$2 IB, $\geq$2 OB Hits"
    ]

    feature_arrays = get_feature_arrays(feature, window, option)

    ax_dict = {
        1: "vb",
        2: "ib",
        3: "ob"
    }

    for ax, arr, title in zip(axes, feature_arrays, titles):
        #print(arr)
        
        arr = arr[np.isfinite(arr)]

        # min = arr.min()
        # max = arr.max()
        # print(f"Min value for {feature} - {ax_dict[ax]} barrel is {min}")
        # print(f"Max value for {feature} - {ax_dict[ax]} barrel is {max}")

        if x_lim is not None:
            arr = arr[(arr >= x_lim[0]) & (arr <= x_lim[1])]

        if arr.size != 0:
            weights = np.full(arr.size, 100.0 / arr.size)
        else:
            continue
        if feature == "hits":
            bins = np.arange(np.min(arr) - 0.5, np.max(arr) + 1.5, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            bins = 50

        ax.hist(arr, bins=bins, weights=weights, histtype="step", 
                color="grey", fill=True, label="BIB background", 
                alpha=0.30, edgecolor="black", linewidth=2.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel_map[feature], fontsize=12)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.2)

        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.text(
            0.02, 0.98,
            "Muon Collider",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            style="italic",
        )

        ax.text(
            0.02, 0.91,
            "MuColl_v1",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=11
        )

    axes[0].set_ylabel("Normalized Counts (%)", fontsize=12)

    fig.suptitle(f"{title_map.get(feature, feature)}  |  bkg: {window} window, {bib_name[option]}")

    # ax.tick_params(axis="both", which="major", labelsize=10, length=6, width=1.5)
    # ax.tick_params(axis="both", which="minor", labelsize=10, length=4, width=1.0)

    
    # ax.text(
    #     0.02, 0.90,
    #     f"BIB, {option}, {window}",
    #     ha="left", va="top",
    #     transform=ax.transAxes,
    #     fontsize=15
    # ) 
        
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

pdf_path = "/scratch/wandriscok/kate_mucoll_scripts/bib_analysis/pdf/velo.pdf"

with PdfPages(pdf_path) as pdf:
    features = ["velocity", "w_rms", "uw_rms", "max_pull", "n_outliers", "frac_outliers", "pulls"]
    for window in windows:
        for option in bib_options:
            plot_feature("velocity", window, option, (-100, 400))
            plot_feature("w_rms", window, option, x_lim=None)
            plot_feature("uw_rms", window, option, x_lim=None)
            plot_feature("max_pull", window, option, (0, 50))
            plot_feature("n_outliers", window, option, x_lim=None)
            plot_feature("frac_outliers", window, option, x_lim=None)
            plot_feature("pulls", window, option, (-40, 40))
    print(f"Saved plots to {pdf_path}") 

