import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

loose_dir = "/ospool/uc-shared/project/futurecolliders/miralittmann/reco/reder_timing/loose4/seeding_10GeV/nobib"
option = ["nobib"]
windows = ["loose"]
samples = ["1000_30", "1500_30", "2000_30", "2500_30", "3000_30", "3500_30", "4000_30", "4500_30"]
track_req_names = ["vb", "ib", "ob"]

stau_ids = [1000015, -1000015, 2000015, -2000015]
CACHE = pathlib.Path("cache/stau_velo_timeresiduals.pkl")
BIB_CACHE = pathlib.Path("/scratch/wandriscok/kate_mucoll_scripts/bib_analysis/cache/bib_velo_timeresiduals.pkl")
file_ranges = {
    "nobib": (0, 2500)
}
Bfield = 3.57
speedoflight = 299792458/1000000  # mm/ns
chi2_cut = 3

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap"
}
bib_name = {
    "10_bib": "10% BIB",
    "bib": "100% BIB",
    "nobib": "No BIB"
}
sample_to_mass = {
    "1000_30": 1.0,
    "1500_30": 1.5,
    "2000_30": 2.0,
    "2500_30": 2.5,
    "3000_30": 3.0,
    "3500_30": 3.5,
    "4000_30": 4.0,
    "4500_30": 4.5
}

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

guess_velo = 180

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

bib_stats = None
if BIB_CACHE.exists():
    print("Loading BIB cache...")
    with open(BIB_CACHE, "rb") as f:
        bib_stats = pickle.load(f)

print("Checking BIB stats...")
for req in track_req_names:
    for opt in ["10_bib"]:
        size = len(bib_stats["loose"][req][opt]["velocity"])
        print(req, opt, size)

stats = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        stats = pickle.load(f)

if stats is None:
    stats = {
        window: {
            req: {
                sample: {
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
                    },
                    "nobib": {
                        "velocity": [],
                        "w_rms": [],
                        "uw_rms": [],
                        "max_pull": [],
                        "n_outliers": [],
                        "frac_outliers": [],
                        "pulls": []
                    }
                } for sample in samples
            } for req in track_req_names
        } for window in windows
    }
    
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()

    # stau path
    # /ospool/uc-shared/project/futurecolliders/miralittmann/reco/reder_timing/loose4/seeding_10GeV/nobib/{mass}_{lifetime}/

    for window in windows:
        print(f"Analyzing {window} window...")
        for opt in option:
            total_track = 0
            track_pass_chi2 = 0
            track_pass_eta = 0
            track_vb = 0
            track_ib = 0
            track_ob = 0
            over_c = 0
            print(f"Analyzing {opt}...")
            for sample in samples:
                print(f"Beginning mass {sample_to_mass[sample]} TeV")
                
                start, stop = file_ranges[opt]
                for ifile in tqdm(range(start, stop)):
                    file_name = f"{sample}/{sample}_reco{ifile}.slcio"
                    file_path = os.path.join(loose_dir, file_name)

                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        print(f"Skipping bad file: reco{ifile}")
                        continue

                    try:
                        reader.open(file_path)
                    except Exception as e:
                        print(f"LCIO failed to open reco{ifile}")
                        continue

                    try:
                        for event in reader:
                            rel_nav = build_rel_nav(event)
                            all_collections = event.getCollectionNames() 
                            mcp_collection = event.getCollection("MCParticle") if "MCParticle" in all_collections else None
                            track_collection = event.getCollection("SiTracks") if "SiTracks" in all_collections else None 
                            track_relation_collection = event.getCollection("MCParticle_SiTracks") if "MCParticle_SiTracks" in all_collections else None
                            
                            if not (mcp_collection and track_collection and track_relation_collection):
                                print("Missing one of the required collections")
                                continue
                            
                            test_hit_coll = event.getCollection("VXDBarrelHits")
                            if test_hit_coll is None:
                                continue
                            encoding = test_hit_coll.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                            decoder = UTIL.BitField64(encoding)
                            
                            nav = UTIL.LCRelationNavigator(track_relation_collection)
                            
                            for itrack, track in enumerate(track_collection):
                                truth_stau = None
                                
                                track_mcps = nav.getRelatedFromObjects(track)
                                track_hits = track.getTrackerHits()

                                for mcp in track_mcps:
                                    pdg_id = mcp.getPDG()
                                    if abs(pdg_id) in stau_ids:
                                        truth_stau = mcp
                                        total_track += 1
                                        break
                                
                                if truth_stau is None:
                                    continue
                                
                                # only looking at stau track
                                if truth_stau is None:
                                    continue

                                chi2 = track.getChi2()
                                ndf = track.getNdf()
                                reduced_chi2 = chi2 / ndf

                                # reduced chi2 cut
                                if reduced_chi2 > chi2_cut:
                                    continue

                                track_pass_chi2 += 1
                                
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
                                    stats[window]["vb"][sample][opt]["velocity"].append(v_fit)
                                    stats[window]["vb"][sample][opt]["w_rms"].append(w_rms)
                                    stats[window]["vb"][sample][opt]["uw_rms"].append(uw_rms)
                                    stats[window]["vb"][sample][opt]["max_pull"].append(max_pull)
                                    stats[window]["vb"][sample][opt]["n_outliers"].append(n_outliers)
                                    stats[window]["vb"][sample][opt]["frac_outliers"].append(frac_outliers)
                                    stats[window]["vb"][sample][opt]["pulls"].append(pulls)
                                    track_vb += 1
                                if vb_hits >= 3 and ib_hits >= 2:
                                    stats[window]["ib"][sample][opt]["velocity"].append(v_fit)
                                    stats[window]["ib"][sample][opt]["w_rms"].append(w_rms)
                                    stats[window]["ib"][sample][opt]["uw_rms"].append(uw_rms)
                                    stats[window]["ib"][sample][opt]["max_pull"].append(max_pull)
                                    stats[window]["ib"][sample][opt]["n_outliers"].append(n_outliers)
                                    stats[window]["ib"][sample][opt]["frac_outliers"].append(frac_outliers)
                                    stats[window]["ib"][sample][opt]["pulls"].append(pulls)
                                    track_ib += 1
                                if vb_hits >= 3 and ib_hits >= 2 and ob_hits >= 2:
                                    stats[window]["ob"][sample][opt]["velocity"].append(v_fit)
                                    stats[window]["ob"][sample][opt]["w_rms"].append(w_rms)
                                    stats[window]["ob"][sample][opt]["uw_rms"].append(uw_rms)
                                    stats[window]["ob"][sample][opt]["max_pull"].append(max_pull)
                                    stats[window]["ob"][sample][opt]["n_outliers"].append(n_outliers)
                                    stats[window]["ob"][sample][opt]["frac_outliers"].append(frac_outliers)
                                    stats[window]["ob"][sample][opt]["pulls"].append(pulls)
                                    track_ob += 1

                    except Exception as e:
                        print(f"Crash while reading {file_path}: {e}")
                    finally:
                        reader.close()

            print(f"Finished {window} / {opt}")

            if track_pass_eta != 0:
                vb_percent = (track_vb / track_pass_eta) * 100
                ib_percent = (track_ib / track_pass_eta) * 100
                ob_percent = (track_ob / track_pass_eta) * 100
                overc_percent = (over_c / track_pass_eta) * 100
            else:
                vb_percent = 0.00
                ib_percent = 0.00
                ob_percent = 0.00
                overc_percent = 0.00

            print(f"{window} window, {opt} option stats:")
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
    "bib": "100% bib",
    "no_bib": "No BIB"
}


def get_bib_feature_arrays(feature, window, option):
    arrays = {req: None for req in track_req_names}

    for req in track_req_names:
        data = bib_stats[window][req][option][feature]

        if feature == "pulls":
            arr = np.concatenate(data) if len(data) > 0 else np.array([])
        else:
            arr = np.asarray(data, float)

        arrays[req] = arr

    return arrays


def get_feature_arrays_by_sample(feature, window, option):
    # structure: [req][sample]
    arrays = {req: {} for req in track_req_names}

    for req in track_req_names:
        for sample in samples:
            data = stats[window][req][sample][option][feature]

            if feature == "pulls":
                arr = np.concatenate(data) if len(data) > 0 else np.array([])
            else:
                arr = np.asarray(data, float)

            arrays[req][sample] = arr

    return arrays


def plot_feature(feature, window, option, x_lim=None):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5))

    titles = [
        "$\\geq$3 VB Hits",
        "$\\geq$3 VB, $\\geq$2 IB Hits",
        "$\\geq$3 VB, $\\geq$2 IB, $\\geq$2 OB Hits"
    ]

    data_dict = get_feature_arrays_by_sample(feature, window, option)
    bib_data_dict = None
    if bib_stats is not None:
        bib_data_dict = get_bib_feature_arrays(feature, window, "10_bib")

    print("--------------------------")
    print(f"Feature: {feature}")
    print("--------------------------")
    for ax, req, title in zip(axes, track_req_names, titles):

        if req == "ob":
            print(f"Barrel requirement: {req}")
            print("--------------------------")
        
        for sample in samples:
            arr = data_dict[req][sample]
            arr = arr[np.isfinite(arr)]

            min_arr = np.min(arr)
            max_arr = np.max(arr)
            
            if req == "ob":
                print(f"Mass {sample_to_mass[sample]} TeV: Min value: {np.round(min_arr, 2)}, Max value: {np.round(max_arr, 2)}")

            if x_lim is not None:
                arr = arr[(arr >= x_lim[0]) & (arr <= x_lim[1])]

            if arr.size == 0:
                continue

            weights = np.full(arr.size, 100.0 / arr.size)

            bins = 50 if feature != "hits" else np.arange(np.min(arr)-0.5, np.max(arr)+1.5, 1)

            ax.hist(
                arr,
                bins=bins,
                weights=weights,
                histtype='step',   # KEY: line-style histogram
                fill=False,
                linewidth=2,
                label=f"{sample_to_mass[sample]} TeV"
            )

        if bib_data_dict is not None:
            arr_bib = bib_data_dict[req]
            arr_bib = arr_bib[np.isfinite(arr_bib)]

            if x_lim is not None:
                arr_bib = arr_bib[(arr_bib >= x_lim[0]) & (arr_bib <= x_lim[1])]

            if arr_bib.size > 0:
                weights_old = np.full(arr_bib.size, 100.0 / arr_bib.size)
                
                ax.hist(
                    arr_bib,
                    bins=bins,
                    weights=weights_old,
                    histtype='step',
                    color='grey',
                    fill=True,
                    alpha=0.30,
                    edgecolor="black",
                    linewidth=2.0,
                    label="BIB background"
                )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel_map[feature], fontsize=12)
        ax.grid(True, alpha=0.2)

        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.legend(fontsize=8, frameon=False)

    axes[0].set_ylabel("Normalized Counts (%)", fontsize=12)

    fig.suptitle(
        f"{title_map.get(feature, feature)}  |  bkg: {window} window, {bib_name[option]}"
    )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

pdf_path = "/scratch/wandriscok/kate_mucoll_scripts/stau_studies/pdf/residual.pdf"

with PdfPages(pdf_path) as pdf:
    features = ["velocity", "w_rms", "uw_rms", "max_pull", "n_outliers", "frac_outliers", "pulls"]
    for window in windows:
        for opt in option:
            plot_feature("velocity", window, opt, (0, 350))
            plot_feature("w_rms", window, opt, (0, 2.5))
            plot_feature("uw_rms", window, opt, (0, 0.1))
            plot_feature("max_pull", window, opt, (0, 7))
            plot_feature("n_outliers", window, opt, (0, 1.5))
            plot_feature("frac_outliers", window, opt, (0, 0.15))
            plot_feature("pulls", window, opt, (-7.5, 7.5))
    print(f"Saved plots to {pdf_path}") 

for window in windows:
    for opt in option:
        for sample in samples:
            n = len(stats[window]["vb"][sample][opt]["velocity"])
            print(f"{window}, {opt}, {sample}: {n}")
