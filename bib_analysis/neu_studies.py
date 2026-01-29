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
bib_options = ["bib"]
#windows = ["loose", "tight"]
#bib_options = ["bib", "no_bib"]
CACHE = pathlib.Path("cache/nu_bkg_stats.pkl")
plot_path = "/scratch/wandriscok/kate_mucoll_script/analysis.pdf"
n_files = 2500
Bfield = 3.57
speedoflight = 299792458/1000000  # mm/ns

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap"
}
bib_name = {
    "bib": "10% BIB",
    "no_bib": "No BIB"
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

guess_velo = 180
def reco_velo(function_type, times, pos, spatial_unc):
    x = np.asarray(times, dtype=float)
    y = np.asarray(pos, dtype=float)
    s = np.asarray(spatial_unc, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    x, y, s = x[m], y[m], s[m]

    if x.size < 3 or np.allclose(x, x.mean()):
        return np.nan, np.nan

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
    except Exception:
        v_err = np.nan

    return float(p[0]), v_err

stats = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        stats = pickle.load(f)

if stats is None:
    stats = {
        "chi2_values": [],
        "data": {window: {
            req: {
                "bib": {
                    "pT": [],
                    "hits": [],
                    "velocity": []
                },
                "no_bib": {
                    "pT": [],
                    "hits": [],
                    "velocity": []
            }
        }
        for req in track_req_names
    }
    for window in windows
    }}

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        total_track = 0
        track_pass_chi2 = 0
        track_vb = 0
        track_ib = 0
        track_ob = 0
        print(f"Analyzing {window} window...")
        for option in bib_options:
            print(f"Analyzing {option}...")
            for ifile in tqdm(range(n_files)):
                file_name = f"nu_background_reco{ifile}.slcio"
                file_path = os.path.join(dir, window, option, file_name)
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    print(f"couldn't open {file_path}")
                    continue
                reader.open(file_path)
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

                        for hit in track_hits:
                            decoder.setValue(int(hit.getCellID0()))
                            system = decoder["system"].value()
                            layer = decoder["layer"].value()
                            if system in (1,2):
                                vb_hits += 0.5
                                spatial_unc.append(0.005)
                            elif system in (3,4):
                                ib_hits += 1
                                spatial_unc.append(0.007)
                            elif system in (5,6):
                                ob_hits += 1
                                spatial_unc.append(0.007)

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

                        v_fit, v_err = reco_velo(linearfunc, track_times, track_pos, spatial_unc) 
                        beta = v_fit / speedoflight
                        
                        total_hits = vb_hits + ib_hits + ob_hits

                        if vb_hits >= 3:
                            stats["data"][window]["vb"][option]["pT"].append(reco_pT)
                            stats["data"][window]["vb"][option]["hits"].append(total_hits)
                            stats["data"][window]["vb"][option]["velocity"].append(v_fit)
                            track_vb += 1
                        if vb_hits >= 3 and ib_hits >= 2:
                            stats["data"][window]["ib"][option]["pT"].append(reco_pT)
                            stats["data"][window]["ib"][option]["hits"].append(total_hits)
                            stats["data"][window]["ib"][option]["velocity"].append(v_fit)
                            track_ib += 1
                        if vb_hits >= 3 and ib_hits >= 2 and ob_hits >= 2:
                            stats["data"][window]["ob"][option]["pT"].append(reco_pT)
                            stats["data"][window]["ob"][option]["hits"].append(total_hits)
                            stats["data"][window]["ob"][option]["velocity"].append(v_fit)
                            track_ob += 1

        vb_percent = (track_vb / track_pass_chi2) * 100
        ib_percent = (track_ib / track_pass_chi2) * 100
        ob_percent = (track_ob / track_pass_chi2) * 100

        print(f"Number of total tracks: {total_track}")
        print(f"Number of tracks passing chi2: {track_pass_chi2}")
        print(f"Vertex cut: {track_vb} / {track_pass_chi2} -> {vb_percent:.2f}%")
        print(f"Inner cut: {track_ib} / {track_pass_chi2} -> {ib_percent:.2f}%")
        print(f"Outer cut: {track_ob} / {track_pass_chi2} -> {ob_percent:.2f}%")
    

    #print(stats)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Writing cache to {CACHE}")
    print("Saved cache successfully.")

title_map = {
        "hits": "Hits per Track",
        "velocity": "Reconstructed Velocity",
        "pT": "Reconstructed pT"
    }

xlabel_map = {
        "hits": "Number of Hits",
        "velocity": "Velocity [mm/ns]",
        "pT": "pT [GeV]"
    }


def plot_feature(feature, x_lim=None):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 4.5))
    
    titles = [
        "$\geq$3 VB Hits",
        "$\geq$3 VB, $\geq$2 IB Hits",
        "$\geq$3 VB, $\geq$2 IB, $\geq$2 OB Hits"
    ]

    feature_arrays = [
        np.asarray(stats["data"][window][req][option][feature])
        for req in track_req_names
    ]
    
    if x_lim is not None:
        feature_arrays = [
            arr[(arr >= x_lim[0]) & (arr <= x_lim[1])]
            for arr in feature_arrays
        ]

    for ax, arr, title in zip(axes, feature_arrays, titles):
        weights = np.full(arr.size, 100.0 / arr.size)
        if feature == "hits":
            bins = np.arange(np.min(arr) - 0.5, np.max(arr) + 1.5, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            bins = 20

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

    fig.suptitle(f"{title_map.get(feature, feature)} (time + tof) |  bkg: {window} window, {bib_name[option]}")

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

with PdfPages('analysis.pdf') as pdf:
    features = ["hits", "velocity", "pT"]
    for window in windows:
        for option in bib_options:
            plot_feature("pT", x_lim=(0,120))
            plot_feature("hits", x_lim=(0, 18))
            plot_feature("velocity", x_lim=(-200, 500))
    print(f"Saved plots to analysis.pdf") 