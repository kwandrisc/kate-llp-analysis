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
CACHE = pathlib.Path("cache/nu_bkg_posplots.pkl")
plot_path = "/scratch/wandriscok/kate_mucoll_script/analysis.pdf"
n_files = 5
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
        return np.nan, np.nan, np.nan

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

    return float(p[0]), float(p[1]), v_err

stats = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        stats = pickle.load(f)

if stats is None:
    stats = {
        "position": [],
        "time": [],
        "rchi2": [],
        "vfit": [],
        "intercept": [],
        "error": [],
        "pT": []
        }

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        total_track = 0
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

                        reco_pT = 0.3 * Bfield / fabs(track.getOmega() * 1000.)

                        track_hits = track.getTrackerHits()

                        track_hit_times = []
                        track_hit_pos = []
                        spatial_unc = []

                        for hit in track_hits:
                            decoder.setValue(int(hit.getCellID0()))
                            system = decoder["system"].value()
                            layer = decoder["layer"].value()
                            if system in (1,2):
                                spatial_unc.append(0.005)
                            elif system in (3,4):
                                spatial_unc.append(0.007)
                            elif system in (5,6):
                                spatial_unc.append(0.007)
                            
                            hit_time = hit.getTime()

                            x = hit.getPosition()[0]
                            y = hit.getPosition()[1]
                            z = hit.getPosition()[2]
                            hit_pos = np.sqrt(x**2 + y**2 + z**2)
                            tof = hit_pos/speedoflight

                            corrected_time = hit_time + tof

                            track_hit_times.append(corrected_time)
                            track_hit_pos.append(hit_pos)

                        v_fit, intercept_fit, v_err = reco_velo(linearfunc, track_hit_times, track_hit_pos, spatial_unc) 
                        
                        pos_arr = np.asarray(track_hit_pos, float)
                        time_arr = np.asarray(track_hit_times, float)

                        m = np.isfinite(pos_arr) & np.isfinite(time_arr)
                        
                        stats["position"].append(pos_arr[m])
                        stats["time"].append(time_arr[m])
                        stats["rchi2"].append(reduced_chi2)
                        stats["vfit"].append(v_fit)
                        stats["intercept"].append(intercept_fit)
                        stats["error"].append(v_err)
                        stats['pT'].append(reco_pT)

    #print(stats)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Writing cache to {CACHE}")
    print("Saved cache successfully.")


def plot_pos(pos: np.ndarray, time: np.ndarray, rchi2: float, v_fit: float, v_int: float, v_err: float, pt: float, track_num: int):
    
    fig, ax = plt.subplots()
    
    pos_line = linearfunc([v_fit, v_int], time)
    
    ax.scatter(time, pos, color="black", label="BIB data points")
    ax.plot(time, pos_line, label=f"Best fit v = {v_fit:.1f} ± {v_err:.1f} mm/ns")

    if np.isfinite(v_err):
        pos_up = linearfunc([v_fit + v_err, v_int], time)
        pos_dn = linearfunc([v_fit - v_err, v_int], time)
        ax.fill_between(time, pos_dn, pos_up, alpha=0.2)
    
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Position (mm)")
    ax.set_title(f"Reconstructed Track #{track_num}")

    ax.text(
        0.05, 0.95,
        f"$\\chi^2_{{red}}$ = {rchi2:.2f}",
        transform=ax.transAxes,
        verticalalignment='top'
    )

    ax.text(
        0.05, 0.88,
        f"pT = {pt:.2f} GeV",
        transform=ax.transAxes,
        verticalalignment='top'
    )

    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2000)

    pdf.savefig(fig)
    plt.close(fig)

#print(stats["position"])

nan_count = 0
# small_v = 0

with PdfPages("pos_plot.pdf") as pdf:
    for i in tqdm(range(500)):
        pos = stats["position"][i]
        time = stats["time"][i]
        rchi2 = stats["rchi2"][i]
        v_fit = stats["vfit"][i]
        y_intercept = stats["intercept"][i]
        v_err = stats["error"][i]
        pT = stats["pT"][i]

        if not np.isfinite(v_fit) or not np.isfinite(y_intercept):
            nan_count += 1
            continue

        plot_pos(pos, time, rchi2, v_fit, y_intercept, v_err, pT, i)

    print("Saved plots to pos_plot.pdf")

print(f"Number of tracks with NaN fit: {nan_count}")
# print(f"Number of tracks with v < 250 mm/ns: {small_v}")

