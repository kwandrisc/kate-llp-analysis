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

# DEFINING CUTS 
PT_MIN   = 800.0
MASS_MIN = 500.0
BETA_MAX = 0.99
def pass_pt(t):   return np.isfinite(t["pT"])   and (t["pT"]   > PT_MIN)
def pass_mass(t): return np.isfinite(t["mass"]) and (t["mass"] > MASS_MIN)
def pass_beta(t): return np.isfinite(t["beta"]) and (t["beta"] < BETA_MAX)
def pass_all(t):  return pass_pt(t) and pass_mass(t) and pass_beta(t)

dir = "/ospool/uc-shared/project/futurecolliders/wandriscok/reco/nu_background/"
windows = ["loose"]
#bib_options = ["10_bib", "bib"]
bib_options = ["10_bib"]
#windows = ["loose", "tight"]
CACHE = pathlib.Path("cache/nu_bkg_event.pkl")
plot_path = "/scratch/wandriscok/kate_mucoll_script/analysis.pdf"
file_ranges = {
    "10_bib": (0, 1800),
    "bib": (4, 8)
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
        window: {
            option: {
                "n_events": 0,   # total processed events
                "vb": {
                    "n_total_tracks": [],
                    "n_pass_pt": [],
                    "n_pass_mass": [],
                    "n_pass_beta": [],
                    "n_pass_all": [],
                },
                "ib": {
                    "n_total_tracks": [],
                    "n_pass_pt": [],
                    "n_pass_mass": [],
                    "n_pass_beta": [],
                    "n_pass_all": [],
                },
                "ob": {
                    "n_total_tracks": [],
                    "n_pass_pt": [],
                    "n_pass_mass": [],
                    "n_pass_beta": [],
                    "n_pass_all": [],
                },
            } for option in bib_options
        } for window in windows
    }

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        total_tracks = 0
        track_vb = 0
        track_ib = 0
        track_ob = 0
        track_pass_chi2 = 0
        track_pass_eta = 0
        over_c = 0
        print(f"Analyzing {window} window...")
        for option in bib_options:
            print(f"Analyzing {option}...")
            start, stop = file_ranges[option]
            for ifile in tqdm(range(start, stop)): 
                file_name = f"nu_background_reco{ifile}.slcio"
                file_path = os.path.join(dir, window, option, file_name)
                
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    print(f"couldn't open {file_path}")
                    continue
                
                try:
                    reader.open(file_path)
                except Exception as e:
                    print(f"LCIO failed to open reco{ifile}")
                    continue

                for event in reader:
                    # tracks_by_req = {req: [] for req in track_req_names}
                    # for req in track_req_names:
                    #     stats[window][req][option]["n_events"] += 1

                    stats[window][option]["n_events"] += 1
                    tracks_by_req = {"vb": [], "ib": [], "ob": []}

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
                        total_tracks += 1

                        chi2 = track.getChi2()
                        ndf = track.getNdf()
                        if ndf == 0:
                            continue
                        reduced_chi2 = chi2 / ndf

                        # reduced chi2 cut
                        if reduced_chi2 > chi2_cut:
                            continue

                        track_pass_chi2 += 1

                        track_hits = track.getTrackerHits()

                        vb_hits = 0
                        ib_hits = 0
                        ob_hits = 0

                        for hit in track_hits:
                            decoder.setValue(int(hit.getCellID0()))
                            system = decoder["system"].value()

                            if system in (1, 2):
                                vb_hits += 0.5
                            elif system in (3, 4):
                                ib_hits += 1
                            elif system in (5, 6):
                                ob_hits += 1

                        ## think about if this is giving right thing
                        reco_pT = 0.3 * Bfield / fabs(track.getOmega() * 1000.)

                        # looking at if pT right
                        # if itrack < 5:
                        #     print(f"DEBUG pT = {reco_pT:.2f} GeV, omega = {track.getOmega():.3e}")

                        
                        track_times = []
                        track_pos = []
                        spatial_unc = []

                        for hit in track_hits:
                            decoder.setValue(int(hit.getCellID0()))
                            system = decoder["system"].value()

                            x = hit.getPosition()[0]
                            y = hit.getPosition()[1]
                            z = hit.getPosition()[2]
                            hit_pos = np.sqrt(x*x + y*y + z*z)
                            tof = hit_pos / speedoflight
                            corrected_t = hit.getTime() + tof

                            track_times.append(corrected_t)
                            track_pos.append(hit_pos)

                            if system in (1,2):
                                spatial_unc.append(0.005)
                            else:
                                spatial_unc.append(0.007)
                        
                        v_fit, v_err = reco_velo(linearfunc, track_times, track_pos, spatial_unc) 
                        
                        beta = v_fit / speedoflight

                        if np.isfinite(v_fit) and v_fit > speedoflight:
                            v_fit = speedoflight
                            over_c += 1

                        pZ = reco_pT * track.getTanLambda()
                        p_total = np.sqrt(reco_pT**2 + pZ**2)

                        tan_lambda = track.getTanLambda()
                        eta = np.arcsinh(tan_lambda)
                        if abs(eta) > 0.8:
                            continue

                        track_pass_eta += 1

                        beta_for_mass = v_fit / speedoflight 
                        
                        if np.isfinite(reco_pT) and np.isfinite(beta_for_mass) and (0 < beta_for_mass <= 1):
                            mass = p_total * math.sqrt(1.0/(beta_for_mass*beta_for_mass) - 1.0)
                        else:
                            mass = np.nan
                        
                        total_hits = vb_hits + ib_hits + ob_hits

                        track_info = {
                            "pT": float(reco_pT) if np.isfinite(reco_pT) else np.nan,
                            "beta": float(beta) if np.isfinite(beta) else np.nan,
                            "mass": float(mass) if np.isfinite(mass) else np.nan,
                            "hits": float(total_hits),
                        } 
                        
                        if vb_hits >= 3 and ib_hits >= 2 and ob_hits >=2:
                            tracks_by_req["ob"].append(track_info)
                            track_ob += 1
                                
                        if vb_hits >= 3 and ib_hits >= 2:
                            tracks_by_req["ib"].append(track_info)
                            track_ib += 1

                        if vb_hits >= 3: 
                            tracks_by_req["vb"].append(track_info)
                            track_vb += 1

                        
                    for req in ["vb", "ib", "ob"]:
                        tracks = tracks_by_req[req]

                        n_total = len(tracks)
                        n_pt   = sum(pass_pt(t)   for t in tracks)
                        n_mass = sum(pass_mass(t) for t in tracks)
                        n_beta = sum(pass_beta(t) for t in tracks)
                        n_all  = sum(pass_all(t)  for t in tracks)

                        d = stats[window][option][req]

                        d["n_total_tracks"].append(n_total)
                        d["n_pass_pt"].append(n_pt)
                        d["n_pass_mass"].append(n_mass)
                        d["n_pass_beta"].append(n_beta)
                        d["n_pass_all"].append(n_all)

                reader.close()

        print(f"Finished {window}")

        eta_percent = (track_pass_eta / track_pass_chi2) * 100
        vb_percent = (track_vb / track_pass_eta) * 100
        ib_percent = (track_ib / track_pass_eta) * 100
        ob_percent = (track_ob / track_pass_eta) * 100
        overc_percent = (over_c / track_pass_eta) * 100

        print(f"{window} window stats:")
        print(f"Number of total tracks: {total_tracks}")
        print(f"Number of tracks passing chi2: {track_pass_chi2}")
        print(f"Number of tracks passing eta cut: {track_pass_eta} / {track_pass_chi2} -> {eta_percent:.2f}%")
        print(f"Vertex cut: {track_vb} / {track_pass_eta} -> {vb_percent:.2f}%")
        print(f"Inner cut: {track_ib} / {track_pass_eta} -> {ib_percent:.2f}%")
        print(f"Outer cut: {track_ob} / {track_pass_eta} -> {ob_percent:.2f}%")
        print(f"Number of tracks with speed over c: {over_c} -> {overc_percent:.2f}% of tracks passing eta")

    
print(stats)
CACHE.parent.mkdir(exist_ok=True)
with CACHE.open("wb") as f:
    pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Writing cache to {CACHE}")
print("Saved cache successfully.")


def print_bib_track_summary(stats, windows, bib_options):
    sep = "-" * 120

    for window in windows:
        for option in bib_options:
            print("\n" + sep)
            print(f"BIB Track Multiplicity Summary | window = {window} | option = {option}")
            print(sep)

            N_events = stats[window][option]["n_events"]
            print(f"Total processed events: {N_events}")
            print(sep)

            header = (
                "REQ | Tracks>=1      |  pT pass      |  Mass pass    |  Beta pass     |  All cuts"
            )
            print(header)
            print(sep)

            for req in ["vb", "ib", "ob"]:
                d = stats[window][option][req]

                def count_events(arr, min_tracks=1):
                    return sum(x >= min_tracks for x in arr)

                def fmt(n):
                    return f"{n}/{N_events} ({100*n/N_events:5.1f}%)"

                evt_tracks = count_events(d["n_total_tracks"], 1)
                evt_pt     = count_events(d["n_pass_pt"], 1)
                evt_mass   = count_events(d["n_pass_mass"], 1)
                evt_beta   = count_events(d["n_pass_beta"], 1)
                evt_all    = count_events(d["n_pass_all"], 1)

                print(
                    f"{req.upper():>3} | "
                    f"{fmt(evt_tracks):>12} | "
                    f"{fmt(evt_pt):>12} | "
                    f"{fmt(evt_mass):>12} | "
                    f"{fmt(evt_beta):>12} | "
                    f"{fmt(evt_all):>12}"
                )

            print(sep)
            print("Each value shows: (# events with ≥1 such track) / (total events)")
            print(sep)

# getting info about the tracks passing each
def summarize_track_multiplicity(arr):
    arr = np.array(arr)

    return {
        "mean": np.mean(arr),
        "max": np.max(arr),
        "n0": np.sum(arr == 0),
        "n1": np.sum(arr == 1),
        "n2": np.sum(arr == 2),
        "n3p": np.sum(arr >= 3),
    }


def print_bib_track_cut_summary(stats, windows, options, reqs):

    sep = "-" * 100

    for window in windows:
        for option in options:

            print("\n" + sep)
            print(f"BIB Track Cut Multiplicity | window = {window} | option = {option}")
            print(sep)

            for req in reqs:
                d = stats[window][option][req]
                N_events = stats[window][option]["n_events"]

                print(f"\nRequirement region: {req.upper()}  (Events = {N_events})")

                for key, label in [
                    ("n_pass_pt",   "pT > 800 GeV"),
                    ("n_pass_mass", "Mass > 500 GeV"),
                    ("n_pass_beta", "Beta < 0.99"),
                    ("n_pass_all",  "All cuts"),
                ]:
                    s = summarize_track_multiplicity(d[key])

                    print(
                        f"  {label:<15} | "
                        f"avg tracks/event = {s['mean']:.2f} | "
                        f"max = {s['max']:3d} | "
                        f"events with 0/1/2/≥3 tracks = "
                        f"{s['n0']:3d} / {s['n1']:3d} / {s['n2']:3d} / {s['n3p']:3d}"
                    )

            print(sep)



def print_bib_cut_efficiencies(stats, windows, bib_options, track_reqs):
    sep = "=" * 120

    for window in windows:
        for option in bib_options:
            print("\n" + sep)
            print(f"BIB CUT EFFICIENCIES | window = {window} | option = {option}")
            print(sep)

            for req in track_reqs:
                d = stats[window][option][req]

                totals = np.array(d["n_total_tracks"])
                pt     = np.array(d["n_pass_pt"])
                mass   = np.array(d["n_pass_mass"])
                beta   = np.array(d["n_pass_beta"])
                allcut = np.array(d["n_pass_all"])

                total_tracks_all_events = totals.sum()

                print(f"\nRegion: {req.upper()}")
                print(f"  Total tracks (all events combined): {total_tracks_all_events}")

                def line(label, arr):
                    passed = arr.sum()
                    frac = 100 * passed / total_tracks_all_events if total_tracks_all_events > 0 else 0
                    print(f"  {label:<12}: {passed:7d}  / {total_tracks_all_events:7d}  = {frac:6.2f}%")

                line("pT cut", pt)
                line("Mass cut", mass)
                line("Beta cut", beta)
                line("All cuts", allcut)

                print("-" * 80)

            print(sep)



print_bib_track_summary(stats, windows, bib_options)


print_bib_track_cut_summary(
    stats,
    windows=["loose"],
    options=["10_bib"],
    reqs=["vb", "ib", "ob"]
)

print_bib_cut_efficiencies(stats, windows, bib_options, ["vb", "ib", "ob"])




# def plot_track_multiplicity(stats, window, option, pdf):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

#     reqs = ["vb", "ib", "ob"]
#     titles = ["≥3 VB hits", "≥3 VB & ≥2 IB", "≥3 VB & ≥2 IB & ≥2 OB"]

#     N_events = stats[window][option]["n_events"]

#     for ax, req, title in zip(axes, reqs, titles):
#         arr = np.array(stats[window][option][req]["n_total_tracks"])

#         if len(arr) == 0:
#             continue

#         # Percent of events normalization
#         weights = np.ones_like(arr) * (100.0 / N_events)

#         bins = np.arange(0, arr.max() + 2) - 0.5

#         ax.hist(arr, bins=bins, weights=weights,
#                 histtype="stepfilled", alpha=0.35,
#                 edgecolor="black", linewidth=1.8)

#         ax.set_title(title)
#         ax.set_xlabel("Tracks per event")
#         ax.grid(alpha=0.2)
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     axes[0].set_ylabel("Events (%)")
#     fig.suptitle(f"BIB Track Multiplicity | {window} window | {bib_name[option]}")

#     fig.tight_layout()
#     pdf.savefig(fig)
#     plt.close(fig)


# def plot_cut_multiplicity(stats, window, option, key, label, pdf):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

#     reqs = ["vb", "ib", "ob"]
#     titles = ["VB", "IB", "OB"]
#     N_events = stats[window][option]["n_events"]

#     for ax, req, title in zip(axes, reqs, titles):
#         arr = np.array(stats[window][option][req][key])
#         weights = np.ones_like(arr) * (100.0 / N_events)
#         bins = np.arange(0, arr.max() + 2) - 0.5

#         ax.hist(arr, bins=bins, weights=weights,
#                 histtype="stepfilled", alpha=0.35,
#                 edgecolor="black", linewidth=1.8)

#         ax.set_title(f"{title} tracks")
#         ax.set_xlabel("Tracks per event")
#         ax.grid(alpha=0.2)
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     axes[0].set_ylabel("Events (%)")
#     fig.suptitle(f"{label} Multiplicity | {window} | {bib_name[option]}")

#     fig.tight_layout()
#     pdf.savefig(fig)
#     plt.close(fig)


# with PdfPages("event_cut_plots.pdf") as pdf:
#     for window in windows:
#         for option in bib_options:
#             plot_track_multiplicity(stats, window, option, pdf)
#             plot_cut_multiplicity(stats, window, option, "n_pass_pt", "pT > 800 GeV", pdf)
#             plot_cut_multiplicity(stats, window, option, "n_pass_mass", "Mass > 500 GeV", pdf)
#             plot_cut_multiplicity(stats, window, option, "n_pass_beta", "Beta < 0.99", pdf)
#             plot_cut_multiplicity(stats, window, option, "n_pass_all", "All Cuts", pdf)



def plot_cut_multiplicity(stats, window, option, req="vb", max_tracks=30, pdf=None):
    d = stats[window][option][req]

    fig, ax = plt.subplots(figsize=(7,5))

    curves = [
        ("n_pass_pt",   r"$p_T > 800$ GeV"),
        ("n_pass_mass", r"Mass > 500 GeV"),
        ("n_pass_beta", r"$\beta < 0.99$"),
        ("n_pass_all",  "All cuts"),
    ]

    bins = np.arange(-0.5, max_tracks + 1.5, 1)

    for key, label in curves:
        arr = np.array(d[key])
        arr = np.clip(arr, 0, max_tracks)  # overflow protection

        ax.hist(arr,
                bins=bins,
                histtype="step",
                linewidth=2,
                label=label)

    ax.set_xlabel("Number of Reconstructed Tracks Passing Cut", fontsize=13)
    ax.set_ylabel("Events", fontsize=13)
    ax.set_xlim(-0.5, max_tracks)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"{window} window — {option} — {req.upper()} tracks")

    plt.tight_layout()

    # Save to PDF if a PdfPages object is provided
    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)  # close the figure so it doesn't display
    else:
        plt.show()  # fallback for interactive display

# Usage with PdfPages
with PdfPages("event_cut_plots.pdf") as pdf:
    for window in windows:
        for option in bib_options:
            for req in ["vb", "ib", "ob"]:
                plot_cut_multiplicity(stats, window, option, req=req, pdf=pdf)
