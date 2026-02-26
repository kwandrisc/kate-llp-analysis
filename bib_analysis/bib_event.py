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
PT_MIN   = 0
MASS_MIN = 0
BETA_MAX = 1.0
def pass_pt(t):   return np.isfinite(t["pT"])   and (t["pT"]   > PT_MIN)
def pass_mass(t): return np.isfinite(t["mass"]) and (t["mass"] > MASS_MIN)
def pass_beta(t): return np.isfinite(t["beta"]) and (t["beta"] < BETA_MAX)
def pass_all(t):  return pass_pt(t) and pass_mass(t) and pass_beta(t)

dir = "/ospool/uc-shared/project/futurecolliders/wandriscok/reco/nu_background/"
windows = ["loose"]
#bib_options = ["10_bib", "bib"]
bib_options = ["10_bib"]
#windows = ["loose", "tight"]
CACHE = pathlib.Path("cache/bib_event_lead_sub.pkl")
SAVE_EVERY = 50
file_ranges = {
    "10_bib": (0, 2500),
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
#track_req_names = ["vb", "ib", "ob"]
track_req_names = ["ob"]

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
        return np.nan, np.nan, False

    t_pred = (r - b) / v
    dt = t - t_pred

    pulls = np.abs(dt / st)
    has_outlier = np.any(pulls > 3)

    uw_rms_t = float(np.sqrt(np.mean(dt * dt)))          
    w_rms_t = float(np.sqrt(np.mean((dt / st) ** 2))) 
    return uw_rms_t, w_rms_t, has_outlier

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
        return np.nan, np.nan, np.nan, np.nan, np.nan

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

    uw_rms_t, w_rms_t, has_outlier = time_rms_from_fit(v, x, y, st, b=0.0)

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

    return v, v_err, uw_rms_t, w_rms_t, has_outlier


stats = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        stats = pickle.load(f)

if stats is None:
    stats = {
        window: {
            req: {
                option: {
                    "n_events": 0,   # total processed events
                    "leading_mass": [], "subleading_mass": [],
                    "leading_pT": [], "subleading_pT": [],
                    "leading_beta": [], "subleading_beta": [],
                    "leading_hits": [], "subleading_hits": [],
                    "leading_d0": [], "subleading_d0": [],
                    "leading_z0": [], "subleading_z0": [],
                    "leading_w_rms": [], "subleading_w_rms": [],
                } for option in bib_options
            } for req in track_req_names
        } for window in windows
    }

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        
        print(f"Analyzing {window} window...")
        for option in bib_options:
            total_tracks = 0
            track_vb = 0
            track_ib = 0
            track_ob = 0
            track_pass_eta = 0
            track_pass_w_rms = 0
            track_over_10tev = 0
            over_c = 0
            nan_value = 0
            tracks_w_outlier = 0
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
                    tracks_by_req = {req: [] for req in track_req_names}
                    for req in track_req_names:
                        stats[window][req][option]["n_events"] += 1

                    # stats[window][option]["n_events"] += 1
                    # tracks_by_req = {"vb": [], "ib": [], "ob": []}

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

                        track_hits = track.getTrackerHits()

                        vb_hits = 0
                        ib_hits = 0
                        ob_hits = 0

                        ## think about if this is giving right thing
                        reco_pT = 0.3 * Bfield / fabs(track.getOmega() * 1000.)

                        # looking at if pT right
                        # if itrack < 5:
                        #     print(f"DEBUG pT = {reco_pT:.2f} GeV, omega = {track.getOmega():.3e}")
                        
                        track_times = []
                        track_pos = []
                        spatial_unc = []
                        time_unc = []

                        for hit in track_hits:
                            decoder.setValue(int(hit.getCellID0()))
                            system = decoder["system"].value()

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
                            
                            x = hit.getPosition()[0]
                            y = hit.getPosition()[1]
                            z = hit.getPosition()[2]
                            hit_pos = np.sqrt(x*x + y*y + z*z)
                            tof = hit_pos / speedoflight
                            corrected_t = hit.getTime() + tof

                            track_times.append(corrected_t)
                            track_pos.append(hit_pos)
                        
                        v_fit, v_err, uw_rms, w_rms, has_outlier = reco_velo_no_intercept(track_times, track_pos, spatial_unc, time_unc) 
                        
                        beta = v_fit / speedoflight

                        # if velo > c, turning velo = c
                        if np.isfinite(v_fit) and v_fit > speedoflight:
                            v_fit = speedoflight
                            over_c += 1

                        pZ = reco_pT * track.getTanLambda()
                        p_total = np.sqrt(reco_pT**2 + pZ**2)

                        tan_lambda = track.getTanLambda()
                        eta = np.arcsinh(tan_lambda)
                        # eta cut
                        if abs(eta) > 0.8:
                            continue
                        track_pass_eta += 1

                        # one hit over 3 sigma cut
                        if has_outlier:
                            tracks_w_outlier += 1

                        # weighted rms cut
                        if (not np.isfinite(w_rms)) or (w_rms > 1.6):
                            continue
                        track_pass_w_rms += 1

                        if reco_pT > 10000:
                            track_over_10tev += 1
                            continue

                        beta_for_mass = v_fit / speedoflight 
                        
                        if np.isfinite(reco_pT) and np.isfinite(beta_for_mass) and (0 < beta_for_mass <= 1):
                            mass = p_total * math.sqrt(1.0/(beta_for_mass*beta_for_mass) - 1.0)
                        else:
                            mass = np.nan
                        
                        total_hits = vb_hits + ib_hits + ob_hits

                        track_state = track.getTrackStates()[0]
                        d0 = track_state.getD0()
                        z0 = track_state.getZ0()

                        track_info = {
                            "pT": float(reco_pT) if np.isfinite(reco_pT) else np.nan,
                            "beta": float(beta) if np.isfinite(beta) else np.nan,
                            "mass": float(mass) if np.isfinite(mass) else np.nan,
                            "hits": float(total_hits),
                            "d0": float(d0) if np.isfinite(d0) else np.nan,
                            "z0": float(z0) if np.isfinite(z0) else np.nan,
                            "w_rms": float(w_rms) if np.isfinite(w_rms) else np.nan
                        } 
                        
                        if vb_hits >= 3: 
                            #tracks_by_req["vb"].append(track_info)
                            track_vb += 1

                        if vb_hits >= 3 and ib_hits >= 2:
                            #tracks_by_req["ib"].append(track_info)
                            track_ib += 1
                        
                        if vb_hits >= 3 and ib_hits >= 2 and ob_hits >=2:
                            tracks_by_req["ob"].append(track_info)
                            track_ob += 1

                        if not np.isfinite(mass) or not np.isfinite(reco_pT) or not np.isfinite(beta):
                            nan_value += 1
                            continue

                    for req in ["ob"]:
                        tracks = tracks_by_req[req]

                        d = stats[window][req][option]
                        tracks.sort(key=lambda t: t["pT"], reverse=True)

                        if len(tracks) >= 1:
                            lead = tracks[0]
                            d["leading_mass"].append(lead["mass"])
                            d["leading_pT"].append(lead["pT"])
                            d["leading_beta"].append(lead["beta"])
                            d["leading_hits"].append(lead["hits"])
                            d["leading_d0"].append(lead["d0"])
                            d["leading_z0"].append(lead["z0"])
                            d["leading_w_rms"].append(lead["w_rms"])
                        else:
                            d["leading_mass"].append(float("nan"))
                            d["leading_pT"].append(float("nan"))
                            d["leading_beta"].append(float("nan"))
                            d["leading_hits"].append(float("nan"))
                            d["leading_d0"].append(float("nan"))
                            d["leading_z0"].append(float("nan"))
                            d["leading_w_rms"].append(float("nan"))

                        if len(tracks) >= 2:
                            sub = tracks[1]
                            d["subleading_mass"].append(sub["mass"])
                            d["subleading_pT"].append(sub["pT"])
                            d["subleading_beta"].append(sub["beta"])
                            d["subleading_hits"].append(sub["hits"])
                            d["subleading_d0"].append(sub["d0"])
                            d["subleading_z0"].append(sub["z0"])
                            d["subleading_w_rms"].append(sub["w_rms"])
                        else:
                            d["subleading_mass"].append(float("nan"))
                            d["subleading_pT"].append(float("nan"))
                            d["subleading_beta"].append(float("nan"))
                            d["subleading_hits"].append(float("nan"))
                            d["subleading_d0"].append(float("nan"))
                            d["subleading_z0"].append(float("nan"))
                            d["subleading_w_rms"].append(float("nan"))

                reader.close()

                if (ifile - start + 1) % SAVE_EVERY == 0:
                    with CACHE.open("wb") as f:
                        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Checkpoint saved at file {ifile}")

        print(f"Finished {window}")

        vb_percent = (track_vb / track_pass_eta) * 100
        ib_percent = (track_ib / track_pass_eta) * 100
        ob_percent = (track_ob / track_pass_eta) * 100
        overc_percent = (over_c / track_pass_eta) * 100

        print(f"num tracks that have 3 sigma hit: {tracks_w_outlier}")

        print(f"{window} window stats:")
        print(f"Number of total tracks: {total_tracks}")
        print(f"Number of tracks passing eta cut: {track_pass_eta}")
        print(f"Number of tracks passing 1.6 weighted rms max cut: {track_pass_w_rms}")
        print(f"Number of tracks over 10TeV pT: {track_over_10tev}")
        print(f"Number of tracks rejected because NaN value: {nan_value}")
        print(f"Vertex cut: {track_vb} / {track_pass_eta} -> {vb_percent:.2f}%")
        print(f"Inner cut: {track_ib} / {track_pass_eta} -> {ib_percent:.2f}%")
        print(f"Outer cut: {track_ob} / {track_pass_eta} -> {ob_percent:.2f}%")
        print(f"Number of tracks with speed over c: {over_c} -> {overc_percent:.2f}% of tracks passing eta")

    
#print(stats["loose"]["ob"]["10_bib"]["leading_w_rms"])
CACHE.parent.mkdir(exist_ok=True)
with CACHE.open("wb") as f:
    pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Writing final cache to {CACHE}")
print("Saved cache successfully.")


def plot_cut_multiplicity(stats, window, option, req="vb", max_tracks=30, pdf=None):
    d = stats[window][req][option]

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
# with PdfPages("pdf/event_cut_plots.pdf") as pdf:
#     for window in windows:
#         for option in bib_options:
#             for req in ["vb", "ib", "ob"]:
#                 plot_cut_multiplicity(stats, window, option, req=req, pdf=pdf)


def _event_norm_hist(ax, arr, N_events, bins, label):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or N_events <= 0:
        return 0.0
    w = np.full_like(x, 1.0 / N_events, dtype=float)  # sums to fraction of events
    ax.hist(x, bins=bins, weights=w, histtype="step", linewidth=2, label=label)
    return x.size / N_events  # fraction of events with a finite value


def plot_lead_sub(stats, window, option, req, bins_cfg=None, xlims_cfg=None, 
                  tick_major=18, tick_minor=16):
    if bins_cfg is None:
        bins_cfg = {
            "pT": np.linspace(0, 100, 60),
            "mass": np.linspace(0, 200, 60),
            "beta": np.linspace(0.8, 1.02, 60),
            "d0": np.linspace(-0.01, 0.01, 60),
            "z0": np.linspace(-0.01, 0.01, 60),
            "hits": np.linspace(0, 18, 60),
            "w_rms": np.linspace(0, 2, 60),
        }
    if xlims_cfg is None:
        xlims_cfg = {
            "pT": (0, 100),
            "mass": (0, 200),
            "beta": (0.8, 1.02),
            "d0": (-0.01, 0.01),
            "z0": (-0.01, 0.01),
            "hits": (0, 18),
            "w_rms": (0, 2),
        }
    
    features = [
        ("pT", "leading_pT", "subleading_pT", r"$p_T$ [GeV]"),
        ("mass", "leading_mass", "subleading_mass", r"Mass [GeV]"),
        ("beta", "leading_beta", "subleading_beta", r"$\beta$"),
        ("d0", "leading_d0", "subleading_d0", r"D0"),
        ("z0", "leading_z0", "subleading_z0", r"Z0"),
        ("hits", "leading_hits", "subleading_hits", r"Number of Hits"),
        ("w_rms", "leading_w_rms", "subleading_w_rms", r"Weighted RMS"),
    ]

    d = stats[window][req][option]

    N_events = int(d.get("n_events", 0))

    for key, lead_key, sub_key, xlabel in features:
        fig, ax = plt.subplots(figsize=(8,6))

        print("Leading length:", len(d.get(lead_key, [])))
        print("Subleading length:", len(d.get(sub_key, [])))

        frac_lead = _event_norm_hist(ax, d.get(lead_key, []), 
                                     N_events, bins_cfg[key],
                                     label="Leading")
        frac_sub = _event_norm_hist(ax, d.get(sub_key, []), 
                                    N_events, bins_cfg[key],
                                    label="Subleading")
        
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel("Fraction of events per bin", fontsize=20)

        ax.tick_params(axis="both", which="major",
                        labelsize=tick_major, length=6, width=1.5)
        ax.tick_params(axis="both", which="minor",
                        labelsize=tick_minor, length=4, width=1.0)

        if key in xlims_cfg and xlims_cfg[key] is not None:
            ax.set_xlim(*xlims_cfg[key])

        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=13, loc="upper right")

        ax.text(0.02, 0.98, "Muon Collider",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=20, fontweight="bold", style="italic")
        ax.text(0.02, 0.93, f"muons, {option}, {window}, req={req}",
                ha="left", va="top", transform=ax.transAxes, fontsize=14)
        ax.text(0.02, 0.89, f"N_events={N_events}",
                ha="left", va="top", transform=ax.transAxes, fontsize=14)

        ax.text(0.98, 0.02,
                f"Frac(events w/ leading) ~ {frac_lead:.3f}\n"
                f"Frac(events w/ subleading) ~ {frac_sub:.3f}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=12)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


print(stats["loose"]["ob"]["10_bib"]["subleading_pT"])

with PdfPages("pdf/lead_sub_plots.pdf") as pdf:
    for window in windows:
        for option in bib_options:
            for req in ["ob"]:
                plot_lead_sub(stats=stats, window=window, option=option, req=req, tick_major=20, tick_minor=18)
                print("Saved event-normalized leading/subleading plots to pdf/lead_sub_plots.pdf")
