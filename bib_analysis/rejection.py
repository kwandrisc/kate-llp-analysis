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

STAU_CUTS = {
    1.0:  {"pt_min": 1500, "beta_min": 0.95, "beta_max": 0.983},
    1.5:  {"pt_min": 2000, "beta_min": 0.90, "beta_max": 0.98},
    2.0:  {"pt_min": 2000, "beta_min": 0.85, "beta_max": 0.95},
    2.5:  {"pt_min": 2000, "beta_min": 0.80, "beta_max": 0.90},
    3.0:  {"pt_min": 2000, "beta_min": 0.70, "beta_max": 0.85},
    3.5:  {"pt_min": 2000, "beta_min": 0.65, "beta_max": 0.80},
    4.0:  {"pt_min": 1900, "beta_min": 0.55, "beta_max": 0.70},
    4.5:  {"pt_min": 1400, "beta_min": 0.40, "beta_max": 0.55},
}

WRMS_CUT = 1.6
PT_MAX = 10000


dir = "/ospool/uc-shared/project/futurecolliders/wandriscok/reco/nu_background/"
windows = ["loose"]
#bib_options = ["10_bib", "bib"]
bib_options = ["bib"]
#windows = ["loose", "tight"]
CACHE = pathlib.Path("cache/100_bib_event_rejection_loose_stauwindows.pkl")
SAVE_EVERY = 2
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

CACHE.parent.mkdir(exist_ok=True)

if stats is None:
    stats = {
        window: {
            option: {
                "n_events": 0,   # total processed events
                "total_tracks": 0, "pass_chi2": 0,
                "pass_eta": 0, "pass_ob": 0,
                "pass_pT_window": 0, "pass_wrms": 0
            } for option in bib_options
        } for window in windows
    }


    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    for window in windows:
        
        print(f"Analyzing {window} window...")
        for option in bib_options:
            total_tracks = 0
            track_pass_ob = 0
            track_pass_eta = 0
            track_pass_chi2 = 0
            track_pass_w_rms = 0
            track_pass_gen_pT_window = 0

            abcd_A = 0  # pass pT, pass wrms
            abcd_B = 0  # fail pT, pass wrms
            abcd_C = 0  # pass pT, fail wrms
            abcd_D = 0  # fail pT, fail wrms

            events_ge1_fake = 0
            events_ge2_fake = 0

            mass_stats = {
                mass: {
                    "quality": 0,
                    "quality_wrms": 0,
                    "quality_pt": 0,
                    "quality_beta": 0,
                    "candidate_tracks": 0,
                    "events_ge1_candidate": 0,
                    "events_ge2_candidate": 0,
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": 0,
                }
                for mass in STAU_CUTS
            }


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

                while True:
                    try:
                        event = reader.readNextEvent()
                    except Exception:
                        break

                    if event is None:
                        print("No event error")
                        break

                    # stats[window][option]["n_events"] += 1
                    # tracks_by_req = {"vb": [], "ib": [], "ob": []}

                    try:
                        all_collections = event.getCollectionNames() 
                    except Exception:
                        print("Skipping invalid/null event")
                        continue
                    track_collection = event.getCollection("SiTracks") if "SiTracks" in all_collections else None 
                    if not track_collection:
                        print("issue 1")
                        continue
                    test_hit_coll = event.getCollection("VXDBarrelHits")
                    if test_hit_coll is None:
                        continue

                    tracks_by_req = {req: [] for req in track_req_names}
                    stats[window][option]["n_events"] += 1

                    encoding = test_hit_coll.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                    decoder = UTIL.BitField64(encoding)

                    n_candidates_this_event = {mass: 0 for mass in STAU_CUTS}
                    for itrack, track in enumerate(track_collection):
                        ###  all of the cuts in this script are reversed, getting the number that PASS each cut
                        
                        total_tracks += 1

                        tan_lambda = track.getTanLambda()
                        eta = np.arcsinh(tan_lambda)
                        # eta cut
                        if abs(eta) <= 0.8:
                            track_pass_eta += 1
                        
                        chi2 = track.getChi2()
                        ndf = track.getNdf()
                        if ndf == 0:
                            continue
                        reduced_chi2 = chi2 / ndf

                        # reduced chi2 cut
                        if reduced_chi2 <= chi2_cut:
                            track_pass_chi2 += 1

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

                        pZ = reco_pT * track.getTanLambda()
                        p_total = np.sqrt(reco_pT**2 + pZ**2)

                        beta_for_mass = v_fit / speedoflight 
                        
                        if np.isfinite(reco_pT) and np.isfinite(beta_for_mass) and (0 < beta_for_mass <= 1):
                            mass = p_total * math.sqrt(1.0/(beta_for_mass*beta_for_mass) - 1.0)
                        else:
                            mass = np.nan
                        
                        total_hits = vb_hits + ib_hits + ob_hits

                        track_state = track.getTrackStates()[0]
                        d0 = track_state.getD0()
                        z0 = track_state.getZ0()

                        passes_pT = np.isfinite(reco_pT) and (1000 <= reco_pT <= 10000)
                        passes_wrms = np.isfinite(w_rms) and (w_rms <= 1.6)

                        if passes_pT:
                            track_pass_gen_pT_window += 1

                        if passes_wrms:
                            track_pass_w_rms += 1

                        passes_eta = abs(eta) < 0.8
                        passes_chi2 = reduced_chi2 <= chi2_cut
                        passes_ob = (vb_hits >= 3) and (ib_hits >= 2) and (ob_hits >= 2)

                        if passes_ob:
                            track_pass_ob += 1

                        preselection = passes_eta and passes_chi2 and passes_ob

                        if preselection:
                            for mass, cuts in STAU_CUTS.items():

                                passes_wrms = np.isfinite(w_rms) and (w_rms <= WRMS_CUT)

                                passes_stau_pt = (
                                    np.isfinite(reco_pT)
                                    and (reco_pT >= cuts["pt_min"])
                                    and (reco_pT <= PT_MAX)
                                )

                                passes_stau_beta = (
                                    np.isfinite(beta)
                                    and (cuts["beta_min"] <= beta <= cuts["beta_max"])
                                )

                                passes_Y = passes_stau_pt and passes_stau_beta
                                is_candidate = passes_wrms and passes_Y

                                mass_stats[mass]["quality"] += 1

                                if passes_wrms:
                                    mass_stats[mass]["quality_wrms"] += 1

                                if passes_stau_pt:
                                    mass_stats[mass]["quality_pt"] += 1

                                if passes_stau_beta:
                                    mass_stats[mass]["quality_beta"] += 1

                                if is_candidate:
                                    mass_stats[mass]["candidate_tracks"] += 1
                                    n_candidates_this_event[mass] += 1

                                # ABCD:
                                # X = pass wrms
                                # Y = pass stau pT AND beta
                                if passes_wrms and passes_Y:
                                    mass_stats[mass]["A"] += 1
                                elif passes_wrms and (not passes_Y):
                                    mass_stats[mass]["B"] += 1
                                elif (not passes_wrms) and passes_Y:
                                    mass_stats[mass]["C"] += 1
                                else:
                                    mass_stats[mass]["D"] += 1
                        
                    
                        
                    #     # ABCD only among tracks that already pass eta, chi2, and OB
                    #     if preselection:

                    #         if passes_pT and passes_wrms:
                    #             abcd_A += 1
                    #             n_fake_tracks_this_event += 1

                    #         elif (not passes_pT) and passes_wrms:
                    #             abcd_B += 1

                    #         elif passes_pT and (not passes_wrms):
                    #             abcd_C += 1

                    #         else:
                    #             abcd_D += 1

                    # if n_fake_tracks_this_event >= 1:
                    #     events_ge1_fake += 1

                    # if n_fake_tracks_this_event >= 2:
                    #     events_ge2_fake += 1

                    for mass in STAU_CUTS:
                        if n_candidates_this_event[mass] >= 1:
                            mass_stats[mass]["events_ge1_candidate"] += 1

                        if n_candidates_this_event[mass] >= 2:
                            mass_stats[mass]["events_ge2_candidate"] += 1

                reader.close()

                print(f"Total tracks for file {ifile}: {total_tracks}")
                
                if (ifile - start + 1) % SAVE_EVERY == 0:
                    with CACHE.open("wb") as f:
                        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Checkpoint saved at file {ifile}")

            stats[window][option]["stau_mass_stats"] = mass_stats

            stats[window][option]["total_tracks"] = total_tracks
            stats[window][option]["pass_chi2"] = track_pass_chi2
            stats[window][option]["pass_eta"] = track_pass_eta
            stats[window][option]["pass_ob"] = track_pass_ob
            stats[window][option]["pass_pT_window"] = track_pass_gen_pT_window
            stats[window][option]["pass_wrms"] = track_pass_w_rms


            for mass, s in mass_stats.items():
                n_events = stats[window][option]["n_events"]

                A = s["A"]
                B = s["B"]
                C = s["C"]
                D = s["D"]

                A_pred = B * C / D if D > 0 else np.nan
                ratio = A / A_pred if A_pred > 0 else np.nan

                print(f"\n=== Stau mass {mass} TeV ===")
                print(f"Tracks passing baseline quality cuts: {s['quality']}")
                print(f"Tracks passing quality + wRMS < {WRMS_CUT}: {s['quality_wrms']}")
                print(f"Tracks passing quality + pT >= {STAU_CUTS[mass]['pt_min']}: {s['quality_pt']}")
                print(f"Tracks passing quality + beta window: {s['quality_beta']}")
                print(f"Candidate tracks: {s['candidate_tracks']}")

                print(f"Events with >=1 candidate track: {s['events_ge1_candidate']}")
                print(f"Events with >=2 candidate tracks: {s['events_ge2_candidate']}")

                print("ABCD:")
                print(f"A = pass wRMS, pass pT+beta: {A}")
                print(f"B = pass wRMS, fail pT+beta: {B}")
                print(f"C = fail wRMS, pass pT+beta: {C}")
                print(f"D = fail wRMS, fail pT+beta: {D}")
                print(f"A_pred = B*C/D = {A_pred}")
                print(f"A_obs / A_pred = {ratio}")

                if n_events > 0:
                    print(f"P(>=1 candidate/event) = {s['events_ge1_candidate'] / n_events}")
                    print(f"P(>=2 candidates/event) = {s['events_ge2_candidate'] / n_events}")

        print(f"Finished {window}")

        eta_percent = (track_pass_eta / total_tracks) * 100
        chi2_percent = (track_pass_chi2 / total_tracks) * 100
        ob_percent = (track_pass_ob / total_tracks) * 100
        highpt_percent = (track_pass_gen_pT_window / total_tracks) * 100
        rms_percent = (track_pass_w_rms / total_tracks) * 100

        print(f"{window} window stats:")
        print(f"Number of total tracks: {total_tracks}")
        print(f"Number of tracks passing eta cut: {track_pass_eta} (/ total tracks -> {eta_percent:.2f}%)")
        print(f"Number of tracks passing chi2 cut: {track_pass_chi2} (/ total tracks -> {chi2_percent:.2f}%)")
        print(f"Outer cut: {track_pass_ob} / {total_tracks} -> {ob_percent:.2f}%")
        print(f"Number of tracks passing  pT cut (1 Tev - 10TeV pT): {track_pass_gen_pT_window} (/ total tracks -> {highpt_percent:.2f}%)")
        print(f"Number of tracks passing 1.6 weighted rms max cut: {track_pass_w_rms} (/ total tracks -> {rms_percent:.2f}%)")

        print("\n")
        
        # NA = abcd_A
        # NB = abcd_B
        # NC = abcd_C
        # ND = abcd_D

        # NA_pred = NB * NC / ND if ND > 0 else np.nan

        # print("ABCD results:")
        # print(f"A observed pass pT, pass wrms: {NA}")
        # print(f"B fail pT, pass wrms: {NB}")
        # print(f"C pass pT, fail wrms: {NC}")
        # print(f"D fail pT, fail wrms: {ND}")
        # print(f"A predicted from ABCD = B*C/D = {NA_pred}")
        # print(f"A observed / predicted = {NA / NA_pred if NA_pred > 0 else np.nan}")

        # P_ge1_fake = events_ge1_fake / stats[window][option]["n_events"]
        # P_ge2_fake = events_ge2_fake / stats[window][option]["n_events"]

        # print("Event-level fake probabilities:")
        # print(f"Events with >=1 fake BIB track: {events_ge1_fake}")
        # print(f"Events with >=2 fake BIB tracks: {events_ge2_fake}")
        # print(f"P(>=1 fake per event) = {P_ge1_fake}")
        # print(f"P(>=2 fake per event) = {P_ge2_fake}")

    
#print(stats["loose"]["ob"]["10_bib"]["leading_w_rms"])
CACHE.parent.mkdir(exist_ok=True)
with CACHE.open("wb") as f:
    pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Writing final cache to {CACHE}")
print("Saved cache successfully.")

#print(stats["medium"]["bib"])


# for window in ["loose"]:
#     for option in bib_options:
#         total_tracks = stats[window][option]["total_tracks"]
#         track_pass_chi2 = stats[window][option]["pass_chi2"]
#         track_pass_eta = stats[window][option]["pass_eta"]
#         track_pass_ob = stats[window][option]["pass_ob"]
#         track_pass_pT_window = stats[window][option]["pass_pT_window"]
#         track_pass_w_rms = stats[window][option]["pass_wrms"]

#         eta_percent = (track_pass_eta / total_tracks) * 100
#         chi2_percent = (track_pass_chi2 / total_tracks) * 100
#         ob_percent = (track_pass_ob / total_tracks) * 100
#         highpt_percent = (track_pass_pT_window / total_tracks) * 100


#         print(f"{window} window stats:")
#         print(f"Number of total tracks: {total_tracks}")
#         print(f"Number of tracks passing eta cut: {track_pass_eta} (/ total tracks -> {eta_percent:.2f}%)")
#         print(f"Number of tracks passing chi2 cut: {track_pass_chi2} (/ total tracks -> {chi2_percent:.2f}%)")
#         print(f"Outer cut: {track_pass_ob} / {total_tracks} -> {ob_percent:.2f}%")
#         print(f"Number of tracks passing high pT cut (under 10TeV pT): {track_pass_pT_window} (/ total tracks -> {highpt_percent:.2f}%)")
#         print(f"Number of tracks passing 1.6 weighted rms max cut: {track_pass_w_rms} (/ total tracks -> {rms_percent:.2f}%)")




import pickle
import numpy as np

CACHE = "cache/100_bib_event_rejection_loose_stauwindows.pkl"

STAU_CUTS = {
    1.0:  {"pt_min": 1500, "beta_min": 0.95, "beta_max": 0.983},
    1.5:  {"pt_min": 2000, "beta_min": 0.90, "beta_max": 0.98},
    2.0:  {"pt_min": 2000, "beta_min": 0.85, "beta_max": 0.95},
    2.5:  {"pt_min": 2000, "beta_min": 0.80, "beta_max": 0.90},
    3.0:  {"pt_min": 2000, "beta_min": 0.70, "beta_max": 0.85},
    3.5:  {"pt_min": 2000, "beta_min": 0.65, "beta_max": 0.80},
    4.0:  {"pt_min": 1900, "beta_min": 0.55, "beta_max": 0.70},
    4.5:  {"pt_min": 1400, "beta_min": 0.40, "beta_max": 0.55},
}

WRMS_CUT = 1.6

with open(CACHE, "rb") as f:
    stats = pickle.load(f)

window = "loose"
option = "bib"

mass_stats = stats[window][option]["stau_mass_stats"]

for mass, s in mass_stats.items():

    n_events = stats[window][option]["n_events"]

    A = s["A"]
    B = s["B"]
    C = s["C"]
    D = s["D"]

    A_pred = B * C / D if D > 0 else np.nan
    ratio = A / A_pred if A_pred > 0 else np.nan

    print(f"\n=== Stau mass {mass} TeV ===")
    print(f"Tracks passing baseline quality cuts: {s['quality']}")
    print(f"Tracks passing quality + wRMS < {WRMS_CUT}: {s['quality_wrms']}")
    print(f"Tracks passing quality + pT >= {STAU_CUTS[mass]['pt_min']}: {s['quality_pt']}")
    print(f"Tracks passing quality + beta window: {s['quality_beta']}")
    print(f"Candidate tracks: {s['candidate_tracks']}")

    print(f"Events with >=1 candidate track: {s['events_ge1_candidate']}")
    print(f"Events with >=2 candidate tracks: {s['events_ge2_candidate']}")

    print("ABCD:")
    print(f"A = pass wRMS, pass pT+beta: {A}")
    print(f"B = pass wRMS, fail pT+beta: {B}")
    print(f"C = fail wRMS, pass pT+beta: {C}")
    print(f"D = fail wRMS, fail pT+beta: {D}")
    print(f"A_pred = B*C/D = {A_pred}")
    print(f"A_obs / A_pred = {ratio}")

    if n_events > 0:
        print(f"P(>=1 candidate/event) = {s['events_ge1_candidate'] / n_events}")
        print(f"P(>=2 candidates/event) = {s['events_ge2_candidate'] / n_events}")

        lambda_per_event = A_pred / n_events if np.isfinite(A_pred) else np.nan

        P_ge1_poisson = 1 - np.exp(-lambda_per_event)
        P_ge2_poisson = 1 - np.exp(-lambda_per_event) * (1 + lambda_per_event)

        print(f"Poisson lambda/event from ABCD A_pred = {lambda_per_event}")
        print(f"Poisson P(>=1 fake candidate/event) = {P_ge1_poisson}")
        print(f"Poisson P(>=2 fake candidates/event) = {P_ge2_poisson}")