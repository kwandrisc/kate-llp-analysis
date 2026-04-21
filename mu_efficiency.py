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
from array import array

dirs = {"bib": "/ospool/uc-shared/project/futurecolliders/miralittmann/reco/mumu_bkg/bib/",
        "nobib": "/ospool/uc-shared/project/futurecolliders/miralittmann/reco/mumu_bkg/nobib/"}
option = ["nobib"]
windows = ["loose"]
track_req_names = ["vb", "ib", "ob"]
CACHE = pathlib.Path("cache/mu_efficiency.pkl")
SAVE_EVERY_FILES = 50  
n_files = 1000
Bfield = 3.57
speedoflight = 299792458/1000000  # mm/ns
chi2_cut = 3

def save_cache_atomic(stats, cache_path):
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(exist_ok=True)
    tmp = str(cache_path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache_path)

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


def passes_hits(track):
    hits = track.getTrackerHits()
    vb = ib = ob = 0

    for hit in hits:
        decoder.setValue(int(hit.getCellID0()))
        system = decoder["system"].value()

        if system in (1,2):
            vb += 1
        elif system in (3,4):
            ib += 1
        elif system in (5,6):
            ob += 1

    return (vb >= 3 and ib >= 2 and ob >= 2)


efficiencies = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        efficiencies = pickle.load(f)

if efficiencies is None:
    efficiencies = {
        "n_events": 0,
        "last_file": -1,

        "true_muons": 0,
        "matched_muons": 0,
        "muon_efficiency": 0
    }
    

    start_file = int(efficiencies.get("last_file", -1)) + 1
    print(f"Resuming from file {start_file}")
         
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    
    n_true_muons = 0
    n_matched_muons = 0
    for ifile in tqdm(range(start_file, n_files)): 
        file_name = f"mumu_bkg_reco{ifile}.slcio"
        file_path = os.path.join(dirs["nobib"], "nominal", file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"couldn't open {file_path}")
            continue
        reader.open(file_path)
        for event in reader:
            efficiencies["n_events"] += 1
            
            tracks_by_req = {req: [] for req in track_req_names}
            all_collections = event.getCollectionNames() 
            mcp_collection = event.getCollection("MCParticle") if "MCParticle" in all_collections else None
            track_collection = event.getCollection("SiTracks") if "SiTracks" in all_collections else None 
            track_relation_collection = event.getCollection("MCParticle_SiTracks") if "MCParticle_SiTracks" in all_collections else None 
            if not (mcp_collection and track_collection and track_relation_collection):
                print("issue 1")
                continue
            test_hit_coll = event.getCollection("VXDBarrelHits")
            if test_hit_coll is None:
                continue
            encoding = test_hit_coll.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            decoder = UTIL.BitField64(encoding)
            nav = UTIL.LCRelationNavigator(track_relation_collection)

            if mcp_collection is not None:
                for mcp in mcp_collection:
                    # selecting muons
                    if abs(mcp.getPDG()) != 13:
                        continue

                    momentum = mcp.getMomentum()
                    tlv = ROOT.TLorentzVector()
                    tlv.SetPxPyPzE(momentum[0], momentum[1], momentum[2], mcp.getEnergy())
                    if abs(tlv.Eta()) > 0.8:
                        continue

                    ## adding detector requirements?
                    # if vb_hits >= 3 and ib_hits >= 2 and ob_hits >=2:
                    #     tracks_by_req["ob"].append(track_info)

                    # adding other things like being good chi2?
                    
                    n_true_muons += 1
                    efficiencies["true_muons"] += 1
                    matched_tracks = nav.getRelatedToObjects(mcp)
                    
                    if matched_tracks:
                        n_matched_muons += 1
                        efficiencies["matched_muons"] += 1
                    
                    # good_tracks = [track for track in matched_tracks 
                    #                if (track.getChi2() / track.getNdf()) < chi2_cut
                    #                and passes_hits(track)]
                    
                    # if len(good_tracks) > 0:
                    #     n_matched_muons += 1
                    #     efficiencies["matched_muons"] += 1

        reader.close()
        efficiencies["last_file"] = ifile

        if (ifile % SAVE_EVERY_FILES) == 0:
            save_cache_atomic(efficiencies, CACHE)
            print(f"[checkpoint] saved at file {ifile}, n_events={efficiencies['n_events']}")

    efficiency = n_matched_muons / n_true_muons if n_true_muons > 0 else 0
    efficiencies["muon_efficiency"] = efficiency
       
    save_cache_atomic(efficiencies, CACHE)
    print(f"Writing cache to {CACHE}")

print(efficiencies)
