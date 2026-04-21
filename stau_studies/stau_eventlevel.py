import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# DEFINING CUTS 
ETA_MAX     = 0.8
CHI2NDF_MAX = 3.0
OB_HITS_MIN = 2
PT_MAX      = 10_000.0   
NO_INTERCEPT = True  

def track_eta_from_tanL(tanL):
    if not np.isfinite(tanL):
        return np.nan
    return float(np.arcsinh(tanL))

def pass_ob_req(t):
    return (t["vb_hits"] >= 3) and (t["ib_hits"] >= 2) and (t["ob_hits"] >= 2)

def pass_track_level(t):
    return (
        pass_ob_req(t) and
        np.isfinite(t["eta"]) and (abs(t["eta"]) < ETA_MAX) and
        np.isfinite(t["chi2ndf"]) and (t["chi2ndf"] < CHI2NDF_MAX) and
        np.isfinite(t["pT"]) and (t["pT"] < PT_MAX) and
        np.isfinite(t["beta"]) and (0 < t["beta"] < 1) and
        np.isfinite(t["mass"])
    )

hit_collections = ["VXDBarrelHits", "VXDEndcapHits", "ITBarrelHits", "ITEndcapHits", "OTBarrelHits", "OTEndcapHits"]
sim_collections = ["VertexBarrelCollection", "VertexEndcapCollection", "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection", "OuterTrackerBarrel", "OuterTrackerEndcapCollectionConed"]
rel_collections = ["VXDBarrelHitsRelations", "VXDEndcapHitsRelations", "ITBarrelHitsRelations", "ITEndcapHitsRelations", "OTBarrelHitsRelations", "OTEndcapHitsRelations"]

loose_dir =  "/ospool/uc-shared/project/futurecolliders/miralittmann/reco/reder_timing/loose4/seeding_10GeV/nobib"
window_to_dir = {"loose": loose_dir}
n_files = 2500

CACHE = pathlib.Path("cache/stau_leadingsub.pkl") #-nozero

plot_path = "/scratch/wandriscok/kate_mucoll_scripts/stau_studies/pdf/stau_leadingsub.pdf"
track_stats_plot_path = "/scratch/wandriscok/kate_mucoll_scripts/stau_studies/pdf/stau_plot.pdf" # -nozero

sample_to_mass = {
    "1000": 1.0,
    "1500": 1.5,
    "2000": 2.0,
    "2500": 2.5,
    "3000": 3.0,
    "3500": 3.5,
    "4000": 4.0,
    "4500": 4.5
}
mass_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
# lifetimes = [3, 10, 30]
lifetimes = [30]

stau_ids = {1000015, -1000015, 2000015, -2000015} 
Bfield = 3.57
chi2_cut = 3
nhits_cut = 4
speedoflight = 299792458/1000000  # mm/ns
guess_velo = 180

def binom_se(p, n):
    # Gaussian SE on a proportion
    if n <= 0:
        return float('nan')
    return math.sqrt(p*(1-p)/n)

parser = argparse.ArgumentParser()
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()
rebuild = args.rebuild

def iter_events(reader): 
    while True:
        try:
            evt = reader.readNextEvent() 
        except Exception as e:
            break
        if evt is None:
            break
        yield evt
def safe_get(event, name):
    try:
        return event.getCollection(name)
    except Exception:
        return None

def acceptance(mcp):
    mcp_stau_vertex_r = np.sqrt(mcp.getVertex()[0]**2 + mcp.getVertex()[1]**2)
    travel_dist = np.sqrt(mcp.getEndpoint()[0]**2 + mcp.getEndpoint()[1]**2 + mcp.getEndpoint()[2]**2)

    mcp_stau_momentum = mcp.getMomentum() 
    mcp_stau_tlv = ROOT.TLorentzVector()
    mcp_stau_tlv.SetPxPyPzE(mcp_stau_momentum[0], mcp_stau_momentum[1], mcp_stau_momentum[2], mcp.getEnergy())  

    if abs(mcp.getPDG()) not in stau_ids or travel_dist==0 or mcp_stau_vertex_r>553.0 or abs(mcp_stau_tlv.Eta()) > 0.8: # or mcp_stau_endpoint_r < 1486.0 
        accepted = False
    else: 
        accepted = True
    
    return(accepted)

def linearfunc(p, x):
    # p[0] = velocity [mm/ns], p[1] = intercept [mm]
    return p[0] * x + p[1]

def residual(p, function_type, times, pos, spatial_unc):
    # weighted residuals
    return (function_type(p, times) - pos) / spatial_unc

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

    rms_t = float(np.sqrt(np.mean(dt * dt)))          
    rms_pull = float(np.sqrt(np.mean((dt / st) ** 2))) 
    return rms_t, rms_pull


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

    rms_t, rms_pull = time_rms_from_fit(v, x, y, st, b=0.0)

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

    return v, v_err, rms_t, rms_pull


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

system_to_relname = {
    1: "VXDBarrel", 2: "VXDEndcap",
    3: "ITBarrel",  4: "ITEndcap",
    5: "OTBarrel",  6: "OTEndcap",
}

track_reqs = ["vb", "ib", "ob"]

efficiencies = None
if (not rebuild) and os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        print("Loading in cached arrays...")
        efficiencies = pickle.load(f)

if efficiencies is None:

    efficiencies = {
        lifetime: {
            sample: {
                    "n_events": 0,
                    "leading_mass": [],
                    "subleading_mass": [],
                    "leading_pT": [],
                    "subleading_pT": [],
                    "leading_beta": [],
                    "subleading_beta": [],
                    "leading_d0": [],
                    "subleading_d0": [],
                    "leading_z0": [],
                    "subleading_z0": [],   
                    "leading_vrmsw": [],
                    "subleading_vrmsw": [] 
            } for sample in sample_to_mass.keys()
        } for lifetime in lifetimes
    } 


    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()

    for lifetime in lifetimes: 
        print(f"                      {lifetime}...")
        for sample in sample_to_mass.keys():
            print(f"                        {sample}...")
            save_info = efficiencies[lifetime][sample]

            for ifile in tqdm(range(n_files)):
                masses = []
                file_name = f"{sample}_{lifetime}/{sample}_{lifetime}_reco{ifile}.slcio"
                file_path = os.path.join(loose_dir,file_name) 
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    print(f"couldn't find {file_path}")
                    continue
                reader.open(file_path)
                for event in reader:
                    tracks_by_req = {req: [] for req in track_reqs}
                    n_acc_staus = 0

                    all_collections = event.getCollectionNames() 
                    mcp_collection = event.getCollection("MCParticle") if "MCParticle" in all_collections else None
                    track_collection = event.getCollection("SiTracks") if "SiTracks" in all_collections else None # REFITTED
                    track_relation_collection = event.getCollection("MCParticle_SiTracks") if "MCParticle_SiTracks" in all_collections else None # REFITTED

                    if not (mcp_collection and track_collection and track_relation_collection):
                        print("issue 1")
                        continue

                    test_hit_coll = event.getCollection("VXDBarrelHits")
                    if test_hit_coll is None:
                        print("no hit collections")
                        continue
                    encoding = test_hit_coll.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                    decoder = UTIL.BitField64(encoding)

                    nav = UTIL.LCRelationNavigator(track_relation_collection)
                    masses_by_req = {req: [] for req in track_reqs}

                    for mcp in mcp_collection:
                        pdg = mcp.getPDG()
                        is_stau = (abs(pdg) in stau_ids)

                        if not is_stau:
                            continue

                        vx = mcp.getVertex()
                        ep = mcp.getEndpoint()

                        r_decay = np.sqrt(ep[0]**2 + ep[1]**2)
                        z_decay = ep[2]



                        related_tracks = nav.getRelatedToObjects(mcp)

                        for track in related_tracks:

                            track_chi2 = track.getChi2() / track.getNdf()
                            if track_chi2 > chi2_cut:
                                continue

                            reco_pT = 0.3 * Bfield / fabs(track.getOmega() * 1000.)

                            vb_hits = 0
                            ib_hits = 0
                            ob_hits = 0

                            track_hits = track.getTrackerHits()

                            track_times = []
                            track_pos = []
                            spatial_unc = []
                            time_unc = []

                            d0 = track.getD0()
                            z0 = track.getZ0()

                            for hit in track_hits:
                                decoder.setValue(int(hit.getCellID0()))
                                system = decoder["system"].value()

                                if system in (1, 2):
                                    vb_hits += 0.5
                                    spatial_unc.append(0.005)
                                    time_unc.append(0.03)
                                elif system in (3, 4):
                                    ib_hits += 1
                                    spatial_unc.append(0.007)
                                    time_unc.append(0.06)
                                elif system in (5, 6):
                                    ob_hits += 1
                                    spatial_unc.append(0.007)
                                    time_unc.append(0.06)
                            
                                hit_time = hit.getTime()
                                x = hit.getPosition()[0]
                                y = hit.getPosition()[1]
                                z = hit.getPosition()[2]
                                hit_pos = np.sqrt(x**2 + y**2 + z**2)
                                tof = hit_pos/speedoflight

                                corrected_t = hit.getTime() - tof
                                corrected_corrected_t = 2*hit.getTime() - corrected_t

                                track_times.append(corrected_corrected_t)
                                track_pos.append(hit_pos)

                            if NO_INTERCEPT:
                                v_fit, v_err, rms_uw, rms_w = reco_velo_no_intercept(track_times, track_pos, spatial_unc, time_unc)
                            else:
                                v_fit, v_err, rms_uw, rms_w = reco_velo(linearfunc, track_times, track_pos, spatial_unc, time_unc)

                            total_hits = vb_hits + ib_hits + ob_hits

                            try:
                                tanL = track.getTanLambda()
                            except Exception:
                                tanL = np.nan

                            if np.isfinite(tanL):
                                reco_pz = reco_pT * tanL
                                reco_p  = math.sqrt(reco_pT**2 + reco_pz**2)
                            else:
                                reco_p  = np.nan
                            beta = v_fit / speedoflight if np.isfinite(v_fit) else np.nan
                            
                            if np.isfinite(reco_p) and np.isfinite(beta) and (0 < beta < 1):
                                m_reco = reco_p * math.sqrt(1.0/(beta*beta) - 1.0)
                            else:
                                m_reco = np.nan

                            try:
                                tanL = track.getTanLambda()
                            except Exception:
                                tanL = np.nan
                            eta = float(np.arcsinh(tanL)) if np.isfinite(tanL) else np.nan

                            track_info = {
                                "pT": float(reco_pT) if np.isfinite(reco_pT) else np.nan,
                                "beta": float(beta) if np.isfinite(beta) else np.nan,
                                "mass": float(m_reco) if np.isfinite(m_reco) else np.nan,
                                "d0": float(d0) if np.isfinite(d0) else np.nan,
                                "z0": float(z0) if np.isfinite(z0) else np.nan,
                                "vrmsw": float(rms_w) if np.isfinite(rms_w) else np.nan,
                                "chi2ndf": float(track.getChi2()/track.getNdf()) if track.getNdf() > 0 else np.nan,
                                "vb_hits": vb_hits,
                                "ib_hits": ib_hits,
                                "ob_hits": ob_hits,
                                "eta": eta,
                            }


                            if vb_hits >= 3 and ib_hits >= 2 and ob_hits >= 2:
                                tracks_by_req["ob"].append(track_info)
                                

                    trks = tracks_by_req["ob"]
                    passing = [t for t in trks if pass_track_level(t)]
                    passing.sort(key=lambda t: t["pT"], reverse=True)  
                    
                    save_info["n_events"] += 1

                    if len(passing) >= 1:
                        lead = passing[0]
                        save_info["leading_pT"].append(lead["pT"])
                        save_info["leading_beta"].append(lead["beta"])
                        save_info["leading_mass"].append(lead["mass"])
                        save_info["leading_d0"].append(lead["d0"])
                        save_info["leading_z0"].append(lead["z0"])
                        save_info["leading_vrmsw"].append(lead["vrmsw"])
                    else:
                        save_info["leading_pT"].append(np.nan)
                        save_info["leading_beta"].append(np.nan)
                        save_info["leading_mass"].append(np.nan)
                        save_info["leading_d0"].append(np.nan)
                        save_info["leading_z0"].append(np.nan)
                        save_info["leading_vrmsw"].append(np.nan)

                    if len(passing) >= 2:
                        sub = passing[1]
                        save_info["subleading_pT"].append(sub["pT"])
                        save_info["subleading_beta"].append(sub["beta"])
                        save_info["subleading_mass"].append(sub["mass"])
                        save_info["subleading_d0"].append(sub["d0"])
                        save_info["subleading_z0"].append(sub["z0"])
                        save_info["subleading_vrmsw"].append(sub["vrmsw"])
                    else:
                        save_info["subleading_pT"].append(np.nan)
                        save_info["subleading_beta"].append(np.nan)
                        save_info["subleading_mass"].append(np.nan)
                        save_info["subleading_d0"].append(np.nan)
                        save_info["subleading_z0"].append(np.nan)
                        save_info["subleading_vrmsw"].append(np.nan)
               
                reader.close()

    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(efficiencies, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Writing cache to {CACHE}")
    print("Saved cache successfully.")

print(efficiencies.keys())

labels = {"pT": r"$p_T$ [GeV]",
          "hits": "Hits on track",
          "velo": "Velocity [mm/ns]",
          "leading_mass": "Leading reconstructed mass [GeV]",
          "subleading_mass": "Subleading reconstructed mass [GeV]",
          "leading_d0": "Leading D0",
          "subleading_d0": "Subleading D0",
          "leading_z0": "Leading Z0",
          "subleading_z0": "Subleading Z0",
          }


def summarize_array(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "N": 0,
            "min": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "max": np.nan
        }
    return {
        "N": int(a.size),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(np.max(a)),
    }

def print_stau_vrms_summaries(efficiencies, lifetimes, sample_to_mass, reqs=("ob",)):
    samples_sorted = [
        s for s in sorted(sample_to_mass.keys(), key=lambda s: sample_to_mass[s])
        if sample_to_mass[s] in [1.0, 2.0, 3.0, 4.0]
    ]

    for lifetime in lifetimes:
        print("\n" + "="*120)
        print(f"STAU RMS summaries | lifetime = {lifetime} ns")
        print("="*120)

        for req in reqs:
            print("\n" + "-"*120)
            print(f"req = {req}")
            print("-"*120)

            for sample in samples_sorted:
                mtev = sample_to_mass[sample]
                d = efficiencies[lifetime][sample][req]

                lead_w = summarize_array(d.get("leading_vrmsw", []))
                sub_w  = summarize_array(d.get("subleading_vrmsw", []))
                lead_u = summarize_array(d.get("leading_vrms_mm", []))
                sub_u  = summarize_array(d.get("subleading_vrms_mm", []))

                print(f"\nSample {sample}  (m = {mtev:.1f} TeV)")
                print("  Weighted RMS (vrmsw):")
                print(f"    Leading   N finite: {lead_w['N']}")
                print(f"    Leading   min/median/90%/max: {lead_w['min']} {lead_w['median']} {lead_w['p90']} {lead_w['max']}")
                print(f"    Sublead   N finite: {sub_w['N']}")
                print(f"    Sublead   min/median/90%/max: {sub_w['min']} {sub_w['median']} {sub_w['p90']} {sub_w['max']}")

                print("  Unweighted RMS (vrms_mm) [mm]:")
                print(f"    Leading   N finite: {lead_u['N']}")
                print(f"    Leading   min/median/90%/max: {lead_u['min']} {lead_u['median']} {lead_u['p90']} {lead_u['max']}")
                print(f"    Sublead   N finite: {sub_u['N']}")
                print(f"    Sublead   min/median/90%/max: {sub_u['min']} {sub_u['median']} {sub_u['p90']} {sub_u['max']}")

# print_stau_vrms_summaries(efficiencies, lifetimes, sample_to_mass, reqs=("ob",))


STAU_CACHE = pathlib.Path("cache/stau_leadingsub.pkl")
BIB_CACHE  = pathlib.Path("/scratch/wandriscok/kate_mucoll_scripts/bib_analysis/cache/bib_event_plot_lead_sub_loose.pkl")
MUON_CACHE = pathlib.Path("/scratch/miralittmann/analysis/mira_analysis_code/cache/mumu_bkg_stats_nominal_nobib_byevent.pkl")

with STAU_CACHE.open("rb") as f:
    efficiencies = pickle.load(f)

with BIB_CACHE.open("rb") as f:
    bib_stats = pickle.load(f)

with MUON_CACHE.open("rb") as f:
    muon_stats = pickle.load(f)


def _shape_norm_hist(ax, arr, bins, label,
                    linestyle="-", linewidth=2,
                    color=None, fill=False, alpha=1.0,
                    edgecolor=None, include_flow=True):

    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    # put underflow into first bin and overflow into last bin
    if include_flow:
        eps = 1e-9
        x = np.clip(x, bins[0] + eps, bins[-1] - eps)
    
    # normalize so total area = 100%
    w = np.full(x.shape, 100.0 / x.size, dtype=float)

    if fill:
        ax.hist(
            x, bins=bins, weights=w,
            histtype="stepfilled",
            color=color, alpha=alpha,
            edgecolor=edgecolor if edgecolor is not None else color, 
            linewidth=linewidth,
            label=label
        )
    else:
        ax.hist(
            x, bins=bins, weights=w,
            histtype="step",
            color=color, linestyle=linestyle,
            linewidth=linewidth,
            label=label
        )

    return 100.0  # total area


def plot_leading_all_masses_with_bib(
    efficiencies,
    bib_stats,
    muon_stats,
    lifetimes,
    sample_to_mass,
    out_pdf,
    window="loose",
    req="ob",
    bib_options=("bib"),   # could also do ("10_bib", "bib")
    bins_cfg=None,
    xlims_cfg=None
):
    if bins_cfg is None:
        bins_cfg = {
            "mass":  np.linspace(0, 5500, 61),
            "pT":    np.linspace(0, 7000, 61),
            "beta":  np.linspace(0.4, 1.05, 53),
            "d0":    np.linspace(-0.01, 0.01, 51),
            "z0":    np.linspace(-0.01, 0.01, 51),
            "vrmsw": np.linspace(0, 2, 51),
        }

    if xlims_cfg is None:
        xlims_cfg = {
            "mass":  (0, 5500),
            "pT":    (0, 7000),
            "beta":  (0.4, 1.05),
            "d0":    (-0.01, 0.01),
            "z0":    (-0.01, 0.01),
            "vrmsw": (0, 2),
        }

    stau_features = [
        ("mass",  "leading_mass",  r"Leading reconstructed mass [GeV]"),
        ("pT",    "leading_pT",    r"Leading $p_T$ [GeV]"),
        ("beta",  "leading_beta",  r"Leading $\beta$"),
        ("d0",    "leading_d0",    r"Leading $d_0$"),
        ("z0",    "leading_z0",    r"Leading $z_0$"),
        ("vrmsw", "leading_vrmsw", r"Leading weighted RMS residual"),
    ]

    bib_key_map = {
        "leading_mass": "leading_mass",
        "leading_pT": "leading_pT",
        "leading_beta": "leading_beta",
        "leading_d0": "leading_d0",
        "leading_z0": "leading_z0",
        "leading_vrmsw": "leading_w_rms",   # note the name difference
    }

    muon_key_map = {
        "leading_mass": "leading_mass",
        "leading_pT": "leading_pT",
        "leading_beta": "leading_beta",
        "leading_d0": "leading_d0",
        "leading_z0": "leading_z0",
        "leading_vrmsw": "leading_wvrms",   # note the name difference
    }

    samples_sorted = [
        s for s in sorted(sample_to_mass.keys(), key=lambda s: sample_to_mass[s])
        if sample_to_mass[s] in [1.0, 2.0, 3.0, 4.0]
    ]

    with PdfPages(out_pdf) as pdf:
        for lifetime in lifetimes:
            for feat_key, stau_data_key, xlabel in stau_features:
                fig, ax = plt.subplots(figsize=(8, 6))

                # --- stau curves ---
                for sample in samples_sorted:
                    mtev = sample_to_mass[sample]
                    d = efficiencies[lifetime][sample]

                    arr = d.get(stau_data_key, [])
                    N_events = len(arr)
                    if N_events == 0:
                        continue

                    _shape_norm_hist(
                        ax,
                        arr,
                        bins_cfg[feat_key],
                        label=fr"{mtev:.1f} TeV"
                    )

                # --- BIB curves ---
                bib_data_key = bib_key_map[stau_data_key]
                for option in bib_options:
                    d_bib = bib_stats[window][req][option]
                    arr_bib = d_bib.get(bib_data_key, [])
                    N_events_bib = d_bib.get("n_events", len(arr_bib))

                    _shape_norm_hist(
                        ax,
                        arr_bib,
                        bins_cfg[feat_key],
                        label="10% BIB background",
                        color="grey",
                        fill=True,
                        alpha=0.3
                    )

                # --- mu+mu- background first: faded blue fill ---
                muon_data_key = muon_key_map[stau_data_key]
                arr_mu = muon_stats.get(muon_data_key, [])

                _shape_norm_hist(
                    ax,
                    arr_mu,
                    bins_cfg[feat_key],
                    label=r"Nominal mu-mu bkg",
                    color="cornflowerblue",
                    fill=True,
                    alpha=0.22,
                    edgecolor=None
                )
                
                ax.set_xlabel(xlabel, fontsize=16)
                ax.set_ylabel("Normalized counts (%)", fontsize=16)
                #ax.legend(frameon=True, fontsize=10, ncol=2)
                ax.legend(loc="upper right", frameon=True, fontsize=10, ncol=2)

                if feat_key in xlims_cfg:
                    ax.set_xlim(*xlims_cfg[feat_key])

                ax.grid(True, alpha=0.2)

                ax.text(
                    0.02, 0.98, "Muon Collider",
                    ha="left", va="top",
                    transform=ax.transAxes,
                    fontsize=16, fontweight="bold", style="italic"
                )
                ax.text(
                    0.02, 0.93,
                    f"Leading tracks only, τ = {lifetime} ns",
                    ha="left", va="top",
                    transform=ax.transAxes, fontsize=12
                )
                ax.text(
                    0.02, 0.88,
                    f"BIB: {window} window, req = {req}",
                    ha="left", va="top",
                    transform=ax.transAxes, fontsize=12
                )

                ax.tick_params(axis="both", which="major", labelsize=14)
                ax.tick_params(axis="both", which="minor", labelsize=12)

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved plots to {out_pdf}")



plot_leading_all_masses_with_bib(
    efficiencies=efficiencies,
    bib_stats=bib_stats,
    muon_stats=muon_stats,
    lifetimes=lifetimes,
    sample_to_mass=sample_to_mass,
    out_pdf="/scratch/wandriscok/kate_mucoll_scripts/stau_studies/pdf/stau_leading_allmasses_with_bib.pdf",
    window="loose",
    req="ob",
    bib_options=("10_bib",)
)


for key in ["leading_d0", "leading_z0", "leading_w_rms"]:
    arr = np.asarray(bib_stats["loose"]["ob"]["10_bib"][key], dtype=float)
    arr = arr[np.isfinite(arr)]
    print(key, len(arr), np.min(arr), np.max(arr), np.median(arr))