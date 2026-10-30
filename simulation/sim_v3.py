# Generate an SPE5 WAG dataset with sampled train/test reservoir families.

import os, re, sys, glob, time, shutil, subprocess, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

SEED = 42
np.random.seed(SEED)

HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_SRC  = os.path.join(HERE, "SPE5.BASE")
WORK      = os.path.join(HERE, "spe5_wag_runs_v3")
OUT_CSV   = os.path.join(HERE, "WAG_simulation_dataset_FULL.csv")

# Select the reservoir family to generate: B is training; A is external testing.
VARIANT     = os.environ.get("WAG_VARIANT", "B").upper()
if VARIANT not in ("A", "B"):
    raise ValueError("WAG_VARIANT must be A or B")
DATASET_ROLE = "training" if VARIANT == "B" else "external test"
N_SCENARIOS = int(os.environ.get("WAG_N", 5000 if VARIANT == "B" else 1000))
if N_SCENARIOS < 1:
    raise ValueError("WAG_N must be a positive integer")
OUT_CSV  = os.path.join(HERE, f"WAG_simulation_dataset_{VARIANT}.csv")   # one file per reservoir family
# Changing the LHS size changes every scenario; use a fresh workspace/checkpoint for that size.
WORK     = WORK.rstrip("/")+f"_{VARIANT}_{N_SCENARIOS}"

# Scenarios are independent and reproducible for a fixed seed.
N_WORKERS    = int(os.environ.get("WAG_WORKERS", 8))  # 8 single-threaded OPM processes per family
FLOW_THREADS = 1       # explicitly disable OPM multithreading; A and B together use 16 threads.
FLOW_TIMEOUT = 1800    # seconds allowed for one OPM run (low-permeability cases are slower)
CHECKPOINT   = os.path.join(HERE, f"step1_checkpoint_{VARIANT}_{N_SCENARIOS}.jsonl")
               # every finished scenario is appended here; an interrupted run resumes from it
FLOW_BIN    = shutil.which("flow") or "flow"

# Sampled inputs and their valid ranges.
RANGES = {
    "perm_mult"               : (0.007, 5.0),    # effective perm 1.75 - 1250 mD (log-uniform); field projects 2-230 mD
    "porosity_frac"           : (0.07, 0.30),    # field projects 7.6 - 26.4 %
    "pressure_gradient_psi_ft": (0.30, 0.55),    # initial pressure = gradient x depth (field projects 0.26 - 0.58 psi/ft)
    "depth_ft"                : (1700., 11500.), # field projects 1,750 - 11,100 ft
    # Injection is specified the way field projects report it, and the rates are derived from it:
    "hcpvi_target"            : (0.05, 1.5),     # gas injected over the WAG period, fraction of HCPV (log-uniform).
                                                 # Karacan 2020: 6-88 %; Belazreg 2020 Fig. 2: up to ~100 %;
                                                 # field evaluation set: 6-88 %
    "wag_ratio_target"        : (0.4, 4.0),      # water:gas reservoir-volume ratio (log-uniform).
                                                 # Christensen 2001: "initially 1 in most cases", up to 3-4;
                                                 # Belazreg 2020: 0.5-3; field evaluation set: 0.5-3
    "prod_orat_stb"           : (8000., 16000.), # 'PROD' oil-rate target (no field equivalent)
    "wag_halfcycle_mo"        : (3, 12),         # months per half-cycle
    "waterflood_years"        : (4, 10),         # years of waterflooding before WAG starts (tertiary WAG);
                                                 # sets how mature the waterflood is when gas injection begins
}
# Sampled SPE5 properties for the two reservoir families.
FAMILY = {
    # Only two formerly fixed SPE5 properties differ between the families, and only slightly (ranges overlap).
    #                        family A (test)    family B (train)
    "perm_contrast_exp": {"A": (0.8, 1.2),   "B": (1.0, 1.4)},    # layer k = SPE5 shape (500/50/200)^exp, 1 = SPE5
    "oil_visc_mult":     {"A": (0.8, 1.5),   "B": (1.2, 2.0)},    # x SPE5 oil viscosity (about 0.2 cp)
    # kept at the original SPE5 values in both families (listed so the deck edits stay explicit):
    "kv_kh_mult":        {"A": (1.0, 1.0),   "B": (1.0, 1.0)},    # SPE5 kv/kh per layer (0.1, 1, 0.125)
    "thickness_mult":    {"A": (1.0, 1.0),   "B": (1.0, 1.0)},    # SPE5 layers 20/30/50 ft
    "cell_size_ft":      {"A": (500., 500.), "B": (500., 500.)},  # SPE5 DX = DY = 500 ft
    "mmp_full_psi":      {"A": (3200., 3200.), "B": (3200., 3200.)}, # SPE5 PMISC: 1500 -> 3200 psi
    "tl_omega":          {"A": (0.6, 0.6),   "B": (0.6, 0.6)},    # SPE5 Todd-Longstaff mixing
}
for _k, _v in FAMILY.items():
    RANGES[_k] = _v[VARIANT]
MISC_WIDTH_PSI = 1700.0   # immiscible below (full - width); SPE5: 1500 -> 3200
SPE5_LAYER_K   = np.array([500.0, 50.0, 200.0])
SPE5_KV_RATIO  = np.array([0.1, 1.0, 0.125])       # SPE5 PERMZ / PERMX per layer
SPE5_DZ        = np.array([20.0, 30.0, 50.0])

PRIMARY_YEARS = 2    # primary depletion, both injectors shut
WAG_YEARS     = 10   # length of the comparison period after the waterflood (WAG branch vs continued waterflood)
# sampled uniformly in log10 space so that low values are covered as well as high ones
LOG_VARS = {"perm_mult", "hcpvi_target", "wag_ratio_target"}
LOG_VARS |= {"kv_kh_mult", "oil_visc_mult"}   # formerly fixed SPE5 properties sampled in log space
# initial pressure is kept inside the SPE5 PVT tables (PVDS/PVTO end at 4,800 psia)
P_MIN, P_MAX = 700., 5500.

# Fixed SPE5 inputs, exported as constant columns.
FIXED = {
    "process_type"          : "Miscible",   # SPE5 is a miscible flood
    "injectant"             : "SPE5 solvent",        # solvent gas (WSOLVENT=1.0)
    "temperature_F"         : 160.0,        # SPE5 isothermal reservoir temperature (approx)
         # approx API of the SPE5 volatile oil
    "grid_nx"               : 7,
    "grid_ny"               : 7,
    "grid_nz"               : 3,
    "reservoir_family"      : VARIANT,
    "producer_cell"         : "7,7,3",
    "injector_cell"         : "1,1,1",
    "solvent_fraction"      : 1.0,          # WSOLVENT INJG = 1.0 (fully miscible solvent)
    "primary_years"         : 2,
    "wag_years"             : 10,
}

def lhs(n, ranges, seed=None):
    seed = (SEED if VARIANT == 'A' else SEED + 1000) if seed is None else seed
    rng=np.random.RandomState(seed); keys=list(ranges); d=len(keys)
    cut=np.linspace(0,1,n+1); u=rng.rand(n,d)
    pts=cut[:n][:,None]+u*(cut[1:]-cut[:n])[:,None]
    for j in range(d): rng.shuffle(pts[:,j])
    out={}
    for j,k in enumerate(keys):
        lo,hi=ranges[k]
        if k in LOG_VARS:
            out[k]=10**(np.log10(lo)+pts[:,j]*(np.log10(hi)-np.log10(lo)))
        else:
            out[k]=lo+pts[:,j]*(hi-lo)
    df=pd.DataFrame(out)
    df["wag_halfcycle_mo"]=df["wag_halfcycle_mo"].round().astype(int)  # integer months
    df["waterflood_years"]=df["waterflood_years"].round().astype(int)  # integer years
    # physically consistent pressure: tied to depth instead of sampled independently
    df["init_pressure_psi"]=(df["pressure_gradient_psi_ft"]*df["depth_ft"]).clip(P_MIN,P_MAX)
    return df

BASE_TXT = open(BASE_SRC).read()

def base_with_field_summary(base):
    field_vars = "FOPT\nFOPR\nFOIP\nFGIP\nFPR\nFWCT\nFGOR\nFGIT\nFNIT\nFWIT\n"
    return re.sub(r"(^SUMMARY\s*\n)", r"\1" + field_vars, base, count=1, flags=re.M)

MONTHS = [31,28,31,30,31,30,31,31,30,31,30,31]
def tstep_line(n_months):
    """Return a TSTEP list covering n_months (repeating the 12-month pattern)."""
    days=[]
    for k in range(n_months):
        days.append(str(MONTHS[k % 12]))
    return " ".join(days)

def make_case(i, row, d, mode="wag"):
    """mode='wag' builds the WAG deck (CASE.DATA); mode='waterflood' builds the
    baseline waterflood deck (BASE.DATA) used to compute the INCREMENTAL recovery."""
    os.makedirs(d, exist_ok=True)
    base = BASE_TXT
    m = row["perm_mult"]

    # Scale the three layer permeabilities.
    def scale_perm(txt, kw, vals):
        new=f"{kw}\n\t49*{vals[0]*m:.4g} 49*{vals[1]*m:.4g} 49*{vals[2]*m:.4g} /"
        return re.sub(rf"{kw}\s*\n[^/]*/", new, txt, count=1)
    shape = (SPE5_LAYER_K/SPE5_LAYER_K.mean())**row["perm_contrast_exp"]
    kx = SPE5_LAYER_K.mean()*shape/shape.mean()
    kz = kx*SPE5_KV_RATIO*row["kv_kh_mult"]
    base=scale_perm(base,"PERMX",list(kx))
    base=scale_perm(base,"PERMY",list(kx))
    base=scale_perm(base,"PERMZ",list(kz))

    cs=row["cell_size_ft"]; dz=SPE5_DZ*row["thickness_mult"]
    base=re.sub(r"(DX\s*\n(?:--[^\n]*\n)?)\s*147\*[0-9.]+\s*/", rf"\g<1>\t147*{cs:.6g} /", base, count=1)
    base=re.sub(r"(DY\s*\n(?:--[^\n]*\n)?)\s*147\*[0-9.]+\s*/", rf"\g<1>\t147*{cs:.6g} /", base, count=1)
    base=re.sub(r"(DZ\s*\n(?:--[^\n]*\n)?)\s*49\*[0-9.]+ 49\*[0-9.]+ 49\*[0-9.]+\s*/",
                rf"\g<1>\t49*{dz[0]:.6g} 49*{dz[1]:.6g} 49*{dz[2]:.6g} /", base, count=1)

    def _scale_pvto(mo):
        out=[]
        for ln in mo.group(2).split("\n"):
            core=ln.split("--")[0]; tail="/" if "/" in core else ""
            nums=core.replace("/"," ").split()
            if len(nums) in (3,4):
                nums[-1]=f"{float(nums[-1])*row['oil_visc_mult']:.5g}"
                ln="\t".join(nums)+(" /" if tail else "")
            out.append(ln)
        return mo.group(1)+"\n".join(out)+mo.group(3)
    base=re.sub(r"(PVTO\s*\n)(.*?)(\n/\s*\n)", _scale_pvto, base, count=1, flags=re.S)

    full=row["mmp_full_psi"]; lo=max(200.0, full-MISC_WIDTH_PSI)
    base=re.sub(r"PMISC\s*\n.*?/", f"PMISC\n{max(100.0, lo-500):.1f} 0.0\n{lo:.1f} 0.0\n{full:.1f} 1.0\n9000 1.0 /",
                base, count=1, flags=re.S)
    base=re.sub(r"(TLMIXPAR\s*\n)\s*[0-9.]+\s*/", rf"\g<1> {row['tl_omega']:.4g} /", base, count=1)

    base=re.sub(r"(PORO\s*\n(?:--[^\n]*\n)?)\s*147\*[0-9.]+\s*/",
                rf"\g<1>\t147*{row['porosity_frac']:.4g} /", base, count=1)

    base=re.sub(r"(TOPS\s*\n(?:--[^\n]*\n)?)\s*49\*[0-9.]+\s*/",
                rf"\g<1>\t49*{row['depth_ft']:.6g} /", base, count=1)

    # Move datum and contacts with the sampled reservoir depth.
    top   = row["depth_ft"]
    bottom= top + float(SPE5_DZ.sum())*row["thickness_mult"]   # sampled layer thicknesses
    goc   = top - 25.0                     # gas-oil contact above the reservoir
    woc   = bottom + 50.0                  # water-oil contact below the reservoir
    datum = top                            # datum at reservoir top
    def fix_equil(mobj):
        nums=mobj.group(1).split()
        nums[0]=f"{datum:.6g}"             # datum depth
        nums[1]=f"{row['init_pressure_psi']:.6g}"  # datum pressure
        nums[2]=f"{woc:.6g}"               # water-oil contact depth
        nums[4]=f"{goc:.6g}"               # gas-oil contact depth
        return "\n\t"+" ".join(nums)+" /"
    base=re.sub(r"\n\t([0-9.]+ [0-9.]+ [0-9.]+ 0 [0-9.]+ 0 1 0 0)\s*/",
                lambda mm: fix_equil(mm), base, count=1)

    base=base_with_field_summary(base)
    open(os.path.join(d,"SPE5.BASE"),"w").write(base)

    # Build the injection schedule.
    hc=int(row["wag_halfcycle_mo"])
    one_year=tstep_line(12)
    hc_steps=tstep_line(hc)
    prim=f"TSTEP\n{one_year}\n{one_year}\n/"     # 2 years primary
    g=row["gas_rate_mscf"]; w=row["wat_rate_stb"]; orat=row["prod_orat_stb"]

    # The waterflood period is shared by both comparison runs.
    wf_years=int(row["waterflood_years"])
    wf_block=("WELOPEN\n 'INJW' 'OPEN' /\n 'INJG' 'SHUT' /\n/\nTSTEP\n"
              + "\n".join(one_year for _ in range(wf_years)) + "\n/")
    if mode=="wag":
        # Alternate gas and water after the waterflood, starting with gas.
        n_half = int(round(WAG_YEARS*12 / hc))
        blocks=[]
        for k in range(n_half):
            if k%2==0:
                blocks.append(f"WELOPEN\n 'INJW' 'SHUT' /\n 'INJG' 'OPEN' /\n/\nTSTEP\n{hc_steps}\n/")
            else:
                blocks.append(f"WELOPEN\n 'INJW' 'OPEN' /\n 'INJG' 'SHUT' /\n/\nTSTEP\n{hc_steps}\n/")
        # Adjust the final half-cycle to match the baseline duration.
        per=sum(MONTHS[j%12] for j in range(hc)); rest=WAG_YEARS*365-n_half*per
        if rest>0:
            blocks.append(f"TSTEP\n{rest}\n/")
        elif rest<0:
            last=blocks.pop(); head=last[:last.index("TSTEP")]
            days=[MONTHS[j%12] for j in range(hc)]
            while rest<0 and days:
                cut=min(-rest, days[-1]); days[-1]-=cut; rest+=cut
                if days[-1]==0: days.pop()
            blocks.append(head+"TSTEP\n"+" ".join(str(x) for x in days)+"\n/")
        schedule=wf_block+"\n"+"\n".join(blocks)
        title=f"SPE5 tertiary WAG LHS case {i}"
        inje=(f"WCONINJE\n 'INJW' 'WATER' 'OPEN' 'RATE' {w:.1f} 1* 10000 /\n"
              f" 'INJG' 'GAS'   'OPEN' 'RATE' {g:.1f} 1* 10000 /\n/\n"
              f"WSOLVENT\n 'INJG' 1.0 /\n/\n"
              f"WELOPEN\n 'INJW' 'SHUT' /\n 'INJG' 'SHUT' /\n/")
        sched_comment=(f"-- {wf_years} years waterflood, then {WAG_YEARS} years of WAG half-cycles "
                       f"({hc} months each, gas first)")
    else:  # Continue the waterflood without gas.
        blocks=[f"WELOPEN\n 'INJW' 'OPEN' /\n 'INJG' 'SHUT' /\n/\nTSTEP\n{one_year}\n/"
                for _ in range(WAG_YEARS)]
        schedule=wf_block+"\n"+"\n".join(blocks)
        title=f"SPE5 continued-waterflood baseline case {i}"
        inje=(f"WCONINJE\n 'INJW' 'WATER' 'OPEN' 'RATE' {w:.1f} 1* 10000 /\n/\n"
              f"WELOPEN\n 'INJW' 'SHUT' /\n 'INJG' 'SHUT' /\n/")
        sched_comment=f"-- {wf_years} years waterflood, then {WAG_YEARS} more years of waterflood (baseline)"

    case=f"""RUNSPEC
UNIFIN
UNIFOUT
TITLE
 {title}
INCLUDE
 'SPE5.BASE' /

WCONPROD
 'PROD' 'OPEN' 'ORAT' {orat:.1f} 4* {min(1000.0, 0.5*row['init_pressure_psi']):.1f} /
/
{inje}
-- 2 years primary production
{prim}
{sched_comment}
{schedule}
END
"""
    fname = "CASE.DATA" if mode=="wag" else "BASE.DATA"
    open(os.path.join(d,fname),"w").write(case)
    return os.path.join(d,fname)

def run_flow(deckfile, d):
    cmd=[FLOW_BIN, deckfile, f"--output-dir={d}"]
    env=dict(os.environ)
    if FLOW_THREADS:
        cmd.append(f"--threads-per-process={FLOW_THREADS}")
        env["OMP_NUM_THREADS"]=str(FLOW_THREADS)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=FLOW_TIMEOUT, env=env)

def read_total_rf(d, prefer=("CASE","BASE")):
    """Read one run's summary and return TOTAL recovery factor + cum oil + OIP.
    'prefer' lets us pick which base filename's summary to read (CASE vs BASE)."""
    try:
        from opm.io.ecl import ESmry
        cand=[]
        for stem in prefer:
            cand += glob.glob(os.path.join(d,f"{stem}.ESMRY"))+glob.glob(os.path.join(d,f"{stem}.SMSPEC"))
        if not cand:
            cand=glob.glob(os.path.join(d,"*.ESMRY"))+glob.glob(os.path.join(d,"*.SMSPEC"))
        if not cand: return None
        s=ESmry(cand[0]); keys=set(s.keys())
        if "FOPT" not in keys: return None
        fopt=np.array(s["FOPT"]); foip=None
        for k in ("FOIP","FOIPT"):
            if k in keys: foip=np.array(s[k]); break
        if len(fopt)==0 or foip is None or len(foip)==0 or foip[0]<=0: return None
        out={"rf_total_pct": float(100.0*fopt[-1]/foip[0]),
             "cum_oil_stb": float(fopt[-1]),
             "oip_initial_stb": float(foip[0])}
        if "FGIT" in keys:
            g=np.array(s["FGIT"]); out["cum_gas_injected"]=float(g[-1]) if len(g) else np.nan
        return out
    except Exception:
        return None


# Convert injected volumes to field-comparable quantities.
# SPE5 grid: 7 x 7 x 3 cells, with connate water saturation of 0.2.
def bulk_ft3(row):
    """Bulk volume of the 7 x 7 x 3 grid for the sampled cell size and thicknesses."""
    return 49*row["cell_size_ft"]**2*float(SPE5_DZ.sum())*row["thickness_mult"]
SPE5_SWC = 0.2
def _pvds_table(base_txt):
    """Parse the PVDS (solvent) table: pressure [psia] -> Bs [rb/Mscf]."""
    m=re.search(r"^PVDS\s*\n(.*?)/", base_txt, re.S|re.M)
    P,B=[],[]
    for line in m.group(1).splitlines():
        line=line.split("--")[0].split()
        if len(line)>=2:
            try: P.append(float(line[0])); B.append(float(line[1]))
            except ValueError: pass
    return np.array(P), np.array(B)
PVDS_P, PVDS_B = _pvds_table(BASE_TXT)
def bs_at(p): return float(np.interp(p, PVDS_P, PVDS_B))

def _inj_days(hc):
    """Gas and water injection days of the simulated WAG schedule (identical to make_case)."""
    hc=int(hc); n=int(round(WAG_YEARS*12/hc)); per=sum(MONTHS[j%12] for j in range(hc))
    gas=((n+1)//2)*per              # WAG starts with gas
    wat=(n//2)*per
    rest=WAG_YEARS*365-n*per
    if (n-1)%2==0: gas+=rest
    else: wat+=rest
    return gas, wat

def derive_rates(design):
    """Target well rates that deliver the sampled HCPVI and reservoir-volume WAG ratio in the
    SPE5 model of the sampled porosity. The solvent volume factor is taken at the initial
    pressure; the volumes actually injected are read back from the summary (hcpvi_gas_frac,
    wag_ratio_resvol) because pressure changes and injector limits alter them."""
    out=design.copy()
    g=[];w=[]
    for _,r in out.iterrows():
        hcpv_rb=bulk_ft3(r)*r["porosity_frac"]*(1-SPE5_SWC)/5.615
        gas_rb=r["hcpvi_target"]*hcpv_rb
        gdays,wdays=_inj_days(r["wag_halfcycle_mo"])
        g.append(gas_rb/(bs_at(r["init_pressure_psi"])*gdays))       # Mscf/d
        w.append(r["wag_ratio_target"]*gas_rb/(1.0*wdays))           # STB/d, Bw ~ 1.0
    out["gas_rate_mscf"]=g; out["wat_rate_stb"]=w
    return out

def read_injection(d, row):
    """From the WAG run summary: cumulative solvent (gas) and water injected, converted to
    reservoir volume and divided by the hydrocarbon pore volume (HCPVI), plus the reservoir-volume
    WAG ratio. These are the quantities reported for real field projects."""
    out={}
    hcpv_rb = bulk_ft3(row)*row["porosity_frac"]*(1-SPE5_SWC)/5.615
    out["hcpv_rb"]=hcpv_rb
    try:
        from opm.io.ecl import ESmry
        cand=glob.glob(os.path.join(d,"CASE.ESMRY"))+glob.glob(os.path.join(d,"CASE.SMSPEC"))
        s=ESmry(cand[0]); keys=set(s.keys())
        gas=None; gas_keys=[k for k in ("FNIT","FGIT") if k in keys]
        for k in gas_keys:                          # prefer solvent total, then gas total
            a=np.array(s[k])
            if len(a) and a[-1]>0: gas=a; break
        if gas is None and gas_keys:                # no gas injected is a valid simulation result
            gas=np.zeros(len(s[gas_keys[0]]))
        wat=np.array(s["FWIT"]) if "FWIT" in keys else None
        prs=np.array(s["FPR"]) if "FPR" in keys else None
        if gas is not None and prs is not None:
            dg=np.diff(np.concatenate([[0.0],gas]))         # Mscf injected per report step
            gas_rb=float(np.sum(dg*np.array([bs_at(p) for p in prs])))
            out["cum_solvent_injected_mscf"]=float(gas[-1])
            out["hcpvi_source"]="summary" if gas[-1]>0 else "summary_no_gas_injected"
        else:   # fallback estimate from the target rate and injection days
            gas_days,_=_inj_days(row["wag_halfcycle_mo"])
            gas_rb=row["gas_rate_mscf"]*gas_days*bs_at(row["init_pressure_psi"])
            out["cum_solvent_injected_mscf"]=row["gas_rate_mscf"]*gas_days; out["hcpvi_source"]="schedule_estimate"
        # Only count water injected during the WAG branch.
        t_branch=365.0*(PRIMARY_YEARS+int(row["waterflood_years"]))
        wat_wag=wat_pre=np.nan
        if wat is not None and len(wat):
            if "TIME" in keys:
                tt=np.array(s["TIME"]); wat_pre=float(np.interp(t_branch, tt, wat))
            else:   # fallback estimate for the pre-WAG water volume
                wat_pre=min(float(wat[-1]), row["wat_rate_stb"]*365.0*int(row["waterflood_years"]))
            wat_wag=float(wat[-1])-wat_pre
        wat_rb=wat_wag*1.0                                               # Bw ~ 1.0 (PVTW)
        out["cum_water_injected_stb"]=wat_wag
        out["pvi_water_waterflood_frac"]=wat_pre/hcpv_rb if wat_pre==wat_pre else np.nan
        out["hcpvi_gas_frac"]=gas_rb/hcpv_rb
        out["pvi_water_frac"]=wat_rb/hcpv_rb if wat_rb==wat_rb else np.nan
        out["wag_ratio_resvol"]=wat_rb/gas_rb if gas_rb>0 and wat_rb==wat_rb else np.nan
    except Exception:
        out["hcpvi_source"]="failed"
    return out

def run_scenario(i, row, d):
    """Run BOTH the WAG case and the waterflood baseline for one scenario, and return
    a dict with the INCREMENTAL recovery = RF(WAG) - RF(waterflood), plus both totals."""
    # Run the WAG case.
    wag_deck = make_case(i, row, d, mode="wag")
    run_flow(wag_deck, d)
    wag = read_total_rf(d, prefer=("CASE",))
    inj = read_injection(d, row)
    # Run the continued-waterflood baseline.
    base_deck = make_case(i, row, d, mode="waterflood")
    run_flow(base_deck, d)
    base = read_total_rf(d, prefer=("BASE",))
    if not wag or not base:
        return None
    rf_wag  = wag["rf_total_pct"]
    rf_base = base["rf_total_pct"]
    return {
        "rf_wag_total_pct":        rf_wag,
        "rf_waterflood_total_pct": rf_base,
        "rf_incremental_pct":      rf_wag - rf_base,
        "cum_oil_stb":             wag["cum_oil_stb"],
        "oip_initial_stb":         wag["oip_initial_stb"],
        "cum_gas_injected":        wag.get("cum_gas_injected", np.nan),
        **inj,
    }

def _worker(args):
    """Run one scenario in its own folder and delete the folder afterwards."""
    i,row=args
    d=os.path.join(WORK,f"case_{i:04d}")
    try:
        o=run_scenario(i,row,d)
    except Exception:
        o=None
    shutil.rmtree(d, ignore_errors=True)
    return i,o

def _jsonable(o):
    if o is None: return None
    return {k:(float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v) for k,v in o.items()}

def _save_checkpoint(fh, i, row, o):
    fh.write(json.dumps({"i":int(i),"perm_mult":float(row["perm_mult"]),
                         "hcpvi_target":float(row["hcpvi_target"]),"out":_jsonable(o)})+"\n")
    fh.flush()

def main():
    if not os.path.exists(BASE_SRC):
        raise SystemExit("SPE5.BASE not found next to this script.")
    if not shutil.which("flow"):
        raise SystemExit("OPM Flow ('flow') not on PATH.")
    os.makedirs(WORK, exist_ok=True)
    design=derive_rates(lhs(N_SCENARIOS, RANGES))
    print(f"[plan] family {VARIANT} | {DATASET_ROLE} | {N_SCENARIOS} sampled scenarios | output: {OUT_CSV}")

    print("[debug] first case check (runs BOTH WAG and waterflood baseline)...")
    d0=os.path.join(WORK,"case_0000")
    o0=run_scenario(0, design.iloc[0], d0)
    if not o0:
        print("[debug] first case failed to produce valid output. Check the deck in", d0)
        # Print the simulator tail for the failed run.
        for stem in ("CASE","BASE"):
            r=run_flow(os.path.join(d0,f"{stem}.DATA"), d0)
            print(f"--- {stem} stdout tail ---")
            print("\n".join((r.stdout or '').splitlines()[-20:]))
        return
    print(f"[debug] OK  RF_WAG={o0['rf_wag_total_pct']:.2f}%  "
          f"RF_waterflood={o0['rf_waterflood_total_pct']:.2f}%  "
          f"=> INCREMENTAL={o0['rf_incremental_pct']:.2f}%  HCPVI_gas={o0.get('hcpvi_gas_frac',float('nan')):.3f} "
          f"({o0.get('hcpvi_source','?')})  WAG_resvol={o0.get('wag_ratio_resvol',float('nan')):.2f}. Running {N_SCENARIOS}...\n")

    # Resume only when the checkpoint matches the current design.
    done={}
    if os.path.exists(CHECKPOINT):
        for line in open(CHECKPOINT):
            if not line.strip(): continue
            j=json.loads(line); i=j["i"]
            if i>=len(design) or abs(design.loc[i,"perm_mult"]-j["perm_mult"])>1e-9 \
               or abs(design.loc[i,"hcpvi_target"]-j["hcpvi_target"])>1e-9:
                raise SystemExit(f"{CHECKPOINT} belongs to a different design (N_SCENARIOS, RANGES or SEED "
                                 f"changed). Delete or rename it to start a new run.")
            done[i]=j["out"]
        print(f"[resume] {len(done)} scenarios already finished, read from {CHECKPOINT}")
    ck=open(CHECKPOINT,"a")
    if 0 not in done:
        _save_checkpoint(ck, 0, design.loc[0], o0); done[0]=_jsonable(o0)

    todo=[(i,row) for i,row in design.iterrows() if i not in done]
    print(f"[parallel] {len(todo)} scenarios to run with {N_WORKERS} workers x {FLOW_THREADS or 'default'} threads")
    t0=time.time(); n_done=0; ok=sum(1 for v in done.values() if v)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures=[ex.submit(_worker,(i,row)) for i,row in todo]
        for fut in as_completed(futures):
            i,o=fut.result()
            _save_checkpoint(ck, i, design.loc[i], o); done[i]=_jsonable(o)
            n_done+=1; ok+=bool(o)
            if n_done%10==0 or n_done==len(todo):
                el=time.time()-t0; eta=el/n_done*(len(todo)-n_done)
                print(f"  {len(done)}/{N_SCENARIOS} finished, {ok} valid | {el/60:.1f} min elapsed, "
                      f"ETA {eta/60:.1f} min", flush=True)
    ck.close()

    rows=[]
    for i,row in design.iterrows():
        rec=row.to_dict(); rec.update(FIXED)
        o=done.get(i)
        if o: rec.update(o)
        else: rec["rf_incremental_pct"]=np.nan
        rec["source"]="OPM_SPE5_simulation"
        rows.append(rec)

    df=pd.DataFrame(rows)
    # Keep all valid outcomes, including non-positive incremental recovery values.
    n_total=len(df)
    n_failed=int(df["rf_incremental_pct"].isna().sum())
    n_nonpos=int((df["rf_incremental_pct"]<=0).sum())
    # Do not filter on the outcome; that would bias the model toward positive WAG effects.
    df=df[df["rf_incremental_pct"].notna()].reset_index(drop=True)
    print(f"\n[summary] scenarios run: {n_total} | simulation failed: {n_failed} | "
          f"incremental <= 0: {n_nonpos} | kept: {len(df)}")
    df["effective_perm_md"]=df["perm_mult"]*(500+50+200)/3.0
    df["porosity_pct"]=df["porosity_frac"]*100.0
    df["wag_ratio_approx"]=(df["wat_rate_stb"]/(df["gas_rate_mscf"])).round(3)

    varied=["perm_mult","effective_perm_md","porosity_frac","porosity_pct",
            "init_pressure_psi","depth_ft","gas_rate_mscf","wat_rate_stb",
            "prod_orat_stb","wag_halfcycle_mo","wag_ratio_approx",
            "hcpvi_target","wag_ratio_target","waterflood_years"]+list(FAMILY.keys())
    fixedcols=list(FIXED.keys())
    # Export both total recoveries and the incremental target.
    outs=["rf_wag_total_pct","rf_waterflood_total_pct","cum_oil_stb","oip_initial_stb",
          "cum_gas_injected","cum_solvent_injected_mscf","cum_water_injected_stb","hcpv_rb",
          "hcpvi_gas_frac","pvi_water_frac","wag_ratio_resvol","pvi_water_waterflood_frac","hcpvi_source","rf_incremental_pct"]
    cols=["source"]+varied+fixedcols+[c for c in outs if c in df.columns]
    df=df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT_CSV,index=False)
    if os.path.exists(CHECKPOINT):
        os.replace(CHECKPOINT, CHECKPOINT.replace(".jsonl","_completed.jsonl"))
    print(f"\nSaved {len(df)} scenarios with TRUE incremental target -> {OUT_CSV}")
    print("Columns:", list(df.columns))
    print(df[["rf_wag_total_pct","rf_waterflood_total_pct","rf_incremental_pct"]].describe().round(2).to_string())

if __name__=="__main__":
    main()
