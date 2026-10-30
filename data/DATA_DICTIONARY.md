# Data dictionary

| File | Family | Rows | Role | SHA-256 |
|---|---|---|---|---|
| `WAG_simulation_dataset_B_train.csv` | B | 4,967 | development (training/validation) | `56f11f281d7133dc0d1ef9e92578c901d7740375e68c8bb4be2670e8f2cbbd3f` |
| `WAG_simulation_dataset_A_test.csv` | A | 995 | final test | `45ced11bd575e47c541996a95bef6b9c66ea818c17a19935a890d25401c29063` |

Each row is one simulated scenario produced by `simulation/sim_v3.py` (OPM Flow, SPE5
benchmark). Both files have the same 48 columns. The two families differ only in the
sampled ranges of `perm_contrast_exp` and `oil_visc_mult`.

**Target:** `rf_incremental_pct` (percentage points of initial oil in place).

**Model predictors (default `volumes` mode):** `effective_perm_md`, `porosity_frac`,
`init_pressure_psi`, `depth_ft`, `prod_orat_stb`, `wag_halfcycle_mo`, `waterflood_years`,
`perm_contrast_exp`, `oil_visc_mult`, `hcpvi_gas_frac`, `wag_ratio_resvol`.
The `design` mode replaces the last two with `hcpvi_target` and `wag_ratio_target`.

## Sampled design inputs

| Column | Unit | Description |
|---|---|---|
| `source` | – | Always `OPM_SPE5_simulation` |
| `perm_mult` | – | Permeability multiplier applied to the SPE5 layer permeabilities (0.007–5) |
| `effective_perm_md` | mD | Arithmetic mean of the three layer permeabilities = 250 × `perm_mult` |
| `porosity_frac` | fraction | Porosity of all cells |
| `porosity_pct` | % | `porosity_frac` × 100 |
| `init_pressure_psi` | psia | Initial pressure at the reservoir top (gradient × depth, limited to 700–5,500) |
| `depth_ft` | ft | Depth of the reservoir top |
| `gas_rate_mscf` | Mscf/d | Target gas injection rate during gas half-cycles |
| `wat_rate_stb` | STB/d | Target water injection rate (waterflood, WAG water half-cycles and baseline) |
| `prod_orat_stb` | STB/d | Producer oil-rate target |
| `wag_halfcycle_mo` | months | Length of one WAG half-cycle |
| `wag_ratio_approx` | STB/Mscf | `wat_rate_stb` / `gas_rate_mscf` (surface-rate ratio, descriptive only) |
| `hcpvi_target` | HCPV | Target gas volume injected over the WAG period |
| `wag_ratio_target` | – | Target reservoir-volume water-to-gas ratio |
| `waterflood_years` | years | Waterflood duration before WAG starts |
| `perm_contrast_exp` | – | Exponent applied to the SPE5 layer-permeability profile (B: 1.0–1.4; A: 0.8–1.2) |
| `oil_visc_mult` | – | Multiplier on PVTO oil viscosities (B: 1.2–2.0; A: 0.8–1.5) |

## Constant columns (fixed SPE5 settings, not model inputs)

`kv_kh_mult` (1.0), `thickness_mult` (1.0), `cell_size_ft` (500), `mmp_full_psi` (3,200;
pressure for full miscibility in the PMISC table), `tl_omega` (0.6; Todd–Longstaff
mixing parameter), `process_type` (`Miscible`), `injectant` (`SPE5 solvent`),
`temperature_F` (160), `grid_nx`/`grid_ny`/`grid_nz` (7/7/3), `reservoir_family`
(`A` or `B`), `producer_cell` (`7,7,3`), `injector_cell` (`1,1,1`), `solvent_fraction`
(1.0), `primary_years` (2), `wag_years` (10), `hcpvi_source` (`summary`).

## Simulation outputs

| Column | Unit | Description |
|---|---|---|
| `rf_wag_total_pct` | % OOIP | Total recovery factor of the WAG branch at the end of the schedule |
| `rf_waterflood_total_pct` | % OOIP | Total recovery factor of the continued-waterflood baseline at the same time |
| `cum_oil_stb` | STB | Cumulative oil of the WAG branch |
| `oip_initial_stb` | STB | Initial oil in place (from the WAG-run summary) |
| `cum_gas_injected` | Mscf | Cumulative gas injected in the WAG run (FGIT) |
| `cum_solvent_injected_mscf` | Mscf | Cumulative solvent injected in the WAG run |
| `cum_water_injected_stb` | STB | Water injected during the WAG period only |
| `hcpv_rb` | rb | Hydrocarbon pore volume used for normalization (pore volume × (1 − 0.2)) |
| `hcpvi_gas_frac` | HCPV | Realized injected gas volume at reservoir conditions / `hcpv_rb` |
| `pvi_water_frac` | HCPV | Realized water injected during WAG / `hcpv_rb` |
| `wag_ratio_resvol` | – | Realized reservoir-volume water-to-gas ratio during WAG |
| `pvi_water_waterflood_frac` | HCPV | Water injected before WAG / `hcpv_rb` |
| `rf_incremental_pct` | % OOIP points | **Target**: `rf_wag_total_pct − rf_waterflood_total_pct` |

Realized gas volumes are converted to reservoir conditions with the solvent formation
volume factor at the field-average pressure of each report step.

Licence: CC BY 4.0.
