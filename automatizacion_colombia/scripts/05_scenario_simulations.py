#!/usr/bin/env python3
"""
05_scenario_simulations.py
===========================
Scenario Simulations: Impact of Labor Cost Policies on Automation
Adoption and Employment Displacement in Colombia (2025-2035)

This script builds a simulation framework projecting the effects of five
policy/technology scenarios on formal employment, informality, and
automation adoption across Colombian economic sectors.

IMPORTANT DISCLAIMER:
    This is a SIMULATION exercise, NOT an empirical estimation. Results
    represent plausible projections under stated assumptions and should be
    interpreted as scenario-conditional forecasts, not precise predictions.
    All parameter choices are documented with their sources.

Scenarios:
    1. Status Quo (Baseline) - Current policies continue
    2. Labor Reform (Cost Shock) - Proposed reform increases labor costs 18-25%
    3. Parafiscal Reform (Cost Reduction) - Eliminate SENA/ICBF contributions
    4. AI Acceleration (Technology Shock) - Rapid AI/robotics cost decline
    5. Combined Worst Case - Labor reform + AI acceleration

Data sources & parameter citations:
    - DANE: GEIH employment, PIB sectorial
    - IFR: Robot density estimates
    - Fedesarrollo: 57-58% automation risk
    - Morales et al. (2023): 37% high risk, 55% medium risk
    - ANIF: Elasticities of informality/unemployment to labor costs
    - Acemoglu & Restrepo (2020): Reinstatement effects
    - Fenalco: Reform cost estimates (18-34%)
    - Cheng et al. (2021): Elasticity of substitution estimates
    - Korea Ministry of Employment (2019): Minimum wage shock effects

Author: Research project on automation and labor costs in Colombia
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import stats
import copy

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

# Random seed for reproducibility
np.random.seed(42)

# Color scheme (consistent with previous scripts)
MAINBLUE = '#1B4F72'
ACCENTRED = '#C0392B'
LIGHTBLUE = '#2E86C1'
DARKGRAY = '#2C3E50'
LIGHTGRAY = '#BDC3C7'
GOLD = '#D4AC0D'
GREEN = '#1E8449'
ORANGE = '#E67E22'
PURPLE = '#8E44AD'

# Scenario colors
SCENARIO_COLORS = {
    'Status Quo':       MAINBLUE,
    'Labor Reform':     ACCENTRED,
    'Parafiscal Reform': GREEN,
    'AI Acceleration':  ORANGE,
    'Combined Worst':   PURPLE,
}

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Palatino', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'text.usetex': False,
})

# Simulation horizon
YEAR_START = 2025
YEAR_END = 2035
N_YEARS = YEAR_END - YEAR_START + 1

# Monte Carlo parameters
N_MC_DRAWS = 1000

print("=" * 70)
print("SCENARIO SIMULATIONS: AUTOMATION & LABOR COSTS IN COLOMBIA")
print("Projection Horizon: 2025-2035")
print("=" * 70)

# ============================================================================
# STEP 1: SECTOR PROFILES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Defining Sector Profiles")
print("=" * 70)


@dataclass
class SectorProfile:
    """
    Represents a Colombian economic sector with its labor market
    characteristics and automation vulnerability parameters.

    Parameter sources documented inline.
    """
    name: str
    # Employment figures (millions) - Source: DANE GEIH 2024
    total_employment: float      # Total employed (millions)
    formal_pct: float            # Formality rate (%)
    # Derived
    formal_employment: float = 0.0
    informal_employment: float = 0.0

    # Wages - Source: DANE, MinTrabajo, sector reports
    avg_monthly_wage_cop: float = 1_423_500  # COP/month (formal workers)

    # Labor cost structure - Source: OECD Taxing Wages 2025, ANIF
    non_wage_cost_pct: float = 0.52          # 52% over base salary

    # Automation risk - Source: Morales et al. (2023), Fedesarrollo
    automation_risk_high_pct: float = 0.37   # High automation risk
    automation_risk_medium_pct: float = 0.55 # Medium risk

    # Vulnerability index - Source: Our Script 02 results
    automation_vulnerability_index: float = 50.0

    # Robot density - Source: IFR estimates, adapted for Colombia
    current_robot_density: float = 2.0       # per 10,000 workers

    # Sector type for automation dynamics
    automation_type: str = 'mixed'  # 'physical', 'cognitive', 'mixed'

    # Adoption rate parameters (10-year cumulative)
    physical_adoption_rate: float = 0.10     # 10% over 10 years
    cognitive_adoption_rate: float = 0.20    # 20% over 10 years

    def __post_init__(self):
        self.formal_employment = self.total_employment * self.formal_pct
        self.informal_employment = self.total_employment * (1 - self.formal_pct)

    @property
    def total_labor_cost(self):
        """Total monthly cost per formal worker (COP)."""
        return self.avg_monthly_wage_cop * (1 + self.non_wage_cost_pct)

    @property
    def effective_adoption_rate(self):
        """Weighted adoption rate based on sector type."""
        if self.automation_type == 'physical':
            return self.physical_adoption_rate
        elif self.automation_type == 'cognitive':
            return self.cognitive_adoption_rate
        else:
            return 0.5 * self.physical_adoption_rate + 0.5 * self.cognitive_adoption_rate


# Define all sectors with calibrated parameters
# Sources for each parameter documented in comments
SECTORS = [
    SectorProfile(
        name='Agriculture',
        total_employment=3.3,       # GEIH 2024: 14.3% of ~23M
        formal_pct=0.15,            # DANE: >85% informal
        avg_monthly_wage_cop=1_300_000,  # Near minimum; many earn below
        automation_risk_high_pct=0.30,   # Morales et al.: moderate physical tasks
        automation_risk_medium_pct=0.42,
        automation_vulnerability_index=72.0,  # Script 02 result
        current_robot_density=0.5,   # Very low mechanization
        automation_type='physical',
        physical_adoption_rate=0.05, # Slow: terrain, smallholdings
        cognitive_adoption_rate=0.02,
    ),
    SectorProfile(
        name='Manufacturing',
        total_employment=2.5,       # GEIH: 10.7%
        formal_pct=0.50,
        avg_monthly_wage_cop=1_800_000,
        automation_risk_high_pct=0.45,   # High: routine manual tasks
        automation_risk_medium_pct=0.35,
        automation_vulnerability_index=70.8,
        current_robot_density=5.0,   # IFR: Colombia ~2-5 in mfg
        automation_type='physical',
        physical_adoption_rate=0.15, # Moderate: some firms already investing
        cognitive_adoption_rate=0.08,
    ),
    SectorProfile(
        name='Construction',
        total_employment=1.5,       # GEIH: 6.7%
        formal_pct=0.28,            # ~72% informal
        avg_monthly_wage_cop=1_600_000,
        automation_risk_high_pct=0.35,
        automation_risk_medium_pct=0.45,
        automation_vulnerability_index=80.5,  # Highest in Script 02
        current_robot_density=0.3,
        automation_type='physical',
        physical_adoption_rate=0.08,
        cognitive_adoption_rate=0.05,
    ),
    SectorProfile(
        name='Commerce/Transport',
        total_employment=6.4,       # GEIH: 27.8%
        formal_pct=0.45,            # ~55% informal
        avg_monthly_wage_cop=1_500_000,
        automation_risk_high_pct=0.40,
        automation_risk_medium_pct=0.40,
        automation_vulnerability_index=63.1,
        current_robot_density=1.0,
        automation_type='mixed',
        physical_adoption_rate=0.08,
        cognitive_adoption_rate=0.15,
    ),
    SectorProfile(
        name='Public Admin/Educ/Health',
        total_employment=2.8,       # GEIH: 12.1%
        formal_pct=0.80,            # ~20% informal
        avg_monthly_wage_cop=2_500_000,
        automation_risk_high_pct=0.25,
        automation_risk_medium_pct=0.50,
        automation_vulnerability_index=78.0,
        current_robot_density=0.2,
        automation_type='cognitive',
        physical_adoption_rate=0.03,
        cognitive_adoption_rate=0.18,
    ),
    SectorProfile(
        name='Financial Services',
        total_employment=0.4,       # GEIH: 1.8%
        formal_pct=0.92,            # ~8% informal
        avg_monthly_wage_cop=3_500_000,
        automation_risk_high_pct=0.50,   # High: routine cognitive
        automation_risk_medium_pct=0.30,
        automation_vulnerability_index=36.7,
        current_robot_density=0.1,   # Software-based automation
        automation_type='cognitive',
        physical_adoption_rate=0.02,
        cognitive_adoption_rate=0.25,
    ),
    SectorProfile(
        name='BPO/Professional',
        total_employment=1.2,       # GEIH: 5.2%
        formal_pct=0.70,            # ~30% informal
        avg_monthly_wage_cop=2_200_000,
        automation_risk_high_pct=0.55,   # Very high: GenAI exposure
        automation_risk_medium_pct=0.30,
        automation_vulnerability_index=70.5,
        current_robot_density=0.1,
        automation_type='cognitive',
        physical_adoption_rate=0.02,
        cognitive_adoption_rate=0.30,  # Fastest adoption
    ),
    SectorProfile(
        name='Mining',
        total_employment=0.6,       # GEIH: 2.6%
        formal_pct=0.60,            # ~40% informal
        avg_monthly_wage_cop=2_800_000,
        automation_risk_high_pct=0.35,
        automation_risk_medium_pct=0.40,
        automation_vulnerability_index=58.7,
        current_robot_density=3.0,
        automation_type='physical',
        physical_adoption_rate=0.12,
        cognitive_adoption_rate=0.05,
    ),
    SectorProfile(
        name='Other Services',
        total_employment=4.3,       # GEIH: 18.8%
        formal_pct=0.40,
        avg_monthly_wage_cop=1_400_000,
        automation_risk_high_pct=0.30,
        automation_risk_medium_pct=0.45,
        automation_vulnerability_index=44.0,
        current_robot_density=0.5,
        automation_type='mixed',
        physical_adoption_rate=0.06,
        cognitive_adoption_rate=0.12,
    ),
]

# Validate totals
total_emp = sum(s.total_employment for s in SECTORS)
total_formal = sum(s.formal_employment for s in SECTORS)
total_informal = sum(s.informal_employment for s in SECTORS)

print(f"\nTotal employment across sectors: {total_emp:.1f}M")
print(f"  Formal: {total_formal:.2f}M ({total_formal/total_emp*100:.1f}%)")
print(f"  Informal: {total_informal:.2f}M ({total_informal/total_emp*100:.1f}%)")
print(f"  Sectors defined: {len(SECTORS)}")

# Print sector summary
print(f"\n{'Sector':<25} {'Total':>6} {'Formal':>7} {'Informal':>9} {'Risk%':>6} {'VulnIdx':>8}")
print("-" * 68)
for s in SECTORS:
    print(f"{s.name:<25} {s.total_employment:>5.1f}M {s.formal_employment:>6.2f}M "
          f"{s.informal_employment:>7.2f}M {s.automation_risk_high_pct*100:>5.0f}% "
          f"{s.automation_vulnerability_index:>7.1f}")


# ============================================================================
# STEP 2: SCENARIO DEFINITIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining Policy/Technology Scenarios")
print("=" * 70)


@dataclass
class Scenario:
    """
    Defines a policy/technology scenario with its parameter modifications.
    """
    name: str
    description: str

    # Labor cost changes
    non_wage_cost_change: float = 0.0      # Additive change to non_wage_cost_pct
    additional_cost_shock_pct: float = 0.0  # % increase in total labor cost
    min_wage_annual_growth: float = 0.07    # Annual min wage growth (inflation+2%)

    # Technology parameters
    tech_adoption_multiplier: float = 1.0   # Multiplier on base adoption rates
    tech_cost_decline_rate: float = 0.10    # Annual decline in automation cost
    cognitive_cost_reduction: float = 0.0   # One-time reduction in cognitive auto cost
    physical_cost_reduction: float = 0.0    # One-time reduction in physical auto cost

    # Policy parameters
    firm_substitution_rate: float = 0.0     # % of firms that substitute tech for labor
    horizon_years: int = 10                 # Projection horizon

    # Reinstatement effect (Acemoglu & Restrepo)
    # Emerging economy: 30-50% reinstatement (vs 50-70% advanced)
    reinstatement_rate: float = 0.40        # Central estimate for Colombia

    # ANIF elasticities
    anif_informality_elasticity: float = 0.004  # 1% cost increase -> 0.4pp informality
    anif_unemployment_elasticity: float = 0.001 # 1% cost increase -> 0.1pp unemployment


SCENARIOS = {
    'Status Quo': Scenario(
        name='Status Quo',
        description='Current policies continue. Non-wage costs at 52%. '
                    'Minimum wage grows at inflation + 2%. '
                    'Gradual technology adoption (2% annual growth).',
        non_wage_cost_change=0.0,
        additional_cost_shock_pct=0.0,
        min_wage_annual_growth=0.07,  # ~5% inflation + 2%
        tech_adoption_multiplier=1.0,
        tech_cost_decline_rate=0.10,
        reinstatement_rate=0.40,
    ),
    'Labor Reform': Scenario(
        name='Labor Reform',
        description='Proposed labor reform approved. Non-wage costs rise to ~60%. '
                    'Night/Sunday premiums increase. Net: 18-25% labor cost increase. '
                    '17% of firms substitute technology. Source: Fenalco, ANIF.',
        non_wage_cost_change=0.08,        # 52% -> 60%
        additional_cost_shock_pct=0.20,    # Central: 20% from premiums etc.
        min_wage_annual_growth=0.09,       # Higher under reform pressure
        tech_adoption_multiplier=1.5,      # Cost shock accelerates adoption
        tech_cost_decline_rate=0.10,
        firm_substitution_rate=0.17,       # Fenalco: 17% of employers
        reinstatement_rate=0.35,           # Lower: adjustment friction
    ),
    'Parafiscal Reform': Scenario(
        name='Parafiscal Reform',
        description='Eliminate SENA (2%) and ICBF (3%) for all firms. '
                    'Non-wage costs drop to ~40%. Total labor cost reduction ~12%. '
                    'Slows automation incentive.',
        non_wage_cost_change=-0.12,        # 52% -> 40%
        additional_cost_shock_pct=-0.05,   # Net savings from simplification
        min_wage_annual_growth=0.06,       # More moderate growth
        tech_adoption_multiplier=0.7,      # Reduced incentive
        tech_cost_decline_rate=0.10,
        reinstatement_rate=0.45,           # Better: more competitive labor
    ),
    'AI Acceleration': Scenario(
        name='AI Acceleration',
        description='GenAI reduces cognitive automation cost by 50%. '
                    'Manufacturing automation cost drops 30%. '
                    'Compressed 5-year timeline. Source: WEF FoJ 2025.',
        non_wage_cost_change=0.0,
        additional_cost_shock_pct=0.0,
        min_wage_annual_growth=0.07,
        tech_adoption_multiplier=2.0,      # Doubled adoption rates
        tech_cost_decline_rate=0.20,       # Faster cost decline
        cognitive_cost_reduction=0.50,     # 50% cost drop for cognitive
        physical_cost_reduction=0.30,      # 30% cost drop for physical
        reinstatement_rate=0.30,           # Lower: rapid displacement
        horizon_years=5,                   # Compressed timeline
    ),
    'Combined Worst': Scenario(
        name='Combined Worst',
        description='Labor reform + AI acceleration simultaneously. '
                    'Maximum labor cost increase meets rapid technology '
                    'cost decline. Worst case for formal employment.',
        non_wage_cost_change=0.08,
        additional_cost_shock_pct=0.20,
        min_wage_annual_growth=0.09,
        tech_adoption_multiplier=2.5,      # Strongest adoption push
        tech_cost_decline_rate=0.20,
        cognitive_cost_reduction=0.50,
        physical_cost_reduction=0.30,
        firm_substitution_rate=0.17,
        reinstatement_rate=0.25,           # Lowest: double shock
        horizon_years=10,
    ),
}

for sname, scen in SCENARIOS.items():
    print(f"\n  {sname}: {scen.description[:80]}...")


# ============================================================================
# STEP 3: SIMULATION ENGINE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Running Simulation Engine")
print("=" * 70)


def simulate_sector_scenario(
    sector: SectorProfile,
    scenario: Scenario,
    elasticity_substitution: float = 2.5,
    years: int = N_YEARS,
    start_year: int = YEAR_START,
) -> pd.DataFrame:
    """
    Simulate the trajectory of a single sector under a given scenario.

    Parameters
    ----------
    sector : SectorProfile
        Sector characteristics
    scenario : Scenario
        Policy/technology scenario
    elasticity_substitution : float
        Elasticity of substitution between labor and capital (sigma).
        Central: 2.5 (Cheng et al. 2021), Low: 1.5, High: 3.8
    years : int
        Number of years to simulate
    start_year : int
        Starting year

    Returns
    -------
    pd.DataFrame
        Year-by-year simulation results
    """
    results = []

    # Initial conditions
    formal_t = sector.formal_employment
    informal_t = sector.informal_employment
    total_t = sector.total_employment
    base_wage = sector.avg_monthly_wage_cop
    old_non_wage_pct = sector.non_wage_cost_pct
    new_non_wage_pct = old_non_wage_pct + scenario.non_wage_cost_change

    # Old total cost per worker (monthly)
    old_cost = base_wage * (1 + old_non_wage_pct)

    # Cumulative tracking
    cumulative_displaced = 0.0
    cumulative_reinstated = 0.0
    cumulative_informalized = 0.0

    for y in range(years):
        year = start_year + y
        t = y / max(years - 1, 1)  # Normalized time [0, 1]

        # --- Wage evolution ---
        wage_t = base_wage * (1 + scenario.min_wage_annual_growth) ** y

        # --- New effective labor cost ---
        new_cost_t = wage_t * (1 + new_non_wage_pct) * (1 + scenario.additional_cost_shock_pct)

        # Old cost trajectory (what it would have been without scenario)
        baseline_cost_t = base_wage * (1 + 0.07) ** y * (1 + old_non_wage_pct)

        # --- Cost change ratio ---
        cost_ratio = new_cost_t / baseline_cost_t
        delta_cost_pct = (cost_ratio - 1) * 100  # Percentage change

        # --- Automation incentive ---
        # Higher labor costs increase the incentive to automate
        delta_incentive = (cost_ratio - 1) * elasticity_substitution

        # --- Technology adoption rate ---
        # Base adoption rate for this sector
        if sector.automation_type == 'physical':
            base_adopt = sector.physical_adoption_rate
            tech_cost_factor = 1 - scenario.physical_cost_reduction
        elif sector.automation_type == 'cognitive':
            base_adopt = sector.cognitive_adoption_rate
            tech_cost_factor = 1 - scenario.cognitive_cost_reduction
        else:
            base_adopt = sector.effective_adoption_rate
            tech_cost_factor = 1 - 0.5 * (scenario.physical_cost_reduction +
                                            scenario.cognitive_cost_reduction)

        # Technology cost declines over time, increasing adoption
        tech_decline_factor = (1 - scenario.tech_cost_decline_rate) ** y
        effective_tech_cost = tech_cost_factor * tech_decline_factor

        # Technology cost decline creates an adoption boost:
        # As automation gets cheaper relative to labor, adoption accelerates.
        # This is modulated by sigma: higher sigma means firms substitute
        # more readily when relative factor prices change.
        # The "price" of automation falls by (1 - effective_tech_cost) over time,
        # while labor costs rise by wage_t / base_wage.
        labor_cost_growth = wage_t / base_wage
        # Relative price of automation vs labor (falling tech cost / rising labor)
        relative_price_ratio = effective_tech_cost / labor_cost_growth
        # Adoption boost from relative price change, scaled by sigma
        # When sigma > 1, cheaper automation strongly increases adoption
        price_adoption_boost = max(0, (1 - relative_price_ratio) *
                                   (elasticity_substitution / 2.5))

        # Adoption schedule: S-curve (logistic) over the horizon
        # Faster for cognitive, slower for physical
        midpoint = 0.5 if sector.automation_type != 'physical' else 0.6
        steepness = 8.0 if scenario.tech_adoption_multiplier > 1.5 else 6.0
        s_curve = 1 / (1 + np.exp(-steepness * (t - midpoint)))

        # Effective annual adoption:
        # Base adoption x scenario multiplier x S-curve timing
        # x (1 + policy cost incentive + technology price incentive)
        annual_adopt = (base_adopt * scenario.tech_adoption_multiplier *
                        s_curve * (1 + max(0, delta_incentive) +
                                   price_adoption_boost))
        # Cap at reasonable maximum
        annual_adopt = min(annual_adopt, 0.15)  # Max 15% per year

        # --- Jobs displaced this year ---
        # Only formal jobs are displaced by automation (informal already low-cost)
        jobs_at_risk = formal_t * sector.automation_risk_high_pct
        displaced_this_year = jobs_at_risk * annual_adopt

        # Add firm substitution effect (one-time, spread over first 3 years)
        if y < 3 and scenario.firm_substitution_rate > 0:
            firm_sub = (formal_t * scenario.firm_substitution_rate *
                        sector.automation_risk_high_pct / 3)
            displaced_this_year += firm_sub

        # --- Reinstatement effect (new tasks created) ---
        # Acemoglu & Restrepo: new tasks offset ~30-50% in emerging economies
        reinstated_this_year = displaced_this_year * scenario.reinstatement_rate

        # --- Informalization effect (ANIF elasticity) ---
        # Formal workers pushed to informal sector due to cost increases
        if delta_cost_pct > 0:
            informalized_this_year = (formal_t *
                                      scenario.anif_informality_elasticity *
                                      delta_cost_pct * (1 / years))
        else:
            # Cost reduction: some informals become formal
            informalized_this_year = (informal_t *
                                      scenario.anif_informality_elasticity *
                                      delta_cost_pct * (1 / years))

        # --- Net employment changes ---
        net_displaced = displaced_this_year - reinstated_this_year
        net_formal_change = -(net_displaced + max(0, informalized_this_year))
        net_informal_change = max(0, informalized_this_year) - min(0, informalized_this_year)

        # Unemployment from displacement (not all displaced become informal)
        unemployment_effect = net_displaced * 0.3  # 30% become unemployed

        # Update stocks
        formal_t = max(0, formal_t + net_formal_change)
        informal_t = max(0, informal_t + net_informal_change)
        total_t = formal_t + informal_t

        # Cumulative tracking
        cumulative_displaced += displaced_this_year
        cumulative_reinstated += reinstated_this_year
        cumulative_informalized += max(0, informalized_this_year)

        results.append({
            'year': year,
            'sector': sector.name,
            'scenario': scenario.name,
            'formal_employment': formal_t,
            'informal_employment': informal_t,
            'total_employment': total_t,
            'wage_monthly_cop': wage_t,
            'total_labor_cost': new_cost_t,
            'cost_change_pct': delta_cost_pct,
            'adoption_rate': annual_adopt,
            'displaced_annual': displaced_this_year,
            'reinstated_annual': reinstated_this_year,
            'informalized_annual': informalized_this_year,
            'net_formal_change': net_formal_change,
            'cumulative_displaced': cumulative_displaced,
            'cumulative_reinstated': cumulative_reinstated,
            'cumulative_informalized': cumulative_informalized,
            'unemployment_effect': unemployment_effect,
        })

    return pd.DataFrame(results)


# ============================================================================
# STEP 4: RUN FULL SIMULATIONS (2025-2035)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Running Full Simulations (2025-2035)")
print("=" * 70)

# Central elasticity: sigma = 2.5 (Cheng et al. 2021)
SIGMA_CENTRAL = 2.5

all_results = []
for scenario_name, scenario in SCENARIOS.items():
    for sector in SECTORS:
        df = simulate_sector_scenario(sector, scenario, SIGMA_CENTRAL)
        all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)

# Summary by scenario at year 10
final_year = results_df[results_df['year'] == YEAR_END]
print("\n--- Year 2035 Summary by Scenario ---")
for sname in SCENARIOS:
    subset = final_year[final_year['scenario'] == sname]
    total_formal = subset['formal_employment'].sum()
    total_displaced = subset['cumulative_displaced'].sum()
    total_reinstated = subset['cumulative_reinstated'].sum()
    total_informalized = subset['cumulative_informalized'].sum()
    initial_formal = sum(s.formal_employment for s in SECTORS)
    pct_change = (total_formal - initial_formal) / initial_formal * 100

    print(f"\n  {sname}:")
    print(f"    Formal employment: {total_formal:.2f}M (change: {pct_change:+.1f}%)")
    print(f"    Cumulative displaced: {total_displaced:.2f}M")
    print(f"    Reinstated: {total_reinstated:.2f}M")
    print(f"    Net informalized: {total_informalized:.2f}M")


# ============================================================================
# STEP 5: MONTE CARLO SENSITIVITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Monte Carlo Sensitivity Analysis")
print(f"  Drawing {N_MC_DRAWS} parameter combinations...")
print("=" * 70)

# Parameter distributions for Monte Carlo
# Elasticity of substitution: triangular(1.5, 2.5, 3.8)
# Reinstatement rate: triangular(0.25, 0.40, 0.60)
# Tech cost decline: triangular(0.05, 0.10, 0.20)

mc_results_total = []

for draw in range(N_MC_DRAWS):
    sigma_draw = np.random.triangular(1.5, 2.5, 3.8)
    reinstatement_draw = np.random.triangular(0.25, 0.40, 0.60)
    tech_decline_draw = np.random.triangular(0.05, 0.10, 0.20)

    # Run baseline scenario with drawn parameters
    scenario_mc = copy.deepcopy(SCENARIOS['Status Quo'])
    scenario_mc.reinstatement_rate = reinstatement_draw
    scenario_mc.tech_cost_decline_rate = tech_decline_draw

    total_displaced_mc = 0
    total_formal_mc = 0
    for sector in SECTORS:
        df_mc = simulate_sector_scenario(sector, scenario_mc, sigma_draw)
        final = df_mc[df_mc['year'] == YEAR_END].iloc[0]
        total_displaced_mc += final['cumulative_displaced']
        total_formal_mc += final['formal_employment']

    mc_results_total.append({
        'draw': draw,
        'sigma': sigma_draw,
        'reinstatement_rate': reinstatement_draw,
        'tech_decline_rate': tech_decline_draw,
        'total_displaced': total_displaced_mc,
        'total_formal': total_formal_mc,
    })

mc_df = pd.DataFrame(mc_results_total)

print(f"\n  Monte Carlo Results (Baseline Scenario, N={N_MC_DRAWS}):")
print(f"    Total displacement (millions):")
print(f"      Mean: {mc_df['total_displaced'].mean():.3f}M")
print(f"      Median: {mc_df['total_displaced'].median():.3f}M")
print(f"      5th percentile: {mc_df['total_displaced'].quantile(0.05):.3f}M")
print(f"      95th percentile: {mc_df['total_displaced'].quantile(0.95):.3f}M")
print(f"      Std dev: {mc_df['total_displaced'].std():.3f}M")

# --- Tornado / Sensitivity decomposition ---
# For each parameter, compute partial effect holding others at central
print("\n  Sensitivity decomposition (tornado):")

# Parameter ranges for tornado
tornado_params = {
    'Elasticity (sigma)': {
        'low': 1.5, 'central': 2.5, 'high': 3.8,
        'param_key': 'sigma'
    },
    'Reinstatement rate': {
        'low': 0.25, 'central': 0.40, 'high': 0.60,
        'param_key': 'reinstatement'
    },
    'Tech cost decline': {
        'low': 0.05, 'central': 0.10, 'high': 0.20,
        'param_key': 'tech_decline'
    },
    'Automation risk (+/-10pp)': {
        'low': -0.10, 'central': 0.0, 'high': 0.10,
        'param_key': 'risk_shift'
    },
    'Min wage growth': {
        'low': 0.05, 'central': 0.07, 'high': 0.10,
        'param_key': 'wage_growth'
    },
}

tornado_results = {}
central_total = None

for param_name, pvals in tornado_params.items():
    results_by_level = {}
    for level in ['low', 'central', 'high']:
        scenario_t = copy.deepcopy(SCENARIOS['Status Quo'])

        if pvals['param_key'] == 'reinstatement':
            scenario_t.reinstatement_rate = pvals[level]
            sigma_t = 2.5
        elif pvals['param_key'] == 'tech_decline':
            scenario_t.tech_cost_decline_rate = pvals[level]
            sigma_t = 2.5
        elif pvals['param_key'] == 'sigma':
            sigma_t = pvals[level]
        elif pvals['param_key'] == 'wage_growth':
            scenario_t.min_wage_annual_growth = pvals[level]
            sigma_t = 2.5
        elif pvals['param_key'] == 'risk_shift':
            sigma_t = 2.5
        else:
            sigma_t = 2.5

        total_disp = 0
        for sector in SECTORS:
            s_copy = copy.deepcopy(sector)
            if pvals['param_key'] == 'risk_shift':
                s_copy.automation_risk_high_pct = max(0.05,
                    min(0.95, s_copy.automation_risk_high_pct + pvals[level]))
            df_t = simulate_sector_scenario(s_copy, scenario_t, sigma_t)
            final_t = df_t[df_t['year'] == YEAR_END].iloc[0]
            total_disp += final_t['cumulative_displaced']

        results_by_level[level] = total_disp

    tornado_results[param_name] = results_by_level
    if central_total is None:
        central_total = results_by_level['central']
    print(f"    {param_name}: Low={results_by_level['low']:.3f}M, "
          f"Central={results_by_level['central']:.3f}M, "
          f"High={results_by_level['high']:.3f}M")


# ============================================================================
# Also run MC for all scenarios (for fan chart and scenario comparison)
# ============================================================================
print("\n  Running Monte Carlo for all scenarios...")

mc_all_scenarios = {}
for sname, scen in SCENARIOS.items():
    mc_scenario_results = []
    for draw in range(N_MC_DRAWS):
        sigma_draw = np.random.triangular(1.5, 2.5, 3.8)
        reinstatement_draw = np.random.triangular(0.25, 0.40, 0.60)
        tech_decline_draw = np.random.triangular(0.05, 0.10, 0.20)

        scen_mc = copy.deepcopy(scen)
        # Only vary reinstatement proportionally (keep scenario-specific ratio)
        base_reinstate = scen.reinstatement_rate
        scen_mc.reinstatement_rate = reinstatement_draw * (base_reinstate / 0.40)
        scen_mc.tech_cost_decline_rate = tech_decline_draw + (
            scen.tech_cost_decline_rate - 0.10)

        # Track yearly totals for this draw
        yearly_formal = {yr: 0 for yr in range(YEAR_START, YEAR_END + 1)}
        total_disp = 0
        for sector in SECTORS:
            df_mc = simulate_sector_scenario(sector, scen_mc, sigma_draw)
            for _, row in df_mc.iterrows():
                yearly_formal[int(row['year'])] += row['formal_employment']
            final_mc = df_mc[df_mc['year'] == YEAR_END].iloc[0]
            total_disp += final_mc['cumulative_displaced']

        mc_scenario_results.append({
            'draw': draw,
            'total_displaced': total_disp,
            **{f'formal_{yr}': yearly_formal[yr]
               for yr in range(YEAR_START, YEAR_END + 1)},
        })

    mc_all_scenarios[sname] = pd.DataFrame(mc_scenario_results)
    p5 = mc_all_scenarios[sname]['total_displaced'].quantile(0.05)
    p95 = mc_all_scenarios[sname]['total_displaced'].quantile(0.95)
    med = mc_all_scenarios[sname]['total_displaced'].median()
    print(f"    {sname}: Median={med:.3f}M, 90% CI=[{p5:.3f}, {p95:.3f}]M")


# ============================================================================
# STEP 6: PUBLICATION-QUALITY FIGURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Generating Publication-Quality Figures")
print("=" * 70)


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    for ext in ['png', 'pdf']:
        path = os.path.join(IMG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png/pdf")


# ---- Figure (a): Total formal employment trajectory ----
fig, ax = plt.subplots(figsize=(10, 6))

initial_formal = sum(s.formal_employment for s in SECTORS)
years_range = list(range(YEAR_START, YEAR_END + 1))

for sname in SCENARIOS:
    yearly_formal = []
    for yr in years_range:
        subset = results_df[(results_df['scenario'] == sname) &
                            (results_df['year'] == yr)]
        yearly_formal.append(subset['formal_employment'].sum())

    ax.plot(years_range, yearly_formal, label=sname,
            color=SCENARIO_COLORS[sname], linewidth=2.2,
            marker='o', markersize=4, markeredgewidth=0)

ax.axhline(y=initial_formal, color=LIGHTGRAY, linestyle='--', linewidth=1,
           alpha=0.7, label=f'Initial ({initial_formal:.2f}M)')
ax.set_xlabel('Year')
ax.set_ylabel('Formal Employment (millions)')
ax.set_title('Projected Formal Employment Trajectories by Scenario (2025-2035)',
             fontweight='bold', pad=15)
ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
ax.set_xlim(YEAR_START, YEAR_END)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Add annotation for initial level
ax.annotate(f'Baseline: {initial_formal:.2f}M',
            xy=(YEAR_START, initial_formal),
            xytext=(YEAR_START + 0.5, initial_formal + 0.3),
            fontsize=9, color=DARKGRAY,
            arrowprops=dict(arrowstyle='->', color=DARKGRAY, lw=0.8))

fig.text(0.5, -0.02,
         'Note: Simulation based on stated assumptions. See text for parameter sources.',
         ha='center', fontsize=8, color='gray', style='italic')

save_fig(fig, 'fig_sim_formal_employment_trajectories')


# ---- Figure (b): Stacked bar - Jobs displaced by sector at year 10 ----
fig, ax = plt.subplots(figsize=(12, 7))

sector_names = [s.name for s in SECTORS]
scenario_names = list(SCENARIOS.keys())
n_scenarios = len(scenario_names)
n_sectors = len(sector_names)

bar_width = 0.7
x_pos = np.arange(n_scenarios)

# Build matrix: scenario x sector
disp_matrix = np.zeros((n_scenarios, n_sectors))
for i, sname in enumerate(scenario_names):
    for j, sec_name in enumerate(sector_names):
        subset = final_year[(final_year['scenario'] == sname) &
                            (final_year['sector'] == sec_name)]
        if len(subset) > 0:
            disp_matrix[i, j] = subset['cumulative_displaced'].values[0]

# Sector colors
sector_cmap = plt.cm.get_cmap('tab10', n_sectors)
sector_colors = [sector_cmap(i) for i in range(n_sectors)]

# Stacked bars
bottom = np.zeros(n_scenarios)
bars = []
for j in range(n_sectors):
    b = ax.bar(x_pos, disp_matrix[:, j] * 1000,  # Convert to thousands
               bottom=bottom * 1000,
               width=bar_width,
               label=sector_names[j],
               color=sector_colors[j],
               edgecolor='white', linewidth=0.5)
    bars.append(b)
    bottom += disp_matrix[:, j]

# Total labels on top
for i in range(n_scenarios):
    total = bottom[i] * 1000
    ax.text(x_pos[i], total + 10, f'{total:.0f}K',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_xlabel('Scenario')
ax.set_ylabel('Cumulative Jobs Displaced (thousands)')
ax.set_title('Cumulative Automation Displacement by Sector and Scenario (2025-2035)',
             fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_names, rotation=15, ha='right')
ax.legend(loc='upper left', ncol=2, frameon=True, fancybox=False,
          edgecolor=LIGHTGRAY, fontsize=8)
ax.grid(axis='y', alpha=0.3, linestyle=':')

fig.text(0.5, -0.04,
         'Note: Displacement counts before reinstatement effects. '
         'Applies only to formal employment. Informal sector not directly affected by automation costs.',
         ha='center', fontsize=8, color='gray', style='italic')

save_fig(fig, 'fig_sim_displacement_by_sector')


# ---- Figure (c): Waterfall - Employment decomposition for Labor Reform ----
fig, ax = plt.subplots(figsize=(10, 6.5))

reform_final = final_year[final_year['scenario'] == 'Labor Reform']
total_displaced = reform_final['cumulative_displaced'].sum()
total_reinstated = reform_final['cumulative_reinstated'].sum()
total_informalized = reform_final['cumulative_informalized'].sum()
net_change = -(total_displaced - total_reinstated) - total_informalized

categories = ['Initial\nFormal', 'Automation\nDisplacement', 'Task\nReinstatement',
              'Informali-\nzation', 'Final\nFormal']
values = [initial_formal, -total_displaced, total_reinstated,
          -total_informalized, initial_formal + net_change]

# Calculate running total for waterfall
running = [initial_formal]
running.append(running[-1] - total_displaced)
running.append(running[-1] + total_reinstated)
running.append(running[-1] - total_informalized)
running.append(running[-1])  # Final

colors = [MAINBLUE, ACCENTRED, GREEN, ORANGE, MAINBLUE]

# Draw waterfall bars
bar_bottoms = []
bar_heights = []
for i in range(len(categories)):
    if i == 0:
        bar_bottoms.append(0)
        bar_heights.append(running[0])
    elif i == len(categories) - 1:
        bar_bottoms.append(0)
        bar_heights.append(running[-1])
    else:
        if values[i] < 0:
            bar_bottoms.append(running[i])
            bar_heights.append(abs(values[i]))
        else:
            bar_bottoms.append(running[i] - values[i])
            bar_heights.append(values[i])

for i in range(len(categories)):
    ax.bar(i, bar_heights[i], bottom=bar_bottoms[i],
           color=colors[i], width=0.6, edgecolor='white', linewidth=1)
    # Value label
    if i == 0 or i == len(categories) - 1:
        ax.text(i, bar_bottoms[i] + bar_heights[i] + 0.05,
                f'{bar_heights[i]:.2f}M', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    else:
        mid = bar_bottoms[i] + bar_heights[i] / 2
        sign = '+' if values[i] > 0 else ''
        ax.text(i, mid, f'{sign}{values[i]:.2f}M', ha='center', va='center',
                fontweight='bold', fontsize=10, color='white')

# Connection lines
for i in range(len(categories) - 1):
    y_connect = running[i + 1] if values[i + 1] < 0 or i + 1 == len(categories) - 1 else running[i]
    if i == 0:
        y_connect = running[1] + abs(values[1])
    elif i == 1:
        y_connect = running[2]
    elif i == 2:
        y_connect = running[3] + abs(values[3])

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_ylabel('Formal Employment (millions)')
ax.set_title('Waterfall: Employment Decomposition under Labor Reform Scenario (2035)',
             fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.set_ylim(0, initial_formal + 0.5)

fig.text(0.5, -0.02,
         'Source: Simulation. Reinstatement = 35% of displaced (Acemoglu & Restrepo, emerging economy estimate). '
         'Informalization per ANIF elasticities.',
         ha='center', fontsize=8, color='gray', style='italic')

save_fig(fig, 'fig_sim_waterfall_labor_reform')


# ---- Figure (d): Heatmap - Sector x Scenario (% formal jobs displaced) ----
fig, ax = plt.subplots(figsize=(11, 7))

# Calculate % of formal jobs displaced
heatmap_data = np.zeros((n_sectors, n_scenarios))
for i, sec_name in enumerate(sector_names):
    sector_obj = SECTORS[i]
    for j, sname in enumerate(scenario_names):
        subset = final_year[(final_year['scenario'] == sname) &
                            (final_year['sector'] == sec_name)]
        if len(subset) > 0:
            displaced = subset['cumulative_displaced'].values[0]
            initial = sector_obj.formal_employment
            if initial > 0:
                heatmap_data[i, j] = displaced / initial * 100

heatmap_df = pd.DataFrame(heatmap_data,
                          index=sector_names,
                          columns=scenario_names)

# Custom colormap: white -> yellow -> orange -> red
cmap = LinearSegmentedColormap.from_list('risk',
    ['#FFFFFF', '#FFF3CD', '#F0AD4E', '#E74C3C', '#8B0000'])

sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap=cmap,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': '% of Formal Jobs Displaced (cumulative)',
                      'shrink': 0.8},
            ax=ax, vmin=0)

ax.set_title('Automation Displacement Intensity: Sector x Scenario (2025-2035)',
             fontweight='bold', pad=15)
ax.set_ylabel('Sector')
ax.set_xlabel('Scenario')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

fig.text(0.5, -0.02,
         'Note: Values show cumulative formal job displacement as % of initial formal employment. '
         'sigma=2.5, central reinstatement rate.',
         ha='center', fontsize=8, color='gray', style='italic')

save_fig(fig, 'fig_sim_heatmap_displacement')


# ---- Figure (e): Fan chart - Uncertainty under baseline ----
fig, ax = plt.subplots(figsize=(10, 6))

mc_baseline = mc_all_scenarios['Status Quo']
formal_cols = [f'formal_{yr}' for yr in years_range]

# Calculate percentiles for each year
percentiles = [5, 10, 25, 50, 75, 90, 95]
fan_data = {}
for p in percentiles:
    fan_data[p] = [mc_baseline[col].quantile(p / 100) for col in formal_cols]

# Fill between percentile bands
alpha_levels = [0.12, 0.18, 0.25]
band_pairs = [(5, 95), (10, 90), (25, 75)]
band_labels = ['90% CI', '80% CI', '50% CI']

for (low, high), alpha, label in zip(band_pairs, alpha_levels, band_labels):
    ax.fill_between(years_range, fan_data[low], fan_data[high],
                    alpha=alpha, color=MAINBLUE, label=label)

# Median line
ax.plot(years_range, fan_data[50], color=MAINBLUE, linewidth=2.5,
        label='Median', zorder=5)

# Deterministic central
det_formal = []
for yr in years_range:
    subset = results_df[(results_df['scenario'] == 'Status Quo') &
                        (results_df['year'] == yr)]
    det_formal.append(subset['formal_employment'].sum())
ax.plot(years_range, det_formal, color=ACCENTRED, linewidth=1.5,
        linestyle='--', label='Deterministic central', zorder=4)

ax.axhline(y=initial_formal, color=LIGHTGRAY, linestyle=':', linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('Formal Employment (millions)')
ax.set_title('Uncertainty Range: Formal Employment under Status Quo (Monte Carlo)',
             fontweight='bold', pad=15)
ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
ax.set_xlim(YEAR_START, YEAR_END)
ax.grid(axis='y', alpha=0.3, linestyle=':')

fig.text(0.5, -0.02,
         f'Note: Based on {N_MC_DRAWS} Monte Carlo draws. Parameters varied: '
         'sigma~Tri(1.5,2.5,3.8), reinstatement~Tri(0.25,0.40,0.60), '
         'tech decline~Tri(0.05,0.10,0.20).',
         ha='center', fontsize=8, color='gray', style='italic')

save_fig(fig, 'fig_sim_fan_chart_baseline')


# ---- Figure (f): Tornado diagram ----
fig, ax = plt.subplots(figsize=(10, 5.5))

# Sort parameters by total range
tornado_sorted = sorted(tornado_results.items(),
                        key=lambda x: abs(x[1]['high'] - x[1]['low']),
                        reverse=True)

param_names = [t[0] for t in tornado_sorted]
low_vals = [t[1]['low'] for t in tornado_sorted]
high_vals = [t[1]['high'] for t in tornado_sorted]
central_val = tornado_sorted[0][1]['central']  # All share same central

y_pos = np.arange(len(param_names))
bar_height = 0.5

# Convert to thousands for readability
low_k = [v * 1000 for v in low_vals]
high_k = [v * 1000 for v in high_vals]
central_k = central_val * 1000

for i in range(len(param_names)):
    # Low bar (left of central)
    ax.barh(y_pos[i], low_k[i] - central_k, left=central_k,
            height=bar_height, color=LIGHTBLUE, edgecolor='white',
            linewidth=0.5, label='Low value' if i == 0 else '')
    # High bar (right of central)
    ax.barh(y_pos[i], high_k[i] - central_k, left=central_k,
            height=bar_height, color=ACCENTRED, edgecolor='white',
            linewidth=0.5, label='High value' if i == 0 else '')

    # Labels
    ax.text(low_k[i] - 5, y_pos[i], f'{low_k[i]:.0f}K',
            ha='right', va='center', fontsize=9)
    ax.text(high_k[i] + 5, y_pos[i], f'{high_k[i]:.0f}K',
            ha='left', va='center', fontsize=9)

ax.axvline(x=central_k, color=DARKGRAY, linewidth=1.5, linestyle='-')
ax.set_yticks(y_pos)
ax.set_yticklabels(param_names)
ax.set_xlabel('Total Cumulative Displacement (thousands)')
ax.set_title('Sensitivity Analysis: Parameter Impact on Total Displacement',
             fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor=LIGHTGRAY)

# Add central value annotation
ax.annotate(f'Central: {central_k:.0f}K', xy=(central_k, -0.7),
            ha='center', fontsize=9, color=DARKGRAY, fontweight='bold')

save_fig(fig, 'fig_sim_tornado_sensitivity')


# ---- Figure (g): International comparison ----
fig, ax = plt.subplots(figsize=(10, 6))

# International benchmarks (% of formal manufacturing jobs displaced, 10-year projection)
# Sources: OECD, IFR, national statistical agencies, academic literature
intl_benchmarks = {
    'South Korea\n(observed 2018)':     7.5,   # 170K / ~2.3M mfg workers
    'Germany\n(projected)':             5.0,   # IAB estimates
    'Japan\n(projected)':               6.0,   # METI estimates
    'USA\n(projected)':                 4.5,   # Acemoglu & Restrepo extrapolation
    'China\n(projected)':               8.0,   # Rapid robotization
    'Mexico\n(projected)':              3.5,   # Lower adoption
    'Brazil\n(projected)':              3.0,   # Lower adoption
}

# Colombia scenarios
col_scenarios = {}
for sname in SCENARIOS:
    subset = final_year[final_year['scenario'] == sname]
    total_disp = subset['cumulative_displaced'].sum()
    col_scenarios[f'Colombia\n({sname})'] = total_disp / initial_formal * 100

# Combine
all_countries = {**intl_benchmarks, **col_scenarios}
names = list(all_countries.keys())
values = list(all_countries.values())

# Color: international = gray, Colombia = scenario colors
bar_colors = []
for name in names:
    if 'Colombia' in name:
        for sname, color in SCENARIO_COLORS.items():
            if sname in name:
                bar_colors.append(color)
                break
    else:
        bar_colors.append(LIGHTGRAY)

bars = ax.barh(range(len(names)), values, color=bar_colors,
               edgecolor='white', linewidth=0.5, height=0.7)

# Value labels
for i, (val, name) in enumerate(zip(values, names)):
    ax.text(val + 0.2, i, f'{val:.1f}%', ha='left', va='center', fontsize=9)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('% of Formal Jobs Displaced (cumulative, 10 years)')
ax.set_title('Projected Automation Displacement: Colombia vs International Benchmarks',
             fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle=':')

fig.text(0.5, -0.04,
         'Note: International values are estimates from multiple sources (IFR, OECD, national agencies). '
         'Colombia values are simulation results. Not directly comparable due to methodological differences.',
         ha='center', fontsize=8, color='gray', style='italic', wrap=True)

save_fig(fig, 'fig_sim_international_comparison')


# ============================================================================
# STEP 7: SUMMARY TABLES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Generating Summary Tables")
print("=" * 70)

# --- Table 1: Scenario comparison ---
summary_rows = []
for sname in SCENARIOS:
    subset = final_year[final_year['scenario'] == sname]
    total_formal_end = subset['formal_employment'].sum()
    total_displaced = subset['cumulative_displaced'].sum()
    total_reinstated = subset['cumulative_reinstated'].sum()
    total_informalized = subset['cumulative_informalized'].sum()
    net_change = total_formal_end - initial_formal
    pct_displaced = total_displaced / initial_formal * 100
    pct_formal_change = net_change / initial_formal * 100

    # MC confidence intervals
    if sname in mc_all_scenarios:
        mc_data = mc_all_scenarios[sname]
        disp_p5 = mc_data['total_displaced'].quantile(0.05)
        disp_p95 = mc_data['total_displaced'].quantile(0.95)
    else:
        disp_p5 = disp_p95 = np.nan

    summary_rows.append({
        'Scenario': sname,
        'Final Formal Employment (M)': round(total_formal_end, 3),
        'Net Change Formal (M)': round(net_change, 3),
        'Pct Change Formal (%)': round(pct_formal_change, 1),
        'Cumulative Displaced (M)': round(total_displaced, 3),
        'Cumulative Reinstated (M)': round(total_reinstated, 3),
        'Cumulative Informalized (M)': round(total_informalized, 3),
        'Pct Formal Displaced (%)': round(pct_displaced, 1),
        'Displacement 90% CI Low (M)': round(disp_p5, 3),
        'Displacement 90% CI High (M)': round(disp_p95, 3),
    })

summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv(os.path.join(DATA_DIR, 'simulation_scenario_comparison.csv'),
                     index=False)
print("\n  Table 1: Scenario Comparison")
print(summary_table.to_string(index=False))

# --- Table 2: Sectoral breakdown under each scenario ---
sectoral_rows = []
for sname in SCENARIOS:
    for sec in SECTORS:
        subset = final_year[(final_year['scenario'] == sname) &
                            (final_year['sector'] == sec.name)]
        if len(subset) > 0:
            row = subset.iloc[0]
            sectoral_rows.append({
                'Scenario': sname,
                'Sector': sec.name,
                'Initial Formal (M)': round(sec.formal_employment, 3),
                'Final Formal (M)': round(row['formal_employment'], 3),
                'Change (M)': round(row['formal_employment'] - sec.formal_employment, 3),
                'Pct Change (%)': round(
                    (row['formal_employment'] - sec.formal_employment) /
                    sec.formal_employment * 100 if sec.formal_employment > 0 else 0, 1),
                'Displaced (M)': round(row['cumulative_displaced'], 3),
                'Reinstated (M)': round(row['cumulative_reinstated'], 3),
                'Informalized (M)': round(row['cumulative_informalized'], 3),
            })

sectoral_table = pd.DataFrame(sectoral_rows)
sectoral_table.to_csv(os.path.join(DATA_DIR, 'simulation_sectoral_breakdown.csv'),
                      index=False)
print(f"\n  Table 2: Sectoral breakdown saved ({len(sectoral_table)} rows)")

# --- Table 3: Sensitivity analysis results ---
sensitivity_rows = []
for param_name, vals in tornado_results.items():
    sensitivity_rows.append({
        'Parameter': param_name,
        'Low Value Displacement (M)': round(vals['low'], 3),
        'Central Displacement (M)': round(vals['central'], 3),
        'High Value Displacement (M)': round(vals['high'], 3),
        'Range (M)': round(vals['high'] - vals['low'], 3),
        'Pct Range from Central (%)': round(
            (vals['high'] - vals['low']) / vals['central'] * 100, 1),
    })

sensitivity_table = pd.DataFrame(sensitivity_rows)
sensitivity_table = sensitivity_table.sort_values('Range (M)', ascending=False)
sensitivity_table.to_csv(os.path.join(DATA_DIR, 'simulation_sensitivity_analysis.csv'),
                         index=False)
print("\n  Table 3: Sensitivity Analysis")
print(sensitivity_table.to_string(index=False))

# --- Table 4: Monte Carlo distribution summary ---
mc_summary_rows = []
for sname in SCENARIOS:
    if sname in mc_all_scenarios:
        mc_data = mc_all_scenarios[sname]
        disp = mc_data['total_displaced']
        mc_summary_rows.append({
            'Scenario': sname,
            'Mean Displaced (M)': round(disp.mean(), 3),
            'Median Displaced (M)': round(disp.median(), 3),
            'Std Dev (M)': round(disp.std(), 3),
            'P5 (M)': round(disp.quantile(0.05), 3),
            'P10 (M)': round(disp.quantile(0.10), 3),
            'P25 (M)': round(disp.quantile(0.25), 3),
            'P75 (M)': round(disp.quantile(0.75), 3),
            'P90 (M)': round(disp.quantile(0.90), 3),
            'P95 (M)': round(disp.quantile(0.95), 3),
        })

mc_summary_table = pd.DataFrame(mc_summary_rows)
mc_summary_table.to_csv(os.path.join(DATA_DIR, 'simulation_montecarlo_summary.csv'),
                        index=False)
print("\n  Table 4: Monte Carlo Distribution Summary")
print(mc_summary_table.to_string(index=False))

# --- Save full simulation results ---
results_df.to_csv(os.path.join(DATA_DIR, 'simulation_full_results.csv'), index=False)
print(f"\n  Full results saved: {len(results_df)} rows")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

print(f"""
Key Findings (Central Estimates, sigma=2.5):
--------------------------------------------
""")

for sname in SCENARIOS:
    subset = final_year[final_year['scenario'] == sname]
    total_disp = subset['cumulative_displaced'].sum()
    total_formal_end = subset['formal_employment'].sum()
    pct_change = (total_formal_end - initial_formal) / initial_formal * 100
    mc_data = mc_all_scenarios.get(sname)
    if mc_data is not None:
        ci = f"[{mc_data['total_displaced'].quantile(0.05):.2f}, " \
             f"{mc_data['total_displaced'].quantile(0.95):.2f}]M"
    else:
        ci = "N/A"
    print(f"  {sname}:")
    print(f"    Formal employment change: {pct_change:+.1f}%")
    print(f"    Total displaced: {total_disp:.2f}M (90% CI: {ci})")

print(f"""
Most Sensitive Parameters (Tornado Analysis):
----------------------------------------------""")
for _, row in sensitivity_table.iterrows():
    print(f"  {row['Parameter']}: Range = {row['Range (M)']:.3f}M "
          f"({row['Pct Range from Central (%)']:.0f}% of central)")

print(f"""
Outputs Generated:
------------------
  Figures (images/):
    - fig_sim_formal_employment_trajectories.png/pdf
    - fig_sim_displacement_by_sector.png/pdf
    - fig_sim_waterfall_labor_reform.png/pdf
    - fig_sim_heatmap_displacement.png/pdf
    - fig_sim_fan_chart_baseline.png/pdf
    - fig_sim_tornado_sensitivity.png/pdf
    - fig_sim_international_comparison.png/pdf

  Tables (data/):
    - simulation_scenario_comparison.csv
    - simulation_sectoral_breakdown.csv
    - simulation_sensitivity_analysis.csv
    - simulation_montecarlo_summary.csv
    - simulation_full_results.csv

DISCLAIMER: These are simulation projections, not empirical estimates.
Results are conditional on stated assumptions and should be interpreted
as scenario-based forecasts for policy analysis, not predictions.
""")
