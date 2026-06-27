#!/usr/bin/env python3
"""
Community-Location Visualization

Generates three panels per city:
  1. Geographic community map  — grid colored by community ID
  2. Community POI composition — stacked bar of dominant POI categories
  3. Community hourly activity — mean active users/vehicles per hour

Usage:
    cd DynamicGraphMobilityGeneration
    python3 util/visual.py                 # both cities
    python3 util/visual.py --city shanghai
    python3 util/visual.py --city shenzhen
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ============================================================
# CITY CONFIGS
# ============================================================

CITY_CONFIGS = {
    'shanghai': {
        'label':     'Shanghai (Mobile Signaling)',
        'data_dir':  Path('data/shanghai'),
        'graph_dir': 'dynamic_graph',
        'grid_rows': 20,
        'grid_cols': 20,
        'lon_min':   121.35, 'lon_max': 121.55,
        'lat_min':    31.15, 'lat_max':  31.35,
        'user_label': 'Users',
    },
    'shenzhen': {
        'label':     'Shenzhen (Private Car GPS)',
        'data_dir':  Path('data/shenzhen'),
        'graph_dir': 'dynamic_graph',
        'grid_rows': 20,
        'grid_cols': 25,
        'lon_min':   113.90, 'lon_max': 114.15,
        'lat_min':    22.50, 'lat_max':  22.70,
        'user_label': 'Vehicles',
    },
}

POI_CATEGORIES = [
    'Transportation Facilities', 'Leisure & Entertainment',
    'Companies & Enterprises',   'Healthcare',
    'Commercial & Residential',  'Tourist Attractions',
    'Automotive',                'Life Services',
    'Science & Education & Culture', 'Shopping & Consumer Goods',
    'Sports & Fitness',          'Hotels & Accommodations',
    'Financial Institutions',    'Dining & Cuisine',
]

# Short labels for charts
POI_SHORT = {
    'Transportation Facilities':     'Transport',
    'Leisure & Entertainment':       'Leisure',
    'Companies & Enterprises':       'Companies',
    'Healthcare':                    'Healthcare',
    'Commercial & Residential':      'Commercial',
    'Tourist Attractions':           'Tourism',
    'Automotive':                    'Automotive',
    'Life Services':                 'Life Svc',
    'Science & Education & Culture': 'Edu/Sci',
    'Shopping & Consumer Goods':     'Shopping',
    'Sports & Fitness':              'Sports',
    'Hotels & Accommodations':       'Hotels',
    'Financial Institutions':        'Finance',
    'Dining & Cuisine':              'Dining',
}

# Colormap for up to 20 communities
COMM_CMAP = matplotlib.colormaps.get_cmap('tab20').resampled(20)

# POI colormap (14 categories)
POI_COLORS = [plt.cm.Set3(i / 13) for i in range(14)]


# ============================================================
# DATA LOADING
# ============================================================

def load_city_data(city: str, base_dir: Path) -> dict:
    cfg = CITY_CONFIGS[city]
    data_dir  = base_dir / cfg['data_dir']
    graph_dir = data_dir / cfg['graph_dir']
    pat_dir   = graph_dir / 'extracted_pattern'

    location = pd.read_csv(data_dir / 'location.csv')

    with open(pat_dir / 'community_fixed.json') as f:
        community_fixed = json.load(f)

    with open(pat_dir / 'community_summary.json') as f:
        community_summary = json.load(f)

    return {
        'cfg':              cfg,
        'location':         location,
        'community_fixed':  community_fixed,
        'community_summary': community_summary,
    }


# ============================================================
# PANEL 1 — Geographic community map
# ============================================================

def plot_community_map(ax, data: dict):
    cfg   = data['cfg']
    loc   = data['location']
    r2c   = data['community_fixed']['region_to_comm']
    n_com = data['community_fixed']['n_communities']
    rows, cols = cfg['grid_rows'], cfg['grid_cols']

    # Build 2-D grid array; -1 = no data
    grid = np.full((rows, cols), -1, dtype=int)
    for region_id, comm_id in r2c.items():
        rid = int(region_id)
        r, c = divmod(rid, cols)
        grid[r, c] = int(comm_id)

    # Discrete colormap
    bounds  = np.arange(-0.5, n_com + 0.5, 1)
    norm    = mcolors.BoundaryNorm(bounds, COMM_CMAP.N)

    # imshow: row 0 = south → flip vertically so south is at bottom
    im = ax.imshow(
        grid,
        origin='lower',          # row 0 at bottom
        cmap=COMM_CMAP,
        norm=norm,
        aspect='equal',
        interpolation='nearest',
    )

    # Axis ticks → real coordinates
    lon_ticks = np.linspace(cfg['lon_min'], cfg['lon_max'], 5)
    lat_ticks = np.linspace(cfg['lat_min'], cfg['lat_max'], 5)
    ax.set_xticks(np.linspace(-0.5, cols - 0.5, 5))
    ax.set_xticklabels([f'{v:.2f}°E' for v in lon_ticks], fontsize=7)
    ax.set_yticks(np.linspace(-0.5, rows - 0.5, 5))
    ax.set_yticklabels([f'{v:.2f}°N' for v in lat_ticks], fontsize=7)

    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude',  fontsize=9)
    ax.set_title(f'Community Map  ({n_com} communities, {rows}×{cols} grid)',
                 fontsize=10, fontweight='bold')

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)

    # Legend patches
    patches = [
        mpatches.Patch(color=COMM_CMAP(i / (n_com - 1) if n_com > 1 else 0),
                       label=f'Comm {i}')
        for i in range(n_com)
    ]
    ncol = max(1, n_com // 4)
    ax.legend(handles=patches, loc='upper right',
              fontsize=6, ncol=ncol, framealpha=0.8,
              handlelength=1.0, handleheight=0.8, columnspacing=0.5)

    return im


# ============================================================
# PANEL 2 — Community POI composition (stacked bar)
# ============================================================

def plot_poi_composition(ax, data: dict):
    cfg   = data['cfg']
    loc   = data['location']
    r2c   = data['community_fixed']['region_to_comm']
    n_com = data['community_fixed']['n_communities']

    # Add community column to location
    loc = loc.copy()
    loc['community'] = loc['region_id'].astype(str).map(r2c)
    loc = loc.dropna(subset=['community'])
    loc['community'] = loc['community'].astype(int)

    # Mean POI composition per community
    # Only use categories that actually appear in location.csv
    available_cats = [c for c in POI_CATEGORIES if c in loc.columns]
    comm_poi = (loc.groupby('community')[available_cats].mean())

    # Normalize rows to sum=1 (some regions have all-zero POI)
    row_sum = comm_poi.sum(axis=1)
    comm_poi = comm_poi.divide(row_sum.where(row_sum > 0, np.nan), axis=0).fillna(0)

    # Stacked bar
    x    = np.arange(n_com)
    bottom = np.zeros(n_com)
    for i, cat in enumerate(available_cats):
        vals = comm_poi[cat].reindex(range(n_com), fill_value=0).values
        ax.bar(x, vals, bottom=bottom,
               color=POI_COLORS[i], label=POI_SHORT.get(cat, cat),
               width=0.8, edgecolor='white', linewidth=0.3)
        bottom += vals

    ax.set_xlim(-0.5, n_com - 0.5)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(n_com)], fontsize=7)
    ax.set_xlabel('Community', fontsize=9)
    ax.set_ylabel('Avg. POI Composition', fontsize=9)
    ax.set_title('Community POI Composition', fontsize=10, fontweight='bold')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Region count annotations above bars
    comm_sizes = loc['community'].value_counts().sort_index()
    ax.text(-0.45, 1.02, '#loc:', ha='left', va='bottom', fontsize=6, color='#555')
    for ci in range(n_com):
        sz = comm_sizes.get(ci, 0)
        ax.text(ci, 1.02, str(sz), ha='center', va='bottom', fontsize=6, color='#444')

    # Legend (two columns, small font)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=6, ncol=1, framealpha=0.8,
              handlelength=1.2, handleheight=0.9)


# ============================================================
# PANEL 3 — Hourly activity per community
# ============================================================

def plot_hourly_activity(ax, data: dict):
    cfg     = data['cfg']
    summary = data['community_summary']
    n_com   = data['community_fixed']['n_communities']

    hours = [entry['hour'] for entry in summary]

    for comm_id in range(n_com):
        series = []
        for entry in summary:
            mus = entry.get('mean_users_per_comm', {})
            series.append(mus.get(str(comm_id), 0.0))
        color = COMM_CMAP(comm_id / (n_com - 1) if n_com > 1 else 0)
        ax.plot(hours, series, color=color, linewidth=1.2,
                label=f'C{comm_id}', alpha=0.85)

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=7)
    ax.set_xlabel('Hour of Day', fontsize=9)
    ax.set_ylabel(f'Mean Active {cfg["user_label"]}', fontsize=9)
    ax.set_title('Hourly Community Activity', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ncol = max(1, n_com // 4)
    ax.legend(loc='upper right', fontsize=6, ncol=ncol,
              framealpha=0.8, handlelength=1.2)


# ============================================================
# MAIN FIGURE BUILDER
# ============================================================

def visualize_city(city: str, base_dir: Path, out_dir: Path):
    print(f'\n[{city.upper()}] Loading data ...')
    data = load_city_data(city, base_dir)
    cfg  = data['cfg']

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(cfg['label'], fontsize=13, fontweight='bold', y=1.01)

    # Layout: 3 panels side-by-side
    # Panel 1 (map) takes more width
    ax1 = fig.add_axes([0.03, 0.08, 0.30, 0.85])
    ax2 = fig.add_axes([0.42, 0.10, 0.32, 0.80])
    ax3 = fig.add_axes([0.79, 0.10, 0.20, 0.80])

    print(f'[{city.upper()}] Plotting community map ...')
    plot_community_map(ax1, data)

    print(f'[{city.upper()}] Plotting POI composition ...')
    plot_poi_composition(ax2, data)

    print(f'[{city.upper()}] Plotting hourly activity ...')
    plot_hourly_activity(ax3, data)

    out_path = out_dir / f'communities_{city}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[{city.upper()}] Saved → {out_path}')


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Community-Location Visualization')
    parser.add_argument('--city', choices=['shanghai', 'shenzhen', 'both'],
                        default='both', help='City to visualize (default: both)')
    args = parser.parse_args()

    # Resolve base directory relative to this script's location
    base_dir = Path(__file__).parent.parent   # DynamicGraphMobilityGeneration/
    out_dir  = base_dir / 'figures'
    out_dir.mkdir(exist_ok=True)

    cities = ['shanghai', 'shenzhen'] if args.city == 'both' else [args.city]
    for city in cities:
        try:
            visualize_city(city, base_dir, out_dir)
        except Exception as e:
            print(f'[ERROR] {city}: {e}')
            raise

    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
