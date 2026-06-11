import numpy as np
import matplotlib.pyplot as plt
import json
import os

GAMMA = 1.4
RHO_WATER = 1000.0
RHO_AIR = 1.225
P_ATM = 101325.0
G = 9.81
C_DRAG = 0.5

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#533483',
    'legend.labelcolor': '#e0e0e0',
    'grid.color': '#2a2a4e',
    'grid.alpha': 0.5,
})

COLORS = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
RATIO_LABELS = ['1/16', '1/8', '1/4', '1/2']
BOTTLE_VOL = 0.0015
WATER_VOLUMES = [BOTTLE_VOL / 16, BOTTLE_VOL / 8, BOTTLE_VOL / 4, BOTTLE_VOL / 2]


class BottleRocket:
    def __init__(self, V_bottle, water_volume, P_initial, m_bottle,
                 d_nozzle, C_discharge):
        self.V_bottle = V_bottle
        self.water_vol = water_volume
        self.P_initial = P_initial
        self.m_bottle = m_bottle
        self.A_nozzle = np.pi * d_nozzle**2 / 4
        self.C_d = C_discharge

        d_bottle = 0.09
        self.A_cross = np.pi * d_bottle**2 / 4

        self.V_air0 = V_bottle - water_volume
        self.m_water0 = water_volume * RHO_WATER
        self.reset()

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.m_w = self.m_water0
        self.V_air = self.V_air0
        self.t = 0.0
        self.history = {'t': [], 'x': [], 'v': [], 'a': [], 'F': [], 'P': []}

    @property
    def total_mass(self):
        return self.m_bottle + self.m_w

    def derivatives(self, state):
        _, v, m_w, V_air = state

        m_total = self.m_bottle + max(m_w, 0)
        drag = 0.5 * RHO_AIR * C_DRAG * self.A_cross * v * abs(v)

        if m_w <= 0:
            return [v, -G - drag / m_total, 0.0, 0.0]

        P_air = self.P_initial * (self.V_air0 / V_air) ** GAMMA
        P_gauge = P_air - P_ATM

        if P_gauge <= 0:
            return [v, -G - drag / m_total, 0.0, 0.0]

        v_e = np.sqrt(2 * P_gauge / RHO_WATER)
        m_dot = RHO_WATER * self.A_nozzle * v_e * self.C_d
        F = m_dot * v_e
        dV_air = m_dot / RHO_WATER
        a = (F - m_total * G - drag) / m_total

        return [v, a, -m_dot, dV_air]

    def step_rk4(self, dt):
        state = [self.x, self.v, self.m_w, self.V_air]

        k1 = self.derivatives(state)
        s2 = [state[i] + 0.5 * dt * k1[i] for i in range(4)]
        k2 = self.derivatives(s2)
        s3 = [state[i] + 0.5 * dt * k2[i] for i in range(4)]
        k3 = self.derivatives(s3)
        s4 = [state[i] + dt * k3[i] for i in range(4)]
        k4 = self.derivatives(s4)

        for i in range(4):
            state[i] += dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6

        state[2] = max(state[2], 0.0)
        state[3] = min(state[3], self.V_bottle)

        self.x, self.v, self.m_w, self.V_air = state
        self.t += dt

    def record_state(self):
        m_total = self.total_mass
        P_air = self.P_initial * (self.V_air0 / max(self.V_air, 1e-12)) ** GAMMA
        P_gauge = max(P_air - P_ATM, 0)

        if self.m_w > 0 and P_gauge > 0:
            v_e = np.sqrt(2 * P_gauge / RHO_WATER)
            m_dot = RHO_WATER * self.A_nozzle * v_e * self.C_d
            F = m_dot * v_e
        else:
            F = 0.0

        drag = 0.5 * RHO_AIR * C_DRAG * self.A_cross * self.v * abs(self.v)
        a = (F - m_total * G - drag) / m_total

        self.history['t'].append(self.t)
        self.history['x'].append(self.x)
        self.history['v'].append(self.v)
        self.history['a'].append(a)
        self.history['F'].append(F)
        self.history['P'].append(P_air / 1e5)

    def simulate(self, dt=0.00005, max_time=15.0, min_height=-0.1):
        self.reset()
        self.record_state()

        while self.t < max_time:
            self.step_rk4(dt)
            self.record_state()

            if self.x < min_height and self.t > 0.2:
                break

        return {k: np.array(v) for k, v in self.history.items()}


def run_all_simulations(config):
    results = {}
    for i, v_w in enumerate(WATER_VOLUMES):
        rocket = BottleRocket(
            V_bottle=config['V_bottle'],
            water_volume=v_w,
            P_initial=config['P_initial'],
            m_bottle=config['m_bottle'],
            d_nozzle=config['d_nozzle'],
            C_discharge=config['C_discharge'],
        )
        data = rocket.simulate(dt=config.get('dt', 5e-5))
        results[RATIO_LABELS[i]] = data
    return results


def make_plots(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_h, ax_v, ax_a, ax_F = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    summaries = {}
    for i, (label, data) in enumerate(results.items()):
        c = COLORS[i]

        ax_h.plot(data['t'], data['x'], color=c, label=label, linewidth=1.5)
        ax_v.plot(data['t'], data['v'], color=c, label=label, linewidth=1.5)
        ax_a.plot(data['t'], data['a'], color=c, label=label, linewidth=1.5)
        ax_F.plot(data['t'], data['F'], color=c, label=label, linewidth=1.5)

        max_h = np.max(data['x'])
        max_v = np.max(data['v'])
        max_a = np.max(data['a'])
        max_F = np.max(data['F'])
        burn_end = np.argmax(data['F'] < 1)
        burn_t = data['t'][burn_end] if np.any(data['F'] < 1) else data['t'][-1]
        summaries[label] = {
            'max_height_m': round(float(max_h), 2),
            'max_velocity_ms': round(float(max_v), 2),
            'max_accel_gs': round(float(max_a / G), 2),
            'max_thrust_N': round(float(max_F), 2),
            'burn_time_s': round(float(burn_t), 3),
            'water_volume_ml': round(float(WATER_VOLUMES[i] * 1_000_000), 1),
        }

    for ax, ylab, title in [
        (ax_h, 'Height (m)', 'Altitude vs Time'),
        (ax_v, 'Velocity (m/s)', 'Velocity vs Time'),
        (ax_a, 'Acceleration (m/s\u00b2)', 'Acceleration vs Time'),
        (ax_F, 'Thrust (N)', 'Thrust vs Time'),
    ]:
        ax.set_xlabel('Time (s)', color='#e0e0e0')
        ax.set_ylabel(ylab, color='#e0e0e0')
        ax.set_title(title, color='#ffffff', fontweight='bold')
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='#e0e0e0')
        for spine in ax.spines.values():
            spine.set_color('#533483')

    fig.suptitle('Bottle Rocket Simulation — Water Ratio Comparison',
                 fontsize=16, fontweight='bold', color='#ffffff')
    fig.savefig(os.path.join(output_dir, 'comparison_plots.png'),
                dpi=150, bbox_inches='tight')

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    heights = [summaries[l]['max_height_m'] for l in labels]
    bars = ax2.bar(labels, heights, color=COLORS, edgecolor='white', linewidth=1.2)

    for bar, h in zip(bars, heights):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{h:.1f} m', ha='center', va='bottom', color='white',
                 fontweight='bold', fontsize=11)

    ax2.set_ylabel('Max Altitude (m)', color='#e0e0e0')
    ax2.set_title('Maximum Altitude by Water Volume Ratio',
                  fontsize=14, fontweight='bold', color='#ffffff')
    ax2.tick_params(colors='#e0e0e0')
    for spine in ax2.spines.values():
        spine.set_color('#533483')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate with water volumes
    for i, label in enumerate(labels):
        vol_ml = int(summaries[label]['water_volume_ml'])
        ax2.text(i, 1, f'{vol_ml} mL', ha='center', va='bottom',
                 color=COLORS[i], fontsize=9, fontweight='bold')

    fig2.savefig(os.path.join(output_dir, 'max_height_bar.png'),
                 dpi=150, bbox_inches='tight')

    # Return summaries table as text
    print(f"{'Ratio':>6} | {'Water':>7} | {'Max Ht':>7} | {'Max V':>7} | {'Max A':>7} | {'Thrust':>7} | {'Burn':>6}")
    print("-" * 65)
    for label in labels:
        s = summaries[label]
        print(f"{label:>6} | {s['water_volume_ml']:>5.0f}mL | {s['max_height_m']:>5.1f}m | {s['max_velocity_ms']:>5.1f}m/s | {s['max_accel_gs']:>4.0f}g | {s['max_thrust_N']:>5.1f}N | {s['burn_time_s']:>.3f}s")

    return summaries


def downsample(arr, max_pts=600):
    n = len(arr)
    if n <= max_pts:
        return arr
    step = n // max_pts
    return arr[::step]


def export_data(results, output_dir):
    export = {}
    for label, data in results.items():
        export[label] = {
            't': downsample(data['t']).tolist(),
            'x': downsample(data['x']).tolist(),
            'v': downsample(data['v']).tolist(),
            'a': downsample(data['a']).tolist(),
            'F': downsample(data['F']).tolist(),
            'P': downsample(data['P']).tolist(),
        }
    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(export, f)

    js = 'const ROCKET_DATA = ' + json.dumps(export) + ';'
    with open(os.path.join(output_dir, 'data.js'), 'w') as f:
        f.write(js)


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    config = {
        'V_bottle': 0.0015,
        'P_initial': 6e5,
        'm_bottle': 0.05,
        'd_nozzle': 0.02,
        'C_discharge': 0.8,
        'dt': 5e-5,
    }

    print("Running bottle rocket simulations...")
    print(f"  Bottle: 1.5 L")
    print(f"  Initial pressure: {config['P_initial']/1e5:.1f} bar absolute ({config['P_initial']/1e5 - 1:.1f} bar gauge)")
    print(f"  Bottle mass: {config['m_bottle']*1000:.0f} g")
    print(f"  Nozzle diameter: {config['d_nozzle']*1000:.0f} mm")
    print()

    results = run_all_simulations(config)
    summaries = make_plots(results, output_dir)
    export_data(results, output_dir)

    print(f"\nPlots saved to {output_dir}/comparison_plots.png")
    print(f"Data saved to {output_dir}/data.json")
    print(f"JS data saved to {output_dir}/data.js")


if __name__ == '__main__':
    main()
