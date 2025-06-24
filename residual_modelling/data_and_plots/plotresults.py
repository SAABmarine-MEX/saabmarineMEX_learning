import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from typing import Dict, List, Optional

def main():
    data_dir = "data_and_plots/dictfiles_plot_tet"
    axes = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]


    models = ["zero-shot","knn","mtgp","svgp"]
    #summarize_metrics_by_model("processed_data", axes, models)
    #summarize_model_improvements("processed_data", axes, models)
    #summarize_combined_metrics("processed_data", axes, models)
    for npz_path in glob.glob(os.path.join(data_dir, "*.npz")):
        bag_name = os.path.splitext(os.path.basename(npz_path))[0]
        data = load_dict(npz_path)
    
        plot_vel_time_series(
            data,
            out_dir="data_and_plots/comparison_plots_tet/vel",
            bag_name=bag_name,
            axis_labels=axes,
            #models=['zero-shot','knn','mtgp','svgp'],
            #title=bag_name
        )
    # 
    #     # plot_vel_time_series(
    #     #     data,
    #     #     out_dir="report_plots/vel/short",
    #     #     bag_name=bag_name + "short",
    #     #     start_bin=300,
    #     #     end_bin=480,
    #     #     axis_labels=axes,
    #     #     #models=['zero-shot','knn','mtgp','svgp'],
    #     #     #title=bag_name
    #     # )
        #Plot acceleration RMSE
        plot_acc_rmse(
            data,
            out_dir="data_and_plots/comparison_plots_tet/comp/rmse",
            bag_name=bag_name,
            axis_labels=axes
            #title="Acceleration RMSE per Axis"
        )

        #Plot velocity MAE
        plot_vel_mae(
            data,
            out_dir="data_and_plots/comparison_plots_tet/comp/mae",
            bag_name=bag_name,
            axis_labels=axes
            #title="Velocity MAE per Axis"
        )

    # Plot change in position error over time
    # plot_pos_error_delta(
    #     data,
    #     axis_labels=axes,
    #     title="Δ Position Error Over Time"
    # )
    # plot_vel_time_series(
    #     data,
    #     out_dir="report",
    #     bag_name="testbag1_6dof",
    #     axis_labels=axes,
    # )


def load_dict(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load the compressed NPZ file produced by process_and_save and return all arrays as a dict.
    Applies a permutation to the 'controls' array for reordering inputs.

    Args:
        npz_path: Path to the .npz file containing saved bag data.

    Returns:
        A dictionary mapping each saved key to its corresponding numpy array.

    Raises:
        FileNotFoundError: If the provided npz_path does not exist.
        ValueError: If the file at npz_path is not a valid NPZ archive.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    try:
        loaded = np.load(npz_path)
    except Exception as e:
        raise ValueError(f"Error loading NPZ file '{npz_path}': {e}")

    # Build a normal dict and close the NPZ file
    data_dict = {key: loaded[key] for key in loaded.files}
    loaded.close()

    # Reorder control inputs if present
    if 'controls' in data_dict:
        perm = [4, 5, 2, 1, 0, 3]
        actions = data_dict['controls']
        # Ensure shape is (N,6)
        if actions.ndim == 2 and actions.shape[1] == len(perm):
            data_dict['controls'] = actions[:, perm]*100
        else:
            raise ValueError(
                f"Unexpected controls shape {actions.shape}, cannot permute with {perm}"
            )

    return data_dict

def plot_acc_rmse(
    data: Dict[str, np.ndarray],
    out_dir: str,                
    bag_name: str,

    axis_labels: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot per-axis RMSE of acceleration (simulated vs real) for each model in a grouped bar chart.

    Args:
        data: Dictionary as returned by load_saved_bag_data, must contain 'real_acc' and 'sim_acc_<suffix>' for each model.
        axis_labels: List of labels for each axis. Defaults to ['axis 0', ..., 'axis N'].
        models: List of model names to include. Defaults to ['zero-shot', 'knn', 'mtgp', 'svgp'].
        title: Optional title for the plot. If None, a generic title is used.
    """
    os.makedirs(out_dir, exist_ok=True)
    if models is None:
        models = ['zero-shot', 'knn', 'mtgp', 'svgp']
    if axis_labels is None:
        axis_labels = [f'axis {i}' for i in range(data['real_acc'].shape[1])]

    # Compute RMSE per axis for each model
    rmse_vals = {}
    real = data['real_acc']
    for model in models:
        suffix = 'zero_shot' if model == 'zero-shot' else model
        key = f'sim_acc_{suffix}'
        if key not in data:
            raise KeyError(f"Missing simulated acceleration data for model '{model}' (expected key '{key}')")
        sim = data[key]
        rmse = np.sqrt(np.mean((sim - real)**2, axis=0))
        rmse_vals[model] = rmse

    # Plot
    n_axes = len(axis_labels)
    x = np.arange(n_axes)
    n_models = len(models)
    width = 0.8 / n_models

    fig = plt.figure()
    for i, model in enumerate(models):
        plt.bar(x + (i - n_models/2 + 0.5)*width, rmse_vals[model], width, label=model)

    plt.xticks(x, axis_labels)
    plt.ylabel('Acceleration RMSE')
    plt.xlabel('Axis')
    #plt.title(title or 'Acceleration RMSE per Axis by Model')
    plt.legend()

    out_path = os.path.join(out_dir, f"{bag_name}_RMSE_acc.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved RMSE {out_path}")

def plot_vel_mae(
    data: Dict[str, np.ndarray],
    out_dir: str,                
    bag_name: str,
    axis_labels: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot per-axis MAE of velocity (simulated vs real) for each model in a grouped bar chart.

    Args:
        data: Dictionary as returned by load_saved_bag_data, must contain 'real_vel' and 'sim_vel_<suffix>' for each model.
        axis_labels: List of labels for each axis. Defaults to ['axis 0', ..., 'axis N'].
        models: List of model names to include. Defaults to ['zero-shot', 'knn', 'mtgp', 'svgp'].
        title: Optional title for the plot. If None, a generic title is used.
    """
    os.makedirs(out_dir, exist_ok=True)
    if models is None:
        models = ['zero-shot', 'knn', 'mtgp', 'svgp']
    if axis_labels is None:
        axis_labels = [f'axis {i}' for i in range(data['real_vel'].shape[1])]

    # Compute MAE per axis for each model
    mae_vals = {}
    real = data['real_vel']
    for model in models:
        suffix = 'zero_shot' if model == 'zero-shot' else model
        key = f'sim_vel_{suffix}'
        if key not in data:
            raise KeyError(f"Missing simulated velocity data for model '{model}' (expected key '{key}')")
        sim = data[key]
        mae = np.mean(np.abs(sim - real), axis=0)
        mae_vals[model] = mae

    # Plot
    n_axes = len(axis_labels)
    x = np.arange(n_axes)
    n_models = len(models)
    width = 0.8 / n_models

    fig = plt.figure()
    for i, model in enumerate(models):
        plt.bar(x + (i - n_models/2 + 0.5)*width, mae_vals[model], width, label=model)

    plt.xticks(x, axis_labels)
    plt.ylabel('Velocity MAE')
    plt.xlabel('Axis')
    #plt.title(title or 'Velocity MAE per Axis by Model')
    plt.legend()
    out_path = os.path.join(out_dir, f"{bag_name}_MAE_vel.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved MAE {out_path}")

def plot_pos_error_delta(
    data: Dict[str, np.ndarray],
    axis_labels: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot per-axis change in absolute position error (sim vs real) over time,
    with control inputs plotted above each. Shows full x-axis ticks, gridlines,
    and a shared legend outside.
    """

    if models is None:
        models = ['zero-shot', 'knn', 'mtgp', 'svgp']
    if axis_labels is None:
        axis_labels = [f'axis {i}' for i in range(data['real_pos'].shape[1])]

    real     = data['real_pos']      # shape (M,6)
    controls = data['controls']      # shape (M,6)
    ts       = data['timestamps']    # shape (M+1,) or (M,)
    ts_rel   = ts - ts[0]            # relative seconds
    M        = real.shape[0]

    # Δ-error has length M-1; align both control & error to ts_rel[:-1]
    t = ts_rel[:-1]

    n_axes = len(axis_labels)
    n_cols = n_axes // 2

    fig = plt.figure(figsize=(5*n_cols, 8))
    outer = GridSpec(2, n_cols, figure=fig, hspace=0.5, wspace=0.3)

    # collect legend handles once
    legend_handles = []
    legend_labels  = []

    for idx, label in enumerate(axis_labels):
        row = 0 if idx < n_cols else 1
        col = idx if idx < n_cols else idx - n_cols

        inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[row, col],
            height_ratios=[1, 4],
            hspace=0.05

        )

        # --- control subplot ---
        ax_ctrl = fig.add_subplot(inner[0])
        ax_ctrl.plot(t[:-2], controls[:-1, idx], linewidth=1)
        ax_ctrl.set_ylabel('ctrl', fontsize=8)
        ax_ctrl.set_xticks([])                # hide its x-ticks
        ax_ctrl.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_ctrl.set_title(label, pad=8)       # title here

        # --- error-delta subplot ---
        ax_err = fig.add_subplot(inner[1], sharex=ax_ctrl)
        for model in models:
            suffix = 'zero_shot' if model == 'zero-shot' else model
            sim = data[f'sim_pos_{suffix}']
            abs_err = np.abs(sim[:, idx] - real[:, idx])
            delta   = abs_err[1:] - abs_err[:-1]
            line, = ax_err.plot(t, delta, label=model)
            # record for legend
            if model not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model)

        ax_err.set_xlabel('time (s)')
        ax_err.set_ylabel('Δ pos err')
        ax_err.grid(True, which='both', linestyle='--', alpha=0.5)
        # ensure ticks show:
        ax_err.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax_err.tick_params(axis='both', labelsize=8)

    # single legend to the right of all subplots
    fig.legend(
        legend_handles, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    # make room on the right for legend
    plt.tight_layout(rect=(0,0,0.95,1))
    plt.show()


def plot_vel_time_series(
    data: Dict[str, np.ndarray],
    out_dir: str,                
    bag_name: str,
    start_bin: int          = 0,    
    end_bin: Optional[int]  = None,
    axis_labels: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    title: Optional[str] = None
    ) -> None:
    """
    Plot per-axis velocity time series (real + models) with controls above.
    Legend floats above all subplots, text is larger.
    Uses whole-second ticks (every 5 seconds for rotational axes) after converting
    timestamps from nanoseconds to seconds to avoid excessively large tick ranges.
    """


    os.makedirs(out_dir, exist_ok=True)
    # Defaults
    if models is None:
        models = ['zero-shot', 'knn', 'mtgp', 'svgp']
    if axis_labels is None:
        axis_labels = [f'axis {i}' for i in range(data['real_vel'].shape[1])]

    real_vel = data['real_vel']       # (M,6)
    controls = data['controls']       # (M,6)
    ts_ns = data['timestamps']        # nanoseconds
    
    N = real_vel.shape[0]
    if end_bin is None or end_bin > N:
        end_bin = N

    real_vel = real_vel[start_bin:end_bin]
    controls = controls[start_bin:end_bin]
    sim_vel_window = {}
    for m in models:
        suf = 'zero_shot' if m == 'zero-shot' else m
        key = f'sim_vel_{suf}'
        sim_vel_window[m] = data[key][start_bin:end_bin]  # now same length as real_vel
    
    # Convert to seconds (float)
    ts_sec = (ts_ns - ts_ns[0]) * 1e-9
    # align both arrays to length M

    full_t = ts_sec[1:N+1]
    # now slice it the same way
    t = full_t[start_bin:end_bin]

    ctrl_data = controls[:N, :]          # shape (M,6)
    ctrl_min, ctrl_max = ctrl_data.min(), ctrl_data.max()
    pad = 0.1 * (ctrl_max - ctrl_min)  # 5% padding

    style_map = {
        'zero-shot': '--',
        'knn':       '-.',
        'mtgp':      ':',
        'svgp':      ':'
    }
    dash_map = {
        'zero-shot': (5, 2),   # 4‐px dash, 2‐px gap
        'knn':       (5, 1, 1, 1),   # shorter dashes & gaps
        'mtgp':      (3, 1),   # almost a solid dotted line
        'svgp':      (3, 1),   # long dash, short gap
    }

    n_axes = len(axis_labels)
    n_cols = n_axes // 2
    fig = plt.figure(figsize=(5*n_cols, 8))
    outer = fig.add_gridspec(2, n_cols, hspace=0.2, wspace=0.2)

    handles, labels = [], []

    for idx, label in enumerate(axis_labels):
        row = 0 if idx < n_cols else 1
        col = idx if idx < n_cols else idx - n_cols

        inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[row, col],
            height_ratios=[1, 4],
            hspace=0.15
        )

        # Control subplot
        axc = fig.add_subplot(inner[0])
        axc.plot(t, controls[:N, idx], color='k', linewidth=1.2)
        if col == 0:
            axc.set_ylabel('PWM [%]', fontsize=11)

        axc.tick_params(labelbottom=False)
        axc.set_xlim(t[0], t[-1])
        axc.set_ylim(ctrl_min - pad, ctrl_max + pad)
    # remove all axis padding on x
        axc.margins(x=0) 
        axc.grid(True, axis='both', linestyle='--', alpha=0.35)
        axc.set_title(label, pad=10, fontsize=12)

        # Velocity subplot
        axv = fig.add_subplot(inner[1], sharex=axc)
        # real
        rl, = axv.plot(t, real_vel[:, idx], color='k', label='Real', linewidth=2)
        if 'Real' not in labels:
            handles.append(rl)
            labels.append('Real')
        # models
        for m in models:
            suf = 'zero_shot' if m == 'zero-shot' else m
            sim = sim_vel_window[m]
            ln, = axv.plot(t, sim[:, idx], linestyle=style_map[m], dashes=dash_map[m],label=m, linewidth=2)
            if m not in labels:
                handles.append(ln); labels.append(m)
        if col == 0 and row == 0: 
            axv.set_ylabel('Velocity [m/s]' , fontsize=12)
        elif col == 0 and row == 1: 
            axv.set_ylabel('Velocity [rad/s]' , fontsize=12)
        axv.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axv.tick_params(labelsize=9)
        # axv.grid(True, linestyle='--', alpha=0.5)
        axv.minorticks_on()
        axv.grid(which='major', linestyle='--', alpha=0.35)
        axv.grid(which='minor', linestyle=':',  alpha=0.1)
        # axv.tick_params(axis='x', which='major', labelsize=10)
        # axv.tick_params(axis='y', which='major', labelsize=10)

        axv.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

        axv.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # X-axis ticks
        if row == 1:

            axv.set_xlabel('Time [s]', fontsize=11)
        else:

            # keep tick marks but hide the labels
            axv.tick_params(axis='x', which='major', labelbottom=False)
            # draw vertical grid lines at each tick
            axv.grid(True, axis='x', linestyle='--', alpha=0.5)


    # Legend above all subplots
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(labels),
        fontsize=12,
        frameon=False
    )

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    out_path = os.path.join(out_dir, f"{bag_name}_vel_timeseries.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved velocity timeseries to {out_path}")

def summarize_model_improvements(
    data_dir: str,
    axis_labels: List[str],
    models: List[str]
) -> None:
    """
    For each model, compute per-axis percentage improvement over the zero-shot baseline:
      Acc improvement (%) = (RMSE_zero-shot - RMSE_model) / RMSE_zero-shot * 100
      Vel improvement (%) = (MAE_zero-shot - MAE_model) / MAE_zero-shot * 100

    Separate summaries for 6-DOF and 3-DOF bags, printing four tables.
    """
    # Prepare storage
    six_acc_imp = {m: {ax: [] for ax in axis_labels} for m in models}
    six_vel_imp = {m: {ax: [] for ax in axis_labels} for m in models}
    thr_acc_imp = {m: {ax: [] for ax in axis_labels} for m in models}
    thr_vel_imp = {m: {ax: [] for ax in axis_labels} for m in models}

    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = os.path.basename(path).lower()
        is6 = "6dof" in name
        is3 = "3dof" in name
        if not (is6 or is3):
            continue

        data     = load_dict(path)
        real_acc = data["real_acc"]    # shape (N, n_axes)
        real_vel = data["real_vel"]
        # load simulated data for each model
        sim_accs = {m: data[f"sim_acc_{m.replace('-', '_')}"] for m in models}
        sim_vels = {m: data[f"sim_vel_{m.replace('-', '_')}"] for m in models}

        # compute raw errors per model
        rmse_acc = {
            m: np.sqrt(np.mean((sim_accs[m] - real_acc)**2, axis=0))
            for m in models
        }
        mae_vel = {
            m: np.mean(np.abs(sim_vels[m] - real_vel), axis=0)
            for m in models
        }

        # baseline errors
        b_acc = rmse_acc["zero-shot"]
        b_vel = mae_vel["zero-shot"]

        # compute and store improvements
        for i, ax in enumerate(axis_labels):
            for m in models:
                if m == "zero-shot" or b_acc[i] == 0:
                    imp_acc = 0.0
                else:
                    imp_acc = (b_acc[i] - rmse_acc[m][i]) / b_acc[i] * 100

                if m == "zero-shot" or b_vel[i] == 0:
                    imp_vel = 0.0
                else:
                    imp_vel = (b_vel[i] - mae_vel[m][i]) / b_vel[i] * 100

                if is6:
                    six_acc_imp[m][ax].append(imp_acc)
                    six_vel_imp[m][ax].append(imp_vel)
                else:
                    thr_acc_imp[m][ax].append(imp_acc)
                    thr_vel_imp[m][ax].append(imp_vel)

    # Helper to print a table of improvements
    def _print_latex_table(
        label: str,
        table: Dict[str, Dict[str, List[float]]],
        axis_labels: List[str],
        models: List[str]
    ) -> None:
        """
        Emit a LaTeX table (tabular environment) for `table`, which is
        dict[model][axis]→list of per‐bag % errors.  Adds a final “Mean”
        row that averages across axes.
        """
        ncols = 1 + len(models)
        colspec = "l" + "r" * len(models)
        print(r"\begin{table}[htb]")
        print(r"  \centering")
        print(r"  \caption{" + label.replace("%", r"\%") + r"}")
        print(r"  \begin{tabular}{" + colspec + r"}")
        print(r"    \toprule")
        # header
        hdr = " & ".join(["Axis"] + models) + r" \\"
        print("    " + hdr)
        print(r"    \midrule")
        # body
        for ax in axis_labels:
            vals = []
            for m in models:
                arr = table[m][ax]
                if arr:
                    vals.append(f"{np.mean(arr):.2f}\\%")
                else:
                    vals.append("--")
            row = " & ".join([ax] + vals) + r" \\"
            print("    " + row)
        # total row
        print(r"    \midrule")
        total_vals = []
        for m in models:
            axis_means = [np.mean(table[m][ax]) for ax in axis_labels if table[m][ax]]
            if axis_means:
                total_vals.append(f"{np.mean(axis_means):.2f}\\%")
            else:
                total_vals.append("--")
        total_row = " & ".join(["Mean"] + total_vals) + r" \\"
        print("    " + total_row)
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")
        print()

    # Print four tables
    _print_latex_table("6-DOF Acceleration Improvement (%)", six_acc_imp, axis_labels, models)
    _print_latex_table("6-DOF Velocity Improvement (%)",    six_vel_imp, axis_labels, models)
    _print_latex_table("3-DOF Acceleration Improvement (%)", thr_acc_imp, axis_labels, models)
    _print_latex_table("3-DOF Velocity Improvement (%)",     thr_vel_imp, axis_labels, models)

def summarize_metrics_by_model(
    data_dir: str,
    axis_labels: List[str],
    models: List[str]
) -> None:
    """
    For each model in `models`, scan `data_dir` for all .npz bags, split
    into 6dof vs 3dof (by filename), and compute for each axis:
      - Acc %RMSE = rmse(sim_acc, real_acc) / mean(|real_acc|) * 100
      - Vel %MAE  = mae (sim_vel, real_vel) / mean(|real_vel|) * 100

    Then print two tables (6dof and 3dof), with one row per axis and one
    column per model, showing mean % error across bags.
    """
    # Initialize storage
    six_acc: Dict[str, Dict[str, List[float]]] = {
        m: {ax: [] for ax in axis_labels} for m in models
    }
    six_vel: Dict[str, Dict[str, List[float]]] = {
        m: {ax: [] for ax in axis_labels} for m in models
    }
    thr_acc = {m: {ax: [] for ax in axis_labels} for m in models}
    thr_vel = {m: {ax: [] for ax in axis_labels} for m in models}

    # Loop over all bags
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = os.path.basename(path).lower()
        is6 = "6dof" in name
        is3 = "3dof" in name
        if not (is6 or is3):
            continue

        data     = load_dict(path)
        real_acc = data["real_acc"]    # (N, n_axes)
        real_vel = data["real_vel"]
        mean_true_acc = np.mean(np.abs(real_acc), axis=0) + 1e-12
        mean_true_vel = np.mean(np.abs(real_vel), axis=0) + 1e-12

        for m in models:
            key_acc = f"sim_acc_{m.replace('-', '_')}"
            key_vel = f"sim_vel_{m.replace('-', '_')}"
            sim_acc = data[key_acc]
            sim_vel = data[key_vel]

            # compute raw errors
            rmse_acc = np.sqrt(np.mean((sim_acc - real_acc)**2, axis=0))
            mae_vel  = np.mean(np.abs(sim_vel - real_vel), axis=0)

            # percentage errors
            pct_acc = rmse_acc / mean_true_acc * 100   # shape (n_axes,)
            pct_vel = mae_vel  / mean_true_vel * 100

            # store
            for i, ax in enumerate(axis_labels):
                if is6:
                    six_acc[m][ax].append(pct_acc[i])
                    six_vel[m][ax].append(pct_vel[i])
                else:
                    thr_acc[m][ax].append(pct_acc[i])
                    thr_vel[m][ax].append(pct_vel[i])

    # Helper to print a table
    def _print_latex_table(
        label: str,
        table: Dict[str, Dict[str, List[float]]],
        axis_labels: List[str],
        models: List[str]
    ) -> None:
        """
        Emit a LaTeX table (tabular environment) for `table`, which is
        dict[model][axis]→list of per‐bag % errors.  Adds a final “Mean”
        row that averages across axes.
        """
        ncols = 1 + len(models)
        colspec = "l" + "r" * len(models)
        print(r"\begin{table}[htb]")
        print(r"  \centering")
        print(r"  \caption{" + label.replace("%", r"\%") + r"}")
        print(r"  \begin{tabular}{" + colspec + r"}")
        print(r"    \toprule")
        # header
        hdr = " & ".join(["Axis"] + models) + r" \\"
        print("    " + hdr)
        print(r"    \midrule")
        # body
        for ax in axis_labels:
            vals = []
            for m in models:
                arr = table[m][ax]
                if arr:
                    vals.append(f"{np.mean(arr):.2f}\\%")
                else:
                    vals.append("--")
            row = " & ".join([ax] + vals) + r" \\"
            print("    " + row)
        # total row
        print(r"    \midrule")
        total_vals = []
        for m in models:
            axis_means = [np.mean(table[m][ax]) for ax in axis_labels if table[m][ax]]
            if axis_means:
                total_vals.append(f"{np.mean(axis_means):.2f}\\%")
            else:
                total_vals.append("--")
        total_row = " & ".join(["Mean"] + total_vals) + r" \\"
        print("    " + total_row)
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")
        print()

    _print_latex_table("6-DOF Acc \\%RMSE", six_acc, axis_labels, models)
    _print_latex_table("6-DOF Vel \\%MAE",  six_vel, axis_labels, models)
    _print_latex_table("3-DOF Acc \\%RMSE", thr_acc, axis_labels, models)
    _print_latex_table("3-DOF Vel \\%MAE",  thr_vel, axis_labels, models)

def summarize_combined_metrics(
    data_dir: str,
    axis_labels: List[str],
    models: List[str]
) -> None:
    """
    Scans data_dir for all .npz bags, splits into 6-DOF vs 3-DOF by filename,
    computes per-axis percentage error for zero-shot and per-axis percentage
    improvement for the other models, then prints four LaTeX tables with columns:

    Axis | Zero-Shot (raw %error) | KNN (%Δ) | MTGP (%Δ) | SVGP (%Δ)
    """
    # Prepare storage: dict[group][model][axis] -> list of per-bag percent errors/improvements
    six = {m: {ax: [] for ax in axis_labels} for m in models}
    thr = {m: {ax: [] for ax in axis_labels} for m in models}

    eps = 1e-12
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = os.path.basename(path).lower()
        is6 = "6dof" in name
        is3 = "3dof" in name
        if not (is6 or is3):
            continue

        data     = load_dict(path)
        real_acc = data["real_acc"]    # shape (N, n_axes)
        real_vel = data["real_vel"]

        # mean absolute true values (for denominator)
        mta = np.mean(np.abs(real_acc), axis=0) + eps
        mtv = np.mean(np.abs(real_vel), axis=0) + eps

        # compute percent‐error for each model and axis
        pct_err_acc = {}
        pct_err_vel = {}
        for m in models:
            sim_acc = data[f"sim_acc_{m.replace('-', '_')}"]
            sim_vel = data[f"sim_vel_{m.replace('-', '_')}"]

            rmse_acc = np.sqrt(np.mean((sim_acc - real_acc)**2, axis=0))
            mae_vel  = np.mean(np.abs(sim_vel - real_vel), axis=0)

            pct_err_acc[m] = rmse_acc / mta * 100
            pct_err_vel[m] = mae_vel  / mtv * 100

        # baseline arrays
        b_acc = pct_err_acc["zero-shot"]
        b_vel = pct_err_vel["zero-shot"]

        # store raw zero-shot and improvements for KNN/MTGP/SVGP
        for i, ax in enumerate(axis_labels):
            for m in models:
                if m == "zero-shot":
                    val_acc = b_acc[i]
                    val_vel = b_vel[i]
                else:
                    val_acc = (b_acc[i] - pct_err_acc[m][i]) / b_acc[i] * 100
                    val_vel = (b_vel[i] - pct_err_vel[m][i]) / b_vel[i] * 100

                target = six if is6 else thr
                target[m][ax].append(val_acc if m != "zero-shot" else val_acc)
                # for velocity we’ll emit a separate table, but reuse same store
                # temporarily store velocity in the same dict,
                # we’ll handle separately below:
                target[m][ax+"_vel"] = target[m].get(ax+"_vel", []) + (
                    [val_vel] if isinstance(target[m].get(ax+"_vel", None), list) else []
                )

    # Helper to print one combined LaTeX table
    def _print_table(label: str,
                    store: Dict[str, Dict[str, List[float]]],
                    use_vel: bool = False):
        """
        label: caption text
        store: six or thr dict
        use_vel: if True, use the _vel entries in store; else use normal entries
        """
        suffix = "_vel" if use_vel else ""
        header_models = models[1:]  # knn, mtgp, svgp
        colspec = "l" + "r" * (1 + len(header_models))
        print(r"\begin{table}[htb]")
        print(r"  \centering")
        print(r"  \caption{" + label.replace("%", r"\%") + r"}")
        print(r"  \begin{tabular}{" + colspec + r"}")
        print(r"    \toprule")
        hdr = " & ".join(
        ["Axis", "Zero-Shot (%Error)"] +
        [f"{m.upper()} (%Δ)" for m in header_models]) + r" \\"
        print("    " + hdr)
        print(r"    \midrule")

        for ax in axis_labels:
            row = f"{ax:6s} & "
            # zero-shot raw %error
            zs_vals = store["zero-shot"][ax + suffix]
            zs = np.mean(zs_vals) if zs_vals else 0.0
            row = f"{ax:6s} & {zs:4.0f}\\%"
            # improvements
            for m in header_models:
                vals = store[m][ax + suffix]
                imp = np.mean(vals) if vals else 0.0
                row += f" & {imp:4.0f}\\%"
            row += r" \\"
            print("    " + row)

        print(r"    \midrule")
        # Mean row
        mean_zs = np.mean([np.mean(store["zero-shot"][ax+suffix]) for ax in axis_labels])
        mrow = f"{'Mean':6s} & {mean_zs:4.0f}\\%"
        for m in header_models:
            imp_means = [np.mean(store[m][ax+suffix]) for ax in axis_labels]
            m_imp = np.mean(imp_means)
            mrow += f" & {m_imp:4.0f}\\%"
        mrow += r" \\"
        print("    " + mrow)

        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")
        print()

    # Emit 4 tables: 6-DOF Acc, 6-DOF Vel, 3-DOF Acc, 3-DOF Vel
    _print_table("6-DOF Acc RMSE & Improvements", six, use_vel=False)
    _print_table("6-DOF Vel MAE & Improvements",  six, use_vel=True)
    _print_table("3-DOF Acc RMSE & Improvements", thr, use_vel=False)
    _print_table("3-DOF Vel MAE & Improvements",  thr, use_vel=True)

if __name__ == "__main__":
    main()
