from __future__ import annotations

from typing import Iterable, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from quaternion import Quaternion
from gyro_simulator import GyroSimulator
from gyro_integrator import GyroIntegrator
from data_io import DataIO
import copy


class VisualizerGraph:
    def __init__(self, figsize: Tuple[float, float] = (11, 7), visualize=True):
        self.figsize = figsize
        self.should_visualize = visualize

    @staticmethod
    def _seq_to_arrays(
        sequence: Iterable[Tuple[object, float]],
        unwrap: bool = True,
        degrees: bool = True,
        t0 = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a sequence of (Quaternion, t) to arrays:
            ts shape (N,), eulers shape (N,3) in rad or deg.
        """
        seq_list = list(sequence)
        if len(seq_list) == 0:
            return np.array([]), np.empty((0, 3))

        ts = np.array([float(t - t0) for (_, t) in seq_list], dtype=float)
        eulers = np.array([q.to_euler() for (q, _) in seq_list], dtype=float)

        if unwrap and eulers.size:
            eulers = np.column_stack([
                np.unwrap(eulers[:, 0]),
                np.unwrap(eulers[:, 1]),
                np.unwrap(eulers[:, 2]),
            ])

        if degrees:
            eulers = np.degrees(eulers)

        return ts, eulers

    @staticmethod
    def _seq_to_lists(sequence: Iterable[Tuple[Quaternion, float]], t0=0):
        """Return (quats_list, times_array) for convenience."""
        seq_list = list(sequence)
        qs = [q for (q, _) in seq_list]
        ts = np.array([float(t - t0) for (_, t) in seq_list], dtype=float)
        return qs, ts

    @staticmethod
    def _nearest_quat_at_time(qs_gt: List[Quaternion], ts_gt: np.ndarray, t: float) -> Quaternion:
        """Pick the ground-truth quaternion closest in time to t."""
        if ts_gt.size == 0:
            raise ValueError("Empty ground-truth sequence.")
        idx = np.searchsorted(ts_gt, t)
        if idx == 0:
            return qs_gt[0]
        if idx >= ts_gt.size:
            return qs_gt[-1]
        
        # choose nearer of idx-1 and idx
        if abs(ts_gt[idx] - t) > abs(t - ts_gt[idx - 1]):
            idx -= 1
        # if the difference is greater than 0.25s ignore it
        if abs(t - ts_gt[idx]) > 0.25:
            return None
        return qs_gt[idx]

    def visualize(
        self,
        *sequences: Iterable[Tuple[object, float]],
        labels: Optional[List[str]] = None,
        validity = None,
        unwrap: bool = True,
        degrees: bool = True,
        legend_loc: str = "best",
        grid: bool = True,
        labels_colors: Optional[List[str]] = None,
    ):
        """
        Plot roll/pitch/yaw vs time for any number of (Quaternion, t) sequences.
        """
        if not self.should_visualize:
            return
        if len(sequences) == 0:
            raise ValueError("visualize() needs at least one sequence.")

        if labels is None:
            labels = [f"seq {i+1}" for i in range(len(sequences))]
        else:
            if len(labels) != len(sequences):
                raise ValueError("labels must have same length as sequences.")
        
        if labels_colors is None:
            labels_colors = [None] * len(sequences)
        else:
            if len(labels_colors) != len(sequences):
                raise ValueError("labels_colors must have same length as sequences.")

        unit = "deg" if degrees else "rad"

        # Calculate the oldest timestamp
        t0 = np.min(np.array([seq[0][1] for seq in sequences]))

        fig, axs = plt.subplots(3, 1, figsize=self.figsize, sharex=False)
        roll_ax, pitch_ax, yaw_ax = axs

        ground_truth_flag = True
        for seq, label, color in zip(sequences, labels, labels_colors):
            ts, eulers = self._seq_to_arrays(seq, unwrap=unwrap, degrees=degrees, t0=t0)
            if eulers.shape[0] == 0:
                continue
            roll_ax.plot(ts, eulers[:, 0], label=label, color=color)
            pitch_ax.plot(ts, eulers[:, 1], label=label, color=color)
            yaw_ax.plot(ts, eulers[:, 2], label=label, color=color)
        
            if ground_truth_flag and validity is not None:
                ground_truth_flag = False
                # Convert validity to arrays aligned to t0
                v_ts = np.array([t - t0 for _, t in validity], dtype=float)
                v_mask = np.array([bool(v) for v, _ in validity])
                invalid_ts = v_ts[~v_mask]

                # If there are any invalid timestamps, interpolate and scatter red points
                if invalid_ts.size > 0:
                    # Interpolate safely within the time bounds
                    in_range = (invalid_ts >= ts.min()) & (invalid_ts <= ts.max())
                    invalid_ts_in = invalid_ts[in_range]

                    if invalid_ts_in.size > 0:
                        r_invalid = np.interp(invalid_ts_in, ts, eulers[:, 0])
                        p_invalid = np.interp(invalid_ts_in, ts, eulers[:, 1])
                        y_invalid = np.interp(invalid_ts_in, ts, eulers[:, 2])

                        roll_ax.scatter(invalid_ts_in, r_invalid, c="red", s=20, marker="o", label="Invalid measurement")
                        pitch_ax.scatter(invalid_ts_in, p_invalid, c="red", s=20, marker="o", label="Invalid measurement")
                        yaw_ax.scatter(invalid_ts_in, y_invalid, c="red", s=20, marker="o", label="Invalid measurement")

        roll_ax.set_title("Roll")
        pitch_ax.set_title("Pitch")
        yaw_ax.set_title("Yaw")

        roll_ax.set_ylabel(f"Angle [{unit}]")
        pitch_ax.set_ylabel(f"Angle [{unit}]")
        yaw_ax.set_ylabel(f"Angle [{unit}]")
        yaw_ax.set_xlabel("Time [s]")

        if grid:
            for ax in axs:
                ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels_ = axs[0].get_legend_handles_labels()
        fig.legend(
            handles, labels_,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=4,
            frameon=True
        )

        fig.subplots_adjust(top=0.86, hspace=0.45)
        plt.show()

    def visualize_error(
        self,
        *sequences: Iterable[Tuple[Quaternion, float]],
        labels: Optional[List[str]] = None,
        degrees: bool = True,
        grid: bool = True,
        legend_loc: str = "best",
        labels_colors: Optional[List[str]] = None,
    ):
        """
        Plot orientation error (geodesic angle) vs time for multiple sequences
        with the FIRST sequence treated as GROUND TRUTH.

        Args:
            *sequences: any number of [(Quaternion, t), ...]; sequences[0] is ground truth.
            labels: optional list of labels (same length as sequences). If None, auto-enumerate.
            degrees: plot in degrees (recommended).
            grid: show grid on the plot.
            legend_loc: legend location.
        """
        if not self.should_visualize:
            return
        if len(sequences) < 2:
            raise ValueError("visualize_error() needs at least two sequences (ground truth + at least one estimate).")

        if labels is None:
            labels = [f"seq {i+1}" for i in range(len(sequences))]
        else:
            if len(labels) != len(sequences):
                raise ValueError("labels must have same length as sequences.")
        
        if labels_colors is None:
            labels_colors = [None] * len(sequences)
        else:
            if len(labels_colors) != len(sequences):
                raise ValueError("labels_colors must have same length as sequences.")
        
        # Calculate the oldest timestamp
        t0 = np.min(np.array([seq[0][1] for seq in sequences]))

        # Ground truth
        qs_gt, ts_gt = self._seq_to_lists(sequences[0], t0=t0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # For each estimate sequence, compute error angle vs nearest GT sample
        for seq, lbl, color in zip(sequences[1:], labels[1:], labels_colors[1:]):
            qs_est, ts_est = self._seq_to_lists(seq, t0=t0)
            if len(qs_est) == 0:
                continue

            errs = []
            for q_est, t in zip(qs_est, ts_est):
                q_true = self._nearest_quat_at_time(qs_gt, ts_gt, t)
                if q_true == None:
                    errs.append(errs[-1])
                    continue
                ang_deg = Quaternion.error_angle(q_est, q_true, True) if degrees else (
                    Quaternion.error_angle(q_est, q_true, False)
                )
                errs.append(ang_deg)

            errs = np.asarray(errs, dtype=float)
            ax.plot(ts_est, errs, label=lbl, color=color)

        unit = "deg" if degrees else "rad"
        ax.set_title("Orientation error vs. Time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Angle error [{unit}]")
        if grid:
            ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc=legend_loc)

        fig.tight_layout()
        plt.show()
    

    def visualize_validity_points(
        self,
        sequence,
        validity,
        unwrap: bool = True,
        degrees: bool = True,
    ):
        if not self.should_visualize:
            return
        if sequence is None or len(sequence) == 0:
            raise ValueError("visualize_validity_points() needs a non-empty sequence.")
        if validity is None or len(validity) == 0:
            raise ValueError("visuality_validity_points() needs a non-empty validity sequence.")
        unit = "deg" if degrees else "rad"

        # Oldest timestamp across both inputs for a common t0
        t0_seq = sequence[0][1]
        t0_val = validity[0][1]
        t0 = min(t0_seq, t0_val)

        # Convert quaternion sequence to arrays (uses your existing helper)
        ts, eulers = self._seq_to_arrays(sequence, unwrap=unwrap, degrees=degrees, t0=t0)
        if eulers.shape[0] == 0:
            raise ValueError("No Euler data available to plot.")

        # Convert validity to arrays aligned to t0
        v_ts = np.array([t - t0 for _, t in validity], dtype=float)
        v_mask = np.array([bool(v) for v, _ in validity])
        invalid_ts = v_ts[~v_mask]

        # Figure and axes like the other visualize()
        fig, axs = plt.subplots(3, 1, figsize=self.figsize, sharex=False)
        roll_ax, pitch_ax, yaw_ax = axs

        # Plot the continuous sequences
        roll_ax.plot(ts, eulers[:, 0], label="sequence")
        pitch_ax.plot(ts, eulers[:, 1], label="sequence")
        yaw_ax.plot(ts, eulers[:, 2], label="sequence")

        # If there are any invalid timestamps, interpolate and scatter red points
        if invalid_ts.size > 0:
            # Interpolate safely within the time bounds
            in_range = (invalid_ts >= ts.min()) & (invalid_ts <= ts.max())
            invalid_ts_in = invalid_ts[in_range]

            if invalid_ts_in.size > 0:
                r_invalid = np.interp(invalid_ts_in, ts, eulers[:, 0])
                p_invalid = np.interp(invalid_ts_in, ts, eulers[:, 1])
                y_invalid = np.interp(invalid_ts_in, ts, eulers[:, 2])

                roll_ax.scatter(invalid_ts_in, r_invalid, c="red", s=20, marker="o", label="invalid")
                pitch_ax.scatter(invalid_ts_in, p_invalid, c="red", s=20, marker="o", label="invalid")
                yaw_ax.scatter(invalid_ts_in, y_invalid, c="red", s=20, marker="o", label="invalid")

        # Titles / labels
        roll_ax.set_title("Roll")
        pitch_ax.set_title("Pitch")
        yaw_ax.set_title("Yaw")

        roll_ax.set_ylabel(f"Angle [{unit}]")
        pitch_ax.set_ylabel(f"Angle [{unit}]")
        yaw_ax.set_ylabel(f"Angle [{unit}]")
        yaw_ax.set_xlabel("Time [s]")

        # Grid like the other function
        for ax in axs:
            ax.grid(True, linestyle="--", alpha=0.5)

        # Build a combined legend (sequence + invalid if present)
        handles0, labels0 = axs[0].get_legend_handles_labels()
        handles1, labels1 = axs[1].get_legend_handles_labels()
        handles2, labels2 = axs[2].get_legend_handles_labels()
        handles = handles0 + [h for h, l in zip(handles1, labels1) if l == "invalid"] + \
                [h for h, l in zip(handles2, labels2) if l == "invalid"]
        labels = ["sequence"] + (["invalid"] if any(l == "invalid" for l in labels1 + labels2) else [])
        if handles:
            fig.legend(handles, labels, loc="upper right")

        fig.tight_layout()
        plt.show()
    
    def calculate_angular_rate(self, omega_rad_s, half_angle_deg):
        return np.rad2deg(2*omega_rad_s*np.sin(np.deg2rad(half_angle_deg/2)))

    def visualize_error_slope_heatmap(
        self,
        omega_range: Iterable[float],
        half_angle_range: Iterable[float],
        measured_fn,
        method_label: str,
        *,
        duration: float = 10.0,
        dt: float = 0.01,
        noise_std: float = 0.1,
        fast_loop_samples = 20,
        degrees: bool = True,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        grid: bool = False,
        log: bool = True,
    ):
        """
        Draw a heat map where X=omega, Y=half_angle_deg, and value is the slope of
        orientation error vs time (via linear regression).

        measured_fn must be callable as: measured_fn(omega, half_angle_deg, duration, noise_std)
        and return a sequence of (Quaternion, t) like your get_measured_* helpers.

        Returns (fig, ax, im, omegas, half_angles, Z) where Z[i,j] is slope at
        half_angles[i], omegas[j].
        """
        # Materialize axes
        omegas = np.array(list(omega_range), dtype=float)
        half_angles = np.array(list(half_angle_range), dtype=float)
        if omegas.size == 0 or half_angles.size == 0:
            raise ValueError("omega_range and half_angle_range must be non-empty.")

        # Allocate result grid: rows = half_angles, cols = omegas
        Z = np.full((half_angles.size, omegas.size), np.nan, dtype=float)

        slopes = []
        # Loop over grid, compute error slope
        for i, ha in enumerate(half_angles):
            for j, om in enumerate(omegas):
                total_slopes = 0
                counter = 0
                # Ground truth at this (omega, half-angle)
                gt_seq = get_ground_truth_orientations(om, ha, duration, dt)
                if len(gt_seq) == 0:
                    continue
                qs_gt, ts_gt = self._seq_to_lists(gt_seq)
                for _ in range(1):
                    # Measured sequence produced by the chosen integrator
                    est_seq = measured_fn(om, ha, duration, noise_std, fast_loop_samples)
                    if len(est_seq) == 0:
                        continue
                    qs_est, ts_est = self._seq_to_lists(est_seq)

                    # Compute geodesic error at each est timestamp vs nearest GT
                    errs = []
                    for q_est, t in zip(qs_est, ts_est):
                        q_true = self._nearest_quat_at_time(qs_gt, ts_gt, t)
                        if q_true == None:
                            errs.append(errs[-1])
                            continue
                        ang = Quaternion.error_angle(q_est, q_true, degrees)
                        errs.append(ang)
                    if len(errs) < 2:
                        continue

                    errs = np.asarray(errs, dtype=float)
                    ts_est = np.asarray(ts_est, dtype=float)

                    # Linear regression slope (deg/s if degrees=True, else rad/s)
                    # Use np.polyfit with degree=1: returns [slope, intercept]
                    try:
                        slope = np.polyfit(ts_est, errs, 1)[0]
                        total_slopes += slope
                        counter += 1
                    except np.linalg.LinAlgError:
                        slope = 0

                slope = total_slopes / counter
                if self.calculate_angular_rate(om, ha) <= 6:
                    slopes.append(slope)
                Z[i, j] = slope

        # Calculate and print the average slope error
        print(f"Average slope for {method_label}: {sum(slopes) / len(slopes)} deg/s")

        eps = 1e-300
        Z_pos = np.where(np.isfinite(Z) & (Z > 0), Z, 0)
        min_pos = np.min(Z[Z > 0])

        # Build the log image
        if log:
            logZ = np.log10(np.clip(Z_pos, min_pos, np.nanmax(Z_pos) if np.isfinite(np.nanmax(Z_pos)) else 1.0))
        else:
            logZ = Z_pos
        
        # # mask of valid positive finite values
        # mask = np.isfinite(Z) & (Z > 0)

        # min_pos = np.min(Z[mask])
        # max_pos = np.max(Z[mask])

        # Z_clean = Z.copy()

        # # +inf  -> highest legal value
        # Z_clean[np.isposinf(Z)] = max_pos

        # # NaN or -inf or <=0 -> lowest legal value
        # Z_clean[~mask] = min_pos

        # if log:
        #     logZ = np.log10(Z_clean)
        # else:
        #     logZ = Z_clean

        # Auto limits if none are given (on the LOG scale)
        if vmin is None or vmax is None:
            finite_logZ = logZ[np.isfinite(logZ)]
            if finite_logZ.size:
                auto_min, auto_max = np.nanmin(finite_logZ), np.nanmax(finite_logZ)
            else:
                auto_min, auto_max = -6.0, -5.0  # harmless fallback
            if vmin is None: vmin = auto_min
            if vmax is None: vmax = auto_max
            if vmin >= vmax:
                vmin, vmax = auto_min, auto_max

        # Plot (log scale image, limits also on log scale)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = ax.imshow(
            logZ,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,   # LOG10 limits
            vmax=vmax,
            extent=[omegas.min(), omegas.max(), half_angles.min(), half_angles.max()],
        )
        ax.set_xlabel("ω [rad/s]")
        ax.set_ylabel("Half-angle α [deg]")
        unit = "deg/s" if degrees else "rad/s"
        if log:
            ax.set_title(f"Log error slope heat map — {method_label}")
        else:
            ax.set_title(f"Error slope heat map — {method_label}")

        cbar = fig.colorbar(im, ax=ax)
        # Nice base-10 ticks on colorbar
        ticks_log = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks_log)
        if log:
            cbar.set_ticklabels([f"{t:.3f}" for t in ticks_log])
            # cbar.set_ticklabels([f"1e{int(t)}" if abs(t-round(t))<1e-6 else f"1e{t:.1f}" for t in ticks_log])
            cbar.set_label(f"log10(error slope) [{unit}]")
        else:
            cbar.set_ticklabels([f"{t:.3f}" for t in ticks_log])
            cbar.set_label(f"Error slope [{unit}]")

        if grid:
            ax.grid(False)

        fig.tight_layout()
        plt.show()
    

    def visualize_error_sequences(
        self,
        *error_sequences: Iterable[Tuple[float, float]],
        labels: Optional[List[str]] = None,
        average_errors,
        ylabel: str = "Angle error [deg]",
        title: str = "Accumulated error vs time",
        legend_loc: str = "best",
        grid: bool = True,
    ):
        """
        Plot multiple error sequences ([(err, t), ...]) on the same axes.
        Each sequence is time-shifted so its first timestamp is t=0.

        Returns (fig, ax).
        """
        if not self.should_visualize:
            return
        if len(error_sequences) == 0:
            raise ValueError("Provide at least one error sequence.")

        if labels is None:
            labels = [f"seq {i+1}" for i in range(len(error_sequences))]
        elif len(labels) != len(error_sequences):
            raise ValueError("labels must match the number of sequences.")
        
        cmap = plt.get_cmap("tab10")
        colors = [cmap((i + 1) % 10) for i in range(len(error_sequences))]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        for seq, lbl, avg_errors, color in zip(error_sequences, labels, average_errors, colors):
            seq = list(seq)
            if not seq:
                continue
            ts = np.array([float(t) for (_, t) in seq], dtype=float)
            errs = np.array([float(e) for (e, _) in seq], dtype=float)
            # ax.scatter(ts, errs, label=lbl + f", avg error rate: {round(avg_errors[0], 6)} deg/s")
            ax.plot(ts, errs, label=lbl + f", avg error rate: {round(avg_errors[0], 6)} deg/s", color=color)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc=legend_loc)
        fig.tight_layout()
        plt.show()
        return fig, ax


    def visualize_coning_rotation_heatmap(self, omega_range, half_angle_range):
        """
        Heatmap where X = omega [rad/s], Y = half-angle [deg],
        and value = max_t ||omega_body(t)|| calculated from GyroSimulator.coning_angular_velocities().
        """
        # Materialize axes
        omegas = np.array(list(omega_range), dtype=float)
        half_angles = np.array(list(half_angle_range), dtype=float)
        if omegas.size == 0 or half_angles.size == 0:
            raise ValueError("omega_range and half_angle_range must be non-empty.")

        # Allocate grid: rows = half_angles, cols = omegas
        Z = np.full((half_angles.size, omegas.size), np.nan, dtype=float)

        # We'll sample with a decent output rate so the per-interval averages are smooth
        output_hz = 200

        for i, ha in enumerate(half_angles):
            for j, om in enumerate(omegas):
                if om <= 0.0:
                    Z[i, j] = 0.0
                    continue

                # Cover ~2 coning cycles for this omega
                period = 2.0 * np.pi / om
                duration = 2.0 * period  # two periods is plenty for max ||w||
                gyro = GyroSimulator(output_hz)

                w_seq = gyro.coning_angular_velocities(om, ha, duration, noise=False)
                if not w_seq:
                    Z[i, j] = 0.0
                    continue

                # Take norm of the averaged body rates, then the max over time
                ws = np.array([np.linalg.norm(w) for (w, _t) in w_seq], dtype=float)
                Z[i, j] = float(np.rad2deg(np.nanmax(ws))) if ws.size else 0.0

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[omegas.min(), omegas.max(), half_angles.min(), half_angles.max()],
        )
        ax.set_xlabel("ω [rad/s]")
        ax.set_ylabel("Half-angle [deg]")
        ax.set_title("Max body-rate magnitude during coning: maxₜ ‖ω_body(t)‖")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("max‖ω_body‖ [deg/s]")

        fig.tight_layout()
        plt.show()

        return fig, ax, im, omegas, half_angles, Z


    def plot_average_errors(self, labels, *average_errors, colors=None):
        """
        Create a bar chart of average errors.
        labels: list of strings
        average_errors: iterable of numeric values
        """

        errs = np.array(average_errors, dtype=float)
        x = np.arange(len(labels))

        if colors is None:
            cmap = plt.get_cmap("tab10")
            colors = [cmap((i + 1) % 10) for i in range(len(errs))]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(x, errs, color=colors)

        # Add values on top of bars
        for bar, val in zip(bars, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.8f}",
                ha="center",
                va="bottom"
            )

        # ax.axvline(1.5, linestyle="--", color="gray", alpha=0.6)
        # ax.axvline(3.5, linestyle="--", color="gray", alpha=0.6)
        # ax.axvline(5.5, linestyle="--", color="gray", alpha=0.6)
        # ax.text(0.5, ax.get_ylim()[1]*0.95, "4 Hz", ha="center")
        # ax.text(2.5, ax.get_ylim()[1]*0.95, "1 Hz", ha="center")
        # ax.text(4.5, ax.get_ylim()[1]*0.95, "0.5 Hz", ha="center")
        # ax.text(6.1, ax.get_ylim()[1]*0.95, "0.2 Hz", ha="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Average error rate [deg/s]")
        ax.set_title("Average Error Comparison")

        fig.tight_layout()
        plt.show()
        return fig, ax




def get_ground_truth_orientations(omega, half_angle_deg, duration, dt):
    gyro = GyroSimulator(80)
    orientations_ground_truth = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations_ground_truth

def get_measured_orientations(omega, half_angle_deg, duration, noise_std, fast_loop_samples=20):
    gyro = GyroSimulator(80, noise_std=noise_std)
    s = np.sin(0.5*np.deg2rad(half_angle_deg))
    c = np.cos(0.5*np.deg2rad(half_angle_deg))
    init_quat = Quaternion(s, 0, 0, c)
    gyro_integrator = GyroIntegrator(initial_orientation=init_quat, fast_loop_samples=fast_loop_samples)

    angular_velocities = gyro.coning_angular_velocities(omega, half_angle_deg, duration, noise=True)
    for w, t in angular_velocities:
        gyro_integrator.update_direct_quaternion(w, t)
    return gyro_integrator.orientations

def get_measured_orientations_fast_loop(omega, half_angle_deg, duration, noise_std, fast_loop_samples=20):
    gyro = GyroSimulator(80, noise_std=noise_std)
    s = np.sin(0.5*np.deg2rad(half_angle_deg))
    c = np.cos(0.5*np.deg2rad(half_angle_deg))
    init_quat = Quaternion(s, 0, 0, c)
    gyro_integrator = GyroIntegrator(initial_orientation=init_quat, fast_loop_samples=fast_loop_samples)

    angular_velocities = gyro.coning_angular_velocities(omega, half_angle_deg, duration, noise=True)
    for w, t in angular_velocities:
        gyro_integrator.update_rotation_vector(w, t)
    return gyro_integrator.orientations

def get_measured_orientations_fast_loop_noconing(omega, half_angle_deg, duration, noise_std, fast_loop_samples=20):
    gyro = GyroSimulator(80, noise_std=noise_std)
    s = np.sin(0.5*np.deg2rad(half_angle_deg))
    c = np.cos(0.5*np.deg2rad(half_angle_deg))
    init_quat = Quaternion(s, 0, 0, c)
    gyro_integrator = GyroIntegrator(initial_orientation=init_quat, fast_loop_samples=fast_loop_samples)

    angular_velocities = gyro.coning_angular_velocities(omega, half_angle_deg, duration, noise=True)
    for i, (w, t) in enumerate(angular_velocities):
        if i == len(angular_velocities) - 1:
            force = True
        else:
            force = False
        gyro_integrator.update_rotation_vector(w, t, coning=False, force_orientation=force)
    return gyro_integrator.orientations

def get_measured_fast_gyro():
    gyro_integrator = GyroIntegrator()
    dataIO = DataIO()

    angular_velocities = dataIO.read_fast_gyro(2)
    for w, t in angular_velocities:
        gyro_integrator.update_direct_quaternion(w, t)
    return gyro_integrator.orientations

def get_error_slope_heatmap_no_noise_params():
    # omega_range = np.linspace(0.1, 1, 20)        # rad/s
    omega_range = np.linspace(1, 10, 20)        # rad/s
    half_angle_range = np.linspace(1.0, 20.0, 20)   # deg
    duration = 10
    vmin = -10
    vmax = 1.15
    # vmin = 0
    # vmax = 0.034
    fast_loop_samples = 20
    noise_std = 0
    log = True
    return omega_range, half_angle_range, duration, vmin, vmax, fast_loop_samples, noise_std, log

def get_error_slope_heatmap_noise_params():
    omega_range = np.linspace(0.1, 1, 20)        # rad/s
    half_angle_range = np.linspace(1.0, 20.0, 20)   # deg
    duration = 10
    vmin = 0
    vmax = 0.034
    fast_loop_samples = 20
    noise_std = np.deg2rad(0.0209)
    log = False
    return omega_range, half_angle_range, duration, vmin, vmax, fast_loop_samples, noise_std, log

def get_error_slope_heatmap_update_frequencies():
    omega_range = np.linspace(0.1, 1, 20)        # rad/s
    half_angle_range = np.linspace(1.0, 20.0, 20)   # deg
    duration = 10
    # vmin = -7
    vmin = -8
    # vmax = 0.4
    vmax = 0.6
    # 20, 81, 161, 400
    fast_loop_samples = 400
    # noise_std = np.deg2rad(0.0209)
    noise_std = 0
    log = True
    return omega_range, half_angle_range, duration, vmin, vmax, fast_loop_samples, noise_std, log


if __name__ == "__main__":
    omega = 2.0
    half_angle_deg = 10.0
    duration = 10
    dt = 0.0125
    noise_std = 0

    orientations_ground_truth = get_ground_truth_orientations(omega, half_angle_deg, duration, dt)
    orientations_measured = get_measured_orientations(omega, half_angle_deg, duration, noise_std)
    orientations_measured_2 = get_measured_orientations_fast_loop(omega, half_angle_deg, duration, noise_std)
    orientations_measured_noconing = get_measured_orientations_fast_loop_noconing(omega, half_angle_deg, duration, noise_std)

    visualizer_graph = VisualizerGraph()
    labels = ["Ground Truth", "Direct Quat. Update", "Rotation Vector Update", "Simple Kinematic Int."]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # # Euler angle plots
    # visualizer_graph.visualize(
    #     orientations_ground_truth, orientations_measured, orientations_measured_2, orientations_measured_noconing,
    #     labels=labels,
    #     labels_colors=colors
    # )

    # # Error plots (each non-GT vs GT)
    # visualizer_graph.visualize_error(
    #     orientations_ground_truth, orientations_measured[:-1], orientations_measured_2[:-1], orientations_measured_noconing[:-1],
    #     labels=labels,
    #     labels_colors=colors
    # )

    # fast_gyro_orientations = get_measured_fast_gyro()
    # labels = ["Fast gyro"]
    # # Euler angle plots
    # visualizer_graph.visualize(
    #     fast_gyro_orientations,
    #     labels=labels
    # )

    # Ranges to sweep
    omega_range = np.linspace(0.1, 1, 20)        # rad/s
    half_angle_range = np.linspace(1.0, 20.0, 20)   # deg

    # omega_range = np.linspace(1, 10, 20)        # rad/s
    # half_angle_range = np.linspace(1.0, 20.0, 20)   # deg

    duration = 10
    vmin = 0
    vmax = 0.034
    # vmin = -6
    # vmax = 0.41

    # vmin = -10
    # vmax = 1.15
    # vmax = 2.57
    fast_loop_samples = 20
    noise_std = np.deg2rad(0.0209) # MIMU noise @80Hz
    # noise_std = np.deg2rad(0.1347) # MIMU noise @3333Hz
    # noise_std = np.deg2rad(0.2887) # ASM IMU noise 0.005 °/s/sqrt(Hz) * sqrt(3333Hz) = 0.2887 °/s
    # noise_std = 0
    cmap = "viridis"
    # cmap = "inferno"
    dt = 0.01

    omega_range, half_angle_range, duration, vmin, vmax, fast_loop_samples, noise_std, log = get_error_slope_heatmap_no_noise_params()

    vg = VisualizerGraph()

    # vg.visualize_coning_rotation_heatmap(omega_range, half_angle_range)

    # Heatmap for direct quaternion update
    vg.visualize_error_slope_heatmap(
        omega_range,
        half_angle_range,
        measured_fn=get_measured_orientations,
        method_label="Direct Quat. Update",
        duration=duration,
        dt=0.01,
        noise_std=noise_std,
        fast_loop_samples=fast_loop_samples,
        degrees=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log=log,
    )

    # Heatmap for rotation-vector fast loop (same color scale)
    vg.visualize_error_slope_heatmap(
        omega_range,
        half_angle_range,
        measured_fn=get_measured_orientations_fast_loop,
        method_label="Rotation Vector Update",
        duration=duration,
        dt=dt,
        noise_std=noise_std,
        fast_loop_samples=fast_loop_samples,
        degrees=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log=log,
    )

    # Heatmap for rotation-vector fast loop (same color scale)
    vg.visualize_error_slope_heatmap(
        omega_range,
        half_angle_range,
        measured_fn=get_measured_orientations_fast_loop_noconing,
        method_label="Simple Kinematic Int.",
        duration=duration,
        dt=dt,
        noise_std=noise_std,
        fast_loop_samples=fast_loop_samples,
        degrees=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log=log,
    )

    # @4 Hz
    error_slopes_no_noise_4 = [3.1e-11, 1.6e-8, 4.4e-4]
    error_slopes_noise_4 = [6.4e-4, 9.8e-4, 1.2e-3]

    # @1 Hz
    error_slopes_no_noise_1 = [4e-5, 6.8e-3]
    error_slopes_noise_1 = [9.4e-4, 7.3e-3]

    # @0.5 Hz
    error_slopes_no_noise_05 = [4.2e-5, 2.6e-2]
    error_slopes_noise_05 = [1.1e-3, 2.67e-2]

    # @0.2 Hz
    error_slopes_no_noise_02 = [2.1e-3, 1.1e-1]
    error_slopes_noise_02 = [2.7e-3, 1.1e-1]

    # # combined graph of noise and no noise
    # labels = ["Direct Quat. Update", "Rotation Vector Update", "Simple Kinematic Int.", "Direct Quat. Update with Noise", "Rotation Vector Update with Noise", "Simple Kinematic Int. with Noise"]
    # colors = ["tab:orange", "tab:green", "tab:red", "tab:orange", "tab:green", "tab:red"]
    # vg.plot_average_errors(labels, *error_slopes_no_noise_4, *error_slopes_noise_4, colors=colors)

    # different update frequencies no noise
    # labels = ["Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int."]
    # colors = ["tab:green", "tab:red", "tab:green", "tab:red", "tab:green", "tab:red", "tab:green", "tab:red"]
    # vg.plot_average_errors(labels, *error_slopes_no_noise_4[1:], *error_slopes_no_noise_1, *error_slopes_no_noise_05, *error_slopes_no_noise_02, colors=colors)

    # # different update frequencies noise
    # labels = ["Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int.", "Rotation Vector Update", "Simple Kinematic Int."]
    # colors = ["tab:green", "tab:red", "tab:green", "tab:red", "tab:green", "tab:red", "tab:green", "tab:red"]
    # vg.plot_average_errors(labels, *error_slopes_noise_4[1:], *error_slopes_noise_1, *error_slopes_noise_05, *error_slopes_noise_02, colors=colors)