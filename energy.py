"""Energy measurement method (named in the proposal).

Method: run open-weight models locally and measure with CodeCarbon (version 3.3.2 or later; that is the version whose
source was checked). CodeCarbon picks its CPU method at start-up:
  - Windows: the Windows Energy Meter Interface (EMI), which exposes the CPU's RAPL energy counters  -> measured
  - Linux:   Intel RAPL under /sys/class/powercap (often root-only)                                   -> measured
  - otherwise it falls back to an ESTIMATE from CPU load and the processor's rated TDP, or a constant.
GPU energy is read through NVML (NVIDIA) or AMD SMI when present. RAM is ALWAYS a constant-power estimate.
Intel Power Gadget was discontinued in December 2023, so it is not a usable method on new machines.

`codecarbon_meter` records which CPU method CodeCarbon actually used (`cpu_method`) and whether it is a hardware
counter (`cpu_measured`). run_real.py prints both and warns when the CPU figure is an estimate. Report the method
per machine in the paper; do not call an estimate a measurement.

Each rung gets its OWN tracker in "machine" mode (so the Ollama server process is included), and idle power is measured
before and after the run (run_real.py --measure-idle), so net energy above idle is reported per rung and per scenario.
Models reached only through a vendor API are NOT given an energy figure; report their tokens and latency.
"""
from contextlib import contextmanager

MEASURED = {"windows_emi", "intel_rapl", "intel_power_gadget"}  # CPU._mode values that read a hardware counter


def _cpu_method(tracker):
    """Best-effort read of the CPU method CodeCarbon chose (private attributes, checked against codecarbon 3.3.2)."""
    for hw in getattr(tracker, "_hardware", []) or []:
        if type(hw).__name__ == "CPU":
            return getattr(hw, "_mode", None)
    return None


@contextmanager
def codecarbon_meter(project="region-selection", country="SGP"):
    """Usage: with codecarbon_meter() as m: ...run local model...; then m['kwh'] holds the energy."""
    holder = {"kwh": None, "cpu_method": None, "cpu_measured": None}
    try:
        from codecarbon import EmissionsTracker
    except ImportError:  # pip install "codecarbon>=3.3.2"
        yield holder
        return
    kw = dict(project_name=project, save_to_file=False, log_level="error", tracking_mode="machine")
    try:  # a fixed country code skips the online location lookup (which fails behind some certificates)
        tracker = EmissionsTracker(country_iso_code=country, **kw)
    except TypeError:
        tracker = EmissionsTracker(**kw)
    tracker.start()
    try:
        yield holder
    finally:
        tracker.stop()
        data = tracker.final_emissions_data
        holder["kwh"] = getattr(data, "energy_consumed", None)
        holder["cpu_kwh"] = getattr(data, "cpu_energy", None)
        holder["ram_kwh"] = getattr(data, "ram_energy", None)   # constant-power estimate
        holder["gpu_kwh"] = getattr(data, "gpu_energy", None)
        m = _cpu_method(tracker)
        holder["cpu_method"] = m or "unknown"
        holder["cpu_measured"] = m in MEASURED


def energy_wh(tok_in, tok_out, wh_per_1k_in, wh_per_1k_out):
    """Apply CALIBRATED coefficients (measured on local runs). There are no defaults on purpose."""
    return tok_in / 1000 * wh_per_1k_in + tok_out / 1000 * wh_per_1k_out
