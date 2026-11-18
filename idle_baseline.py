"""Measure the idle machine for 60 seconds with CodeCarbon. Close other programs and do not touch the PC while it runs.
Compare the CPU figure with a run that has the model working: if they are similar, the CPU reading is not tracking load."""
import time
from energy import codecarbon_meter

SECONDS = 60
with codecarbon_meter("idle-baseline") as m:
    time.sleep(SECONDS)

print(f"\nIdle for {SECONDS} s")
if m.get("cpu_kwh") is not None:
    print(f"  CPU {m['cpu_kwh']*1000:.3f} Wh = {m['cpu_kwh']*1000*3600/SECONDS:.1f} W average")
    print(f"  RAM {(m.get('ram_kwh') or 0)*1000:.3f} Wh = {(m.get('ram_kwh') or 0)*1000*3600/SECONDS:.1f} W average (estimated)")
    print(f"  CPU method used by CodeCarbon: {m['cpu_method']} "
          f"({'hardware counter' if m['cpu_measured'] else 'ESTIMATE from load/TDP - not a measurement'})")
else:
    print("  No breakdown available. Is codecarbon installed?")
