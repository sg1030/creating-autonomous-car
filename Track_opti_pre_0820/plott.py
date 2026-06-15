import numpy as np
import matplotlib.pyplot as plt

lap_times_dynamics_nominal=[11.9,11.9,11.9,11.9,11.9,11.9,11.9,11.9,11.9,11.9]
lap_times_gp = [11.9,11.2,11.2,11.6,11.2,11.2,12.3,11.2,11.3,11.2]
lap_times_modelbank_updated_bruteforce = [11.9,11.3,11.3,11.1,11.1,11.1,11.1,11.1,11.1,11.1]
lap_times_modelbank_static_bruteforce = [11.9,11.3,11.3,11.3,11.3,11.3,11.3,11.3,11.3,11.3]
lap_times_proposed=[11.9,11.2,11.1,11.1,11.6,11.1,11.1,11.1,11.2,11.1]
lap_times_dynamics_true=[11.1,11.1,11.1,11.1,11.1,11.1,11.1,11.1,11.1,11.1]

iters = np.arange(0, 10)  # iter 0..9

plt.figure(figsize=(10, 5))

plt.plot(iters, lap_times_dynamics_nominal, marker='o', linestyle='-',  label='Nominal dynamics')
plt.plot(iters, lap_times_dynamics_true,    marker='o', linestyle='--', label='True dynamics')
plt.plot(iters, lap_times_gp,               marker='o', linestyle='-',  label='GP')
plt.plot(iters, lap_times_modelbank_static_bruteforce,  marker='s', linestyle='--', label='Modelbank static (bruteforce)')
plt.plot(iters, lap_times_modelbank_updated_bruteforce, marker='s', linestyle='-',  label='Modelbank updated (bruteforce)')
plt.plot(iters, lap_times_proposed,         marker='o', linestyle='-',  label='Proposed')

plt.xlabel("Iteration")
plt.ylabel("Lap Time [s]")
plt.title("Lap Time Comparison (iter 0..9)")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
