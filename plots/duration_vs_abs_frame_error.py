import matplotlib.pyplot as plt

# Data
video_durations = [80, 120, 240, 360, 480]
methods = ["median"]
data = {
    "dtw": {
        "abs_frame_error_mean": [15.0989010989011, 13.824175824175825, 10.478021978021978, 10.236263736263735, 9.95054945054945],
        "moe": [1.5092792898383214, 1.6163823808996085, 1.5394737104300698, 1.5756742048590475, 1.877010212661156],
    },
    "log_reg": {
        "abs_frame_error_mean": [19.956043956043956, 15.15934065934066, 19.71978021978022, 21.51098901098901, 22.532967032967033],
        "moe": [2.3121601423763267, 2.0864897243232035, 2.40483215538842, 2.444648542710554, 2.473989336424549],
    },
    "mean": {
        "abs_frame_error_mean": [15.26923076923077, 14.384615384615383, 11.763736263736265, 12.763736263736265, 12.846153846153848],
        "moe": [1.4470816017485228, 1.5293494430278138, 1.4098498503596002, 1.5888967690881943, 1.6657776542236342],
    },
    "median": {
        "abs_frame_error_mean": [15.087912087912088, 14.0, 10.747252747252746, 11.335164835164836, 10.65934065934066],
        "moe": [1.4748691977651955, 1.5512093016252044, 1.436015731452616, 1.6280542815453334, 1.698381779352947],
    },
}

# Plotting
plt.figure(figsize=(10, 6))
for method in methods:
    means = data[method]["abs_frame_error_mean"]
    moe = data[method]["moe"]
    plt.errorbar(
        video_durations, means, yerr=moe, label=method, fmt='o-', capsize=5, linewidth=2
    )

plt.xlabel("Video Duration (frames)")
plt.ylabel("Absolute Frame Error")
plt.title("Absolute Frame Error vs Video Duration")
# plt.legend(title="Methods")
plt.grid(True)
plt.show()

# Save the plot as an image file
plt.savefig("duration_vs_abs_frame_error.png", dpi=300)
