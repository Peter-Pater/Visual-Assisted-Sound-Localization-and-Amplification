from scipy.signal import signaltools
import numpy as np
import matplotlib.pyplot as plt

def find_delta_with_xcorr(signal1, signal2):
    if signal1.std() == 0 or signal2.std() == 0:
        return True, None, None
    signal1 -= signal1.mean()
    signal2 -= signal2.mean()
    signal1 /= signal1.std()
    signal2 /= signal2.std()
    nsamples = signal1.shape[0]

    xcorr = signaltools.correlate(signal1, signal2)
    # print(xcorr)
    dt = np.arange(1-nsamples, nsamples)
    filtered_xcorr = xcorr[len(xcorr)//2 - 20 : len(xcorr)//2 + 20]
    recovered_time_shift = dt[len(xcorr)//2 - 20 + filtered_xcorr.argmax()]
    return False, xcorr, recovered_time_shift

def find_delta(signal1, signal2): # signal1 takes place after if +ve
    if signal1.std() == 0 or signal2.std() == 0:
        return True, None

    signal1 -= signal1.mean()
    signal2 -= signal2.mean()
    signal1 /= signal1.std()
    signal2 /= signal2.std()
    nsamples = signal1.shape[0]

    xcorr = signaltools.correlate(signal1, signal2)
    # print(xcorr)
    dt = np.arange(1-nsamples, nsamples)
    filtered_xcorr = xcorr[len(xcorr)//2 - 20 : len(xcorr)//2 + 20]
    recovered_time_shift = dt[len(xcorr)//2 - 20 + filtered_xcorr.argmax()]
    return False, recovered_time_shift

def get_tdoa_phase(loc1, loc2, u, freq, c): # loc1 takes place after if +ve
    return ((np.linalg.norm(u - loc1) / c) - (np.linalg.norm(u - loc2) / c)) * freq

def interpolate(l, idx):
    flr = int(np.floor(idx))
    cl = int(np.ceil(idx))
    ratio = idx % 1.0
    return (1-ratio) * l[flr] + ratio * l[cl]

def get_phase_diffs(pts, mic_locs, sample_freq):
    c = 343 # metres per second
    phase_diffs = {}
    for p in pts:
        locs = list(mic_locs.values())
        phase_diff = []
        for i in range(0, len(locs)-1):
            for j in range(i + 1, len(locs)):
                tdoa_phase = get_tdoa_phase(locs[i], locs[j], p, sample_freq, c)
                phase_diff.append(tdoa_phase)
        phase_diff = np.array(phase_diff)
        phase_diffs[tuple(p)] = tuple(phase_diff)
    return phase_diffs

if __name__ == "__main__":
    # pts = [np.array([0,0,1])]
    # mic_locs = {1:np.array([0,0,0]), 2:np.array([0.08,0,0]), 3:np.array([-0.08,0,0]), 4:np.array([0,-0.08,0])}
    # print(get_phase_diffs(pts, mic_locs, 44100))

    # time = np.arange(0, 10, 0.1)
    # signal2 = np.sin(time/5 * np.pi)
    # signal1 = np.roll(signal2, -1)
    # print(find_delta(signal1, signal2))
    # plt.plot(time, signal1, c = "blue")
    # plt.plot(time, signal2, c = "red")
    # plt.show()

    time = np.arange(0, 10, 0.1)
    signal1 = np.sin(time/5 * np.pi)
    signal2 = np.roll(signal1, 2)
    signal3 = np.roll(signal1, 2)
    print(find_delta(signal2, signal1))
    plt.plot(time, signal1)
    plt.plot(time, signal2)
    plt.show()