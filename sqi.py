import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from mne.time_frequency import psd_array_welch

#cv
class CVSignal:
    def __init__(self, signal):
        self.signal = signal

    def algorithm(self):
        cv=[]
        for i in range(self.signal.shape[0]):
            column_data = self.signal[i,:]
            mean = np.mean(column_data)
            sd = np.std(column_data)
            cv.append(float(sd / mean))
        return cv






#SNR
class SNRSignal:
    def __init__(self, signal):
        self.signal = signal

    def algorithm(self):
        SNR=[]
        for i in range(self.signal.shape[0]):
            column_data = self.signal[i,:]
            mean = np.mean(column_data)
            sd = np.std(column_data)
            SNR.append(float(mean/sd))
        return SNR








#pf_psp
class FNIRSPeakFrequencyExtractor:
    def __init__(self, raw_data):
        self.data = raw_data.get_data()
        self.target_fs = raw_data.info['sfreq']
        self.peak_freqs = []
        self.psp = []

    def calculate_peak_frequencies(self):

        for i in range(self.data.shape[0]):
            channel_data = self.data[i, :]
            f, Pxx = welch(channel_data, fs=self.target_fs, nperseg=1024)
            peak_freq = f[np.argmax(Pxx)]
            self.peak_freqs.append(peak_freq)
            self.psp.append(max(Pxx))

    def get_peak_frequencies(self):

        return self.peak_freqs

    def get_peak_psp(self):

        return self.psp










#Cp
class FNIRSHeartPowerCalculator:
    def __init__(self, raw):
        self.raw_combined = raw
        self.picks = raw._pick_drop_channels
        self.data = raw.get_data()
        self.sfreq =raw.info['sfreq']


    def calculate_psd_and_cp(self, fmin=0.5, fmax=2.5, n_fft=2048):
        psds, freqs = psd_array_welch(self.data, self.sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
        cp = np.sum(psds * 100, axis=1)
        return  cp




#Slope
class FNIRSSlopeExtractor:
    def __init__(self, raw):
        self.data = raw.get_data()
        self.n_samples = self.data.shape[1]
        self.time = np.arange(self.n_samples).reshape(-1, 1)
        self.slopes = []

    def calculate_slopes(self):
        for i in range(self.data.shape[0]):
            channel_data = self.data[i, :].reshape(-1, 1)
            model = LinearRegression()
            model.fit(self.time, channel_data)
            slope = model.coef_[0][0]
            self.slopes.append(slope)


    def get_slopes_array(self):
        return np.array(self.slopes)



#Skewness and Kurtosis


class FNIRSStatisticalFeaturesExtractor:
    def __init__(self, raw_data):
        self.data = raw_data.get_data()
        self.skewness_values = None
        self.kurtosis_values = None

    def calculate_skewness(self):
        self.skewness_values = skew(self.data, axis=1)


    def calculate_kurtosis(self):
        self.kurtosis_values = kurtosis(self.data, axis=1)


    def get_skewness_df(self):
        return np.array(self.skewness_values)

    def get_kurtosis_df(self):
        return np.array(self.kurtosis_values)



