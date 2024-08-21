import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch


class CVSignal:
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold
        self.dimensions = {'time': 1}

    def algorithm(self, signal):
        signal_values = signal.ravel()  # Changed to handle numpy array directly
        mean = np.mean(signal_values)
        sd = np.std(signal_values)
        cv = float(sd / mean)
        return cv


class SNRSignal:
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold
        self.dimensions = {'time': 1}

    def algorithm(self, signal):
        signal_values = signal.ravel()  # Changed to handle numpy array directly
        mean = np.mean(signal_values)
        sd = np.std(signal_values)
        SNR = float(mean / sd)
        return SNR


class FNIRSPeakFrequencyExtractor:
    def __init__(self, raw_data, target_fs=1000):
        """
        初始化FNIRSPeakFrequencyExtractor类。

        参数:
        - raw_data: fNIRS的原始数据 (使用MNE-Python的Raw对象)
        - target_fs: 采样频率，默认为1000Hz
        """
        self.data = raw_data.get_data()  # 获取数据
        self.target_fs = target_fs  # 目标采样频率
        self.peak_freqs = []  # 存储峰值频率的列表
        self.psp = []  # 存储峰值功率谱密度的列表

    def calculate_peak_frequencies(self):
        """
        计算所有通道的峰值频率和峰值功率谱密度，并存储在类的属性中。
        """
        for i in range(self.data.shape[0]):
            channel_data = self.data[i, :]  # 选择第 i 个通道数据
            f, Pxx = welch(channel_data, fs=self.target_fs, nperseg=1024)  # 计算功率谱密度 (PSD)
            peak_freq = f[np.argmax(Pxx)]  # 找到峰值频率
            self.peak_freqs.append(peak_freq)  # 保存峰值频率
            self.psp.append(max(Pxx))  # 保存峰值功率谱密度

    def get_peak_frequencies(self):
        """
        返回计算的峰值频率列表。

        返回:
        - peak_freqs: 峰值频率的列表
        """
        return self.peak_freqs

    def get_peak_psp(self):
        """
        返回计算的峰值功率谱密度列表。

        返回:
        - psp: 峰值功率谱密度的列表
        """
        return self.psp



class FNIRSStatisticalFeaturesExtractor:
    def __init__(self, raw_data):
        """
        初始化FNIRSStatisticalFeaturesExtractor类。

        参数:
        - raw_data: fNIRS的原始数据 (使用MNE-Python的Raw对象)
        """
        self.data = raw_data.get_data()  # 获取数据
        self.channel_names = raw_data.ch_names  # 获取通道名称
        self.skewness_values = None  # 存储偏度的数组
        self.kurtosis_values = None  # 存储峰度的数组

    def calculate_skewness(self):
        """
        计算每个通道的偏度，并存储在类的属性中。
        """
        self.skewness_values = skew(self.data, axis=1)
        return self.get_skewness_df()

    def calculate_kurtosis(self):
        """
        计算每个通道的峰度，并存储在类的属性中。
        """
        self.kurtosis_values = kurtosis(self.data, axis=1)
        return self.get_kurtosis_df()

    def get_skewness_df(self):
        """
        返回存储偏度值的数据框。

        返回:
        - skewness_df: 存储偏度值的DataFrame
        """
        if self.skewness_values is not None:
            skewness_df = pd.DataFrame(self.skewness_values, index=self.channel_names, columns=["Skewness"])
            return skewness_df
        else:
            raise ValueError("Skewness values have not been calculated. Call calculate_skewness() first.")

    def get_kurtosis_df(self):
        """
        返回存储峰度值的数据框。

        返回:
        - kurtosis_df: 存储峰度值的DataFrame
        """
        if self.kurtosis_values is not None:
            kurtosis_df = pd.DataFrame(self.kurtosis_values, index=self.channel_names, columns=["Kurtosis"])
            return kurtosis_df
        else:
            raise ValueError("Kurtosis values have not been calculated. Call calculate_kurtosis() first.")







class FNIRSSlopeExtractor:
    def __init__(self, raw_data):
        """
        初始化FNIRSSlopeExtractor类。

        参数:
        - raw_data: fNIRS的原始数据 (使用MNE-Python的Raw对象)
        """
        self.data = raw_data.get_data()  # 获取数据
        self.n_samples = self.data.shape[1]  # 获取样本数
        self.time = np.arange(self.n_samples).reshape(-1, 1)  # 创建时间序列
        self.channel_names = raw_data.ch_names  # 获取通道名称
        self.slopes = []  # 存储斜率的列表

    def calculate_slopes(self):
        """
        计算每个通道的斜率，并存储在类的属性中。
        """
        for i in range(self.data.shape[0]):
            channel_data = self.data[i, :].reshape(-1, 1)  # 获取第 i 个通道数据
            model = LinearRegression()  # 初始化线性回归模型
            model.fit(self.time, channel_data)  # 拟合线性模型
            slope = model.coef_[0][0]  # 提取斜率
            self.slopes.append(slope)  # 存储斜率

    def get_slopes_array(self):
        """
        返回存储斜率的数组。

        返回:
        - slopes_array: 存储斜率的NumPy数组
        """
        return np.array(self.slopes)

    def get_slopes_df(self):
        """
        返回存储斜率值的数据框。

        返回:
        - slopes_df: 存储斜率值的DataFrame
        """
        slopes_df = pd.DataFrame(self.get_slopes_array(), index=self.channel_names, columns=["Slope"])
        return slopes_df


class FNIRSDataProcessor:
    def __init__(self, raw_od):
        """
        初始化FNIRSDataProcessor类。

        参数:
        - raw_od: fNIRS的光密度数据 (使用MNE-Python的Raw对象)
        """
        self.raw_od = raw_od  # 光密度数据

    def plot_data(self, duration=500, show_scrollbars=False):
        """
        可视化fNIRS数据。

        参数:
        - duration: 可视化的时长，单位为秒。默认为500秒。
        - show_scrollbars: 是否显示滚动条。默认为False。
        """
        self.raw_od.plot(n_channels=len(self.raw_od.ch_names), duration=duration, show_scrollbars=show_scrollbars)

    def calculate_sci(self):
        """
        计算头皮耦合指数(SCI)。

        返回:
        - sci: 计算的SCI值
        """
        raw_od = mne.preprocessing.nirs.optical_density(self.raw_od)
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
        return sci

class FNIRSHeartPowerCalculator:
    def __init__(self, raw_combined):
        """
        初始化FNIRSHeartPowerCalculator类。

        参数:
        - raw_combined: fNIRS的原始数据 (使用MNE-Python的Raw对象)
        """
        self.raw_combined = raw_combined  # 原始fNIRS数据
        self.picks = self._pick_fnirs_channels()  # 选择fNIRS通道
        self.data = self._get_data()  # 获取fNIRS通道的数据
        self.sfreq = self.raw_combined.info['sfreq']  # 获取采样频率

    def _pick_fnirs_channels(self):
        """
        选择fNIRS通道。

        返回:
        - picks: fNIRS通道的索引
        """
        picks = mne.pick_types(self.raw_combined.info, fnirs=True)
        if len(picks) == 0:
            raise ValueError("No fNIRS channels found in the data.")
        return picks

    def _get_data(self):
        """
        获取fNIRS通道的数据。

        返回:
        - data: fNIRS数据
        """
        data = self.raw_combined.get_data(picks=self.picks)
        if np.all(data == 0):
            raise ValueError("All data values are zero.")
        return data

    def calculate_psd_and_cp(self, fmin=0.5, fmax=2.5, n_fft=2048):
        """
        计算功率谱密度(PSD)并计算心脏功率(CP)。

        参数:
        - fmin: 最小频率，默认为0.5Hz
        - fmax: 最大频率，默认为2.5Hz
        - n_fft: FFT点数，默认为2048

        返回:
        - psds: 计算的功率谱密度
        - freqs: 对应的频率
        - cp: 计算的心脏功率
        """
        psds, freqs = psd_array_welch(self.data, self.sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
        cp = np.sum(psds * 100, axis=1)  # 计算心脏功率
        return psds, freqs, cp