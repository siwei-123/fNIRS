import mne
import pandas as pd
from SQI import CVSignal, SNRSignal, FNIRSDataProcessor, FNIRSPeakFrequencyExtractor, FNIRSStatisticalFeaturesExtractor, \
    FNIRSSlopeExtractor, FNIRSHeartPowerCalculator
import os

# 加载 SNIRF 文件
file_path_main = 'fNiRS_data/OneDrive_1_2024-8-1'
count=0
for dirpath, dirnames, filenames in os.walk(file_path_main):
        for file_name in filenames:
            print(count)
            if file_name.endswith('.snirf'):  # 检查文件是否为 SNIRF 文件
                count=count+1
                file_path = os.path.join(dirpath, file_name)

                raw = mne.io.read_raw_snirf(file_path, preload=True)

                # 假设你有一个 fNIRS 信号，可以是 raw.get_data() 返回的数据
                signal = raw.get_data()

                # 使用 CVSignal 类计算变异系数
                cv_calculator = CVSignal(threshold=0.5)
                df = pd.DataFrame(signal.T, columns=raw.ch_names)
                cv_values = df.apply(lambda col: cv_calculator.algorithm(col.values), axis=0)


                # 使用 SNRSignal 类计算信噪比
                snr_calculator = SNRSignal(threshold=0.5)
                df = pd.DataFrame(signal.T, columns=raw.ch_names)
                snr_value = snr_calculator.algorithm(signal)
                SNR_results = df.apply(lambda col: snr_calculator.algorithm(col.values), axis=0)


                # 使用 FNIRSPeakFrequencyExtractor 类计算峰值频率和功率谱密度
                peak_freq_extractor = FNIRSPeakFrequencyExtractor(raw)
                peak_freq_extractor.calculate_peak_frequencies()

                # 获取峰值频率和功率谱密度
                peak_frequencies = peak_freq_extractor.get_peak_frequencies()
                peak_psp = peak_freq_extractor.get_peak_psp()



                # 使用 FNIRSStatisticalFeaturesExtractor 类计算偏度和峰度
                stats_extractor = FNIRSStatisticalFeaturesExtractor(raw)

                # 计算偏度
                skewness_df = stats_extractor.calculate_skewness()


                # 计算峰度
                kurtosis_df = stats_extractor.calculate_kurtosis()


                # 使用 FNIRSSlopeExtractor 类计算斜率
                slope_extractor = FNIRSSlopeExtractor(raw)
                slope_extractor.calculate_slopes()

                # 获取斜率数据框
                slopes_df = slope_extractor.get_slopes_df()


                # 使用 FNIRSDataProcessor 类可视化数据并计算头皮耦合指数(SCI)
                processor = FNIRSDataProcessor(raw)

                # 计算SCI
                sci = processor.calculate_sci()


                sci_psp = sci * sci * peak_psp

                # 使用 FNIRSHeartPowerCalculator 类计算心脏功率 (CP)
                heart_power_calculator = FNIRSHeartPowerCalculator(raw)
                psds, freqs, cp = heart_power_calculator.calculate_psd_and_cp(fmin=0.5, fmax=2.5, n_fft=2048)



                results = {
                    "Channel": raw.ch_names,
                    "CV": cv_values,
                    "SNR": SNR_results,
                    "Peak Frequency": peak_frequencies,
                    "SCI^2*PSP": sci_psp,
                    "Skewness": skewness_df["Skewness"],
                    "Kurtosis": kurtosis_df["Kurtosis"],
                    "Slope": slopes_df["Slope"],
                    "Cardiac Power": cp,

                }

                # 将结果转换为DataFrame
                results_df = pd.DataFrame(results)

                # 保存到Excel文件
                root_path='analyse_outcome'
                output_file_path = f"{os.path.join(root_path, file_name+str(count))}.xlsx"
                results_df.to_excel(output_file_path, index=False)

                print(f"Results have been saved to {output_file_path}")
