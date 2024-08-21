import mne
import pandas as pd
from sqi import CVSignal, SNRSignal, FNIRSPeakFrequencyExtractor, FNIRSStatisticalFeaturesExtractor, \
    FNIRSSlopeExtractor, FNIRSHeartPowerCalculator
import os

file_path_main = 'fNiRS_data/OneDrive_1_2024-8-1'
count=0
for dirpath, dirnames, filenames in os.walk(file_path_main):
        for file_name in filenames:
            print(count)
            if file_name.endswith('.snirf'):
                count=count+1
                file_path = os.path.join(dirpath, file_name)

                raw = mne.io.read_raw_snirf(file_path, preload=True)

                signal = raw.get_data()

                df=pd.DataFrame(raw.ch_names, columns=['Channel'])

                cv_calculator = CVSignal(signal)
                cv_values = cv_calculator.algorithm()
                df1 = pd.DataFrame(cv_values, columns=["CV"])

                SNR_calculate = SNRSignal(signal)
                snr_values = SNR_calculate.algorithm()
                df2 = pd.DataFrame(snr_values, columns=["SNR"])

                peak_freq_extractor = FNIRSPeakFrequencyExtractor(raw)
                peak_freq_extractor.calculate_peak_frequencies()

                peak_frequencies = peak_freq_extractor.get_peak_frequencies()
                peak_psp = peak_freq_extractor.get_peak_psp()

                df3 = pd.DataFrame(peak_frequencies, columns=["Peak Frequency"])

                raw_od = mne.preprocessing.nirs.optical_density(raw)
                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

                sci_psp = sci * sci * peak_psp
                df4 = pd.DataFrame(sci_psp, columns=["SCI^2*PSP"])

                stats_extractor = FNIRSStatisticalFeaturesExtractor(raw)

                stats_extractor.calculate_skewness()

                skewness_df = stats_extractor.get_skewness_df()

                df5 = pd.DataFrame(skewness_df, columns=["Skewness"])

                stats_extractor.calculate_kurtosis()

                kurtosis_df = stats_extractor.get_kurtosis_df()

                df6 = pd.DataFrame(kurtosis_df, columns=["Kurtosis"])

                slope_extractor = FNIRSSlopeExtractor(raw)
                slope_extractor.calculate_slopes()
                slope = slope_extractor.get_slopes_array()
                df7 = pd.DataFrame(slope, columns=["Slope"])

                heart_power_calculator = FNIRSHeartPowerCalculator(raw)
                cp = heart_power_calculator.calculate_psd_and_cp(fmin=0.5, fmax=2.5, n_fft=2048)
                df8 = pd.DataFrame(cp, columns=["Cardiac Power"])

                result = pd.concat([df,df1,df2,df3,df4,df5,df6,df7,df8], axis=1)
                results_df = pd.DataFrame(result)

                root_path = 'analyse_outcome_1'
                output_file_path = f"{os.path.join(root_path, file_name + str(count))}.xlsx"
                results_df.to_excel(output_file_path, index=False)

                print(f"Results have been saved to {output_file_path}")