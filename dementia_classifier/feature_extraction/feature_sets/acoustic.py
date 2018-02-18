import os
import scipy.io.wavfile as wav
import scipy.stats.stats as st
import numpy as np
from python_speech_features import mfcc

def get_mean_var_skew_kurt(np_array):
    return {"mean": np_array.mean(),
            "var": np_array.var(),
            "skewness": st.skew(np_array),
            "kurtosis": st.kurtosis(np_array), }


# mean, variance, skewness, and kurtosis of the first 14 MFCCs
def get_mfcc_features(filename):
    feature_dict = {}
    (rate, sig) = wav.read(filename)

    if sig.ndim == 2:
        # wav is stereo so average over both channels
        mfcc_feat_chan0 = mfcc(sig[:, 0], rate, numcep=15, appendEnergy=True)
        mfcc_feat_chan1 = mfcc(sig[:, 1], rate, numcep=15, appendEnergy=True)
        mfcc_feat = (mfcc_feat_chan0 + mfcc_feat_chan1) / 2
    else:
        mfcc_feat = mfcc(sig, rate, numcep=15, appendEnergy=True)

    # Velocity is the difference between timestep t+1 and t for each mfcc_feat / 2
    vel = (mfcc_feat[:-1, :] - mfcc_feat[1:, :]) / 2.0
    # Acceleration is the difference between timestep t+1 and t for each velocity / 2
    acc = (vel[:-1, :] - vel[1:, :]) / 2.0
    mfcc_means = []
    for i in xrange(0, 14):
        key = "energy" if i == 0 else "mfcc" + str(i)
        # mfcc
        feature_dict[key + "_mean"]     = mfcc_feat[:, i].mean()
        feature_dict[key + "_var"]      = mfcc_feat[:, i].var()
        feature_dict[key + "_skewness"] = st.skew(mfcc_feat[:, i])
        feature_dict[key + "_kurtosis"] = st.kurtosis(mfcc_feat[:, i])
        # Vel
        feature_dict[key + "_vel_mean"]     = vel[:, i].mean()
        feature_dict[key + "_vel_var"]      = vel[:, i].var()
        feature_dict[key + "_vel_skewness"] = st.skew(vel[:, i])
        feature_dict[key + "_vel_kurtosis"] = st.kurtosis(vel[:, i])
        # Accel
        feature_dict[key + "_accel_mean"]     = acc[:, i].mean()
        feature_dict[key + "_accel_var"]      = acc[:, i].var()
        feature_dict[key + "_accel_skewness"] = st.skew(acc[:, i])
        feature_dict[key + "_accel_kurtosis"] = st.kurtosis(acc[:, i])

        # Need the skewness and kurtosis of all mfcc means
        if i > 0:
            mfcc_means.append(feature_dict[key + "_mean"])

    feature_dict["mfcc_skewness"] = st.skew(mfcc_means)
    feature_dict["mfcc_kurtosis"] = st.kurtosis(mfcc_means)
    return feature_dict


def get_fundamental_frequency_mean_var(filename):
    ff = np.loadtxt(open(filename, "rb"), usecols=(1,))
    return ff.mean(), ff.var()


def get_all(path):
    # Extract data from soundfile directory
    if os.path.exists(path):
        parsed_data = {}
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                print "Processing " + filename
                wavfile = os.path.join(path, filename)
                feat = get_mfcc_features(wavfile)
                feat["ff_mean"], feat["ff_var"] = get_fundamental_frequency_mean_var(path + "/pitches/" + filename.replace(".wav", ".txt"))
                parsed_data[filename.replace('.wav', '')] = feat
        return parsed_data
    else:
        raise IOError("File not found: " + path + " does not exist")


if __name__ == '__main__':
    get_all("../data/soundfiles/control")
