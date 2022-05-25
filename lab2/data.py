from pathlib import Path

import plotly.express as px
import pandas as pd


def check_dataset(path):
    # evaluation dataset
    DATA_ROOT = Path(path)
    types = [t.name for t in sorted(DATA_ROOT.glob('*')) if t.is_dir()]
    print('Machine types:', types)

    df = pd.DataFrame()
    df['file'] = sorted(DATA_ROOT.glob('*/*/*.wav'))
    df['type'] = df.file.map(lambda f: f.parent.parent.name)
    df['split'] = df.file.map(lambda f: f.parent.name)
    df['id'] = df.file.map(lambda s: str(s).split('/')[-1].split('_')[-2])

    agg = df.groupby(['id', 'split']).agg('count')
    fig = px.bar(agg.reset_index(), x="id", y="file", color="split")
    fig.show()
    print(agg.transpose())


def get_log_mel_spectrogram(filename, n_mels=64,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0):
    wav, sampling_rate = com.file_load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=wav,
                                                     sr=sampling_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram, wav