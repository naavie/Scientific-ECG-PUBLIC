def load_wsdb(file):
    file = os.path.splitext(file)[0]
    record = wfdb.io.rdrecord(file)
    ecg = record.p_signal.T.astype('float32')
    leads = tuple(record.sig_name)
    sr = record.fs
    ecg[np.isnan(ecg)] = 0.0
    return ecg, leads, sr

def load_raw_data(df, sampling_rate, path):
    data = []
    not_found_files = []
    if sampling_rate == 100:
        filenames = df.filename_lr
    else:
        filenames = df.filename_hr

    for f in filenames:
        try:
            record = wfdb.rdsamp(path+f)
            data.append(record)
        except FileNotFoundError:
            not_found_files.append(f)
            continue

    data = np.array([signal for signal, meta in data])
    return data, not_found_files


def resample(ecg, shape):
    resized = cv2.resize(ecg, (shape, ecg.shape[0]))
    resized = resized.astype(ecg.dtype)
    return resized

class CLIP_ECG_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, config):
        self.df = df
        self.df['filename_hr'] = '/content/PTB-XL/' + self.df['filename_hr'].astype(str) + '.hea'
        self.config = config

        self.ecg_files = self.df['filename_hr'].values
        self.captions = self.df['report_ENG'].values

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        ecg, leads, sr = load_wsdb(self.ecg_files[idx])
        caption = self.captions[idx]
        image = self.process_ecg(ecg, sr)
        return {'image': image, 'caption': caption}

    def process_ecg(self, ecg, sr):
        new_shape = int(self.config.ecg_sr * ecg.shape[1] / sr)
        ecg = resample(ecg, new_shape)
        return ecg