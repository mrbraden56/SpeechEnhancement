import sys
import os
from natsort import natsorted
from pydub import AudioSegment
from typing import List, Dict
import numpy as np
import librosa
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re
import shutil





class SNR:
    def make_soundDB_equal(self, clean, background):
        if clean.dBFS < background.dBFS:
            db = background.dBFS
            db = db * -2
            background = background.apply_gain(db)
            difference = int(clean.dBFS + background.dBFS)
            background = background + difference
            background = background.apply_gain(clean.dBFS - background.dBFS)
            # print(f'Clean: {clean.dBFS} Combined: {background.dBFS}')
            return clean, background
        else:
            db = clean.dBFS
            db = db * -2
            clean = clean.apply_gain(db)
            difference = int(background.dBFS + clean.dBFS)
            clean = clean + difference
            clean = clean.apply_gain(-1 * (clean.dBFS - background.dBFS))
            # print(f'Clean: {clean.dBFS} Combined: {background.dBFS}')
            return clean, background


class FolderSize:
    def __init__(self, path: str):
        self.base: str = path

    def create_folder(self, i: int) -> str:
        os.chdir(self.base)
        if (os.getcwd() == self.base):
            os.mkdir(self.base + "/" + str(i))
            temp: str = str(self.base + "/" + str(i))
            return temp


class ConvertToWav:
    def __init__(self, listOfFiles: List[str], source: str, destination: str, CreateFolder: FolderSize,
                 folder_size: int, audio_features: Dict[str, str]):
        self.listOfFiles = listOfFiles
        self.source = source
        self.destination = destination
        self.CreateFolder = CreateFolder
        self.folder_size = folder_size
        self.audio_features = audio_features

    def change_audio_features(self, audio: AudioSegment) -> AudioSegment:
        audio = audio.set_frame_rate(self.audio_features['frame_rate'])
        audio = audio.set_channels(self.audio_features['channels'])
        return audio

    def equivalent_parameters(self, framerate: int, channels: int) -> bool:
        if framerate == self.audio_features['frame_rate'] and channels == self.audio_features['channels']:
            return True
        else:
            return False

    def ConvertFiles(self) -> None:
        os.mkdir(self.destination)
        for i, file in enumerate(self.listOfFiles):
            try:
                if (i % self.folder_size == 0):
                    print(f'Creating Folder: {i}')
                    self.destination = self.CreateFolder.create_folder(i)
                sound: AudioSegment = AudioSegment.from_mp3(self.source + '/' + file)
                sound: AudioSegment = self.change_audio_features(sound)
                if (self.equivalent_parameters(sound.frame_rate, sound.channels)):
                    print(f'Exporting audio file {i} of {len(self.listOfFiles)}')
                    sound.export(self.destination + '/' + file[:-3] + 'wav', format="wav")
                else:
                    print("Features not matching")
                    os.remove(self.source + '/' + file)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_tb.tb_lineno)
                os.remove(self.source + '/' + file)

    def run(self) -> None:
        self.ConvertFiles()


class CombineAudio:
    def __init__(self, Clean_folder: str, Background_folder: str, Combined_folder: str, SNR: SNR
                 , audio_features: Dict[str, str]) -> None:
        self.Clean_folder_base = Clean_folder
        self.Background_folder = Background_folder
        self.Combined_folder = Combined_folder
        self.files_background: List[str] = natsorted(os.listdir(self.Background_folder))
        self.SNR = SNR
        self.folders: List[str] = natsorted(os.listdir(self.Clean_folder_base))
        self.audio_features = audio_features

    def change_audio_features(self, audio: AudioSegment) -> AudioSegment:
        audio = audio.set_frame_rate(self.audio_features['frame_rate'])
        audio = audio.set_channels(self.audio_features['channels'])
        return audio

    def equivalent_parameters(self, framerate, channels) -> bool:
        if framerate == self.audio_features['frame_rate'] and channels == self.audio_features['channels']:
            return True
        else:
            return False

    def combine_audio(self) -> None:
        os.mkdir(self.Combined_folder)
        """Looping over clean folders"""
        for folder in self.folders:
            # Creating a list of audio files in each folder
            data: str = self.Clean_folder_base + '/' + folder
            files: List[str] = natsorted(os.listdir(data))
            os.mkdir(self.Combined_folder + '/' + folder)
            """Looping over clean folders"""
            for i, (clean, background) in enumerate(zip(files, self.files_background)):
                print(f'Folder: {folder} File: {i} of {len(files)}')
                try:
                    clean_audio: AudioSegment = AudioSegment.from_mp3(data + '/' + clean)
                    background_audio: AudioSegment = AudioSegment.from_wav('/home/braden/Environments/Research/Audio/Research(Refactored)/Data/UrbanSound8K/344-3-4-0.wav')#self.Background_folder + '/' + background
                    clean_audio, background_audio = self.SNR.make_soundDB_equal(clean_audio, background_audio)
                    overlapped: AudioSegment = clean_audio.overlay(background_audio, loop=True)
                    overlapped_audio: AudioSegment = self.change_audio_features(overlapped)
                    if self.equivalent_parameters(overlapped_audio.frame_rate, overlapped_audio.channels):
                        p=re.compile(r'-.-')
                        m=p.search(background).group()
                        file_handle = overlapped_audio.export(
                            self.Combined_folder + '/' + folder + '/' + clean[:-4] + m[:-1] + '.wav', format="wav")
                    else:
                        print("Features not matching")
                        os.remove(self.Clean_folder_base + '/' + folder + '/' + clean)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(e, exc_tb.tb_lineno)
                    os.remove(self.Clean_folder_base + '/' + folder + '/' + clean)

    def run(self) -> None:
        self.combine_audio()


class New_Folder:
    def __init__(self, base):
        self.base = base

    def create_folder(self, folder_name):
        os.chdir(self.base)
        if os.getcwd() == self.base:
            os.mkdir(self.base + "/" + folder_name)
            return self.base + "/" + folder_name


class Extract_Files:
    def __init__(self):
        self.audio_file: int = 0
        self.audio_file_dir: str = ''
        self.data: List[np.array] = []

    def your_counter(self, count, stop):
        if stop == count:
            # count = 0 for periodic break
            return True
        else:
            return False

    def extract_single_frame(self, stft_combined: np.array, stft_clean: np.array, export_to: str, combined: str) -> None:
        # (frequency bins, frames)
        self.audio_file_dir = export_to
        os.mkdir(self.audio_file_dir)
        for i in range(0, stft_clean.shape[1]):
            stft_combined_split = stft_combined[:, i]  # [0,1]
            stft_clean_split = stft_clean[:, i]
            self.extract_files(stft_combined_split, stft_clean_split, i, combined)

    def extract_frames(self, stft_combined: np.array, stft_clean: np.array, export_to: str, combined: str) -> None:
        # (frequency bins, frames)
        # self.audio_file_dir = export_to + '/' + str(self.audio_file)
        self.audio_file_dir = export_to
        os.mkdir(self.audio_file_dir)
        for i in range(0, stft_clean.shape[1]):
            stft_combined_split = stft_combined[:, i:i + 8]  # [0,1]
            stft_clean_split = stft_clean[:, i:i + 1]
            self.extract_files(stft_combined_split, stft_clean_split, i, combined, True)

    def extract_files(self, stft_combined: np.array, stft_clean: np.array, count: int, combined_name: str, multi=False) -> None:
        # I had to change dimensions because numpy has a weird bug where they cant be the same dimensions
        if multi:
            zeros: np.array = np.zeros((1, 8))
            stft_combined = np.append(zeros, stft_combined, axis=0)
        self.data.append(np.array(stft_combined))
        self.data.append(np.array(stft_clean))
        np.save(self.audio_file_dir + '/' + combined_name[:-4] + '-' + str(count) + '.wav', self.data)  # saving as [combined, clean] combined_name[:-4]
        self.data.clear()

    def spectrogram(self, stft_combined: np.array, stft_clean: np.array, export_to: str, combined: str):
        self.data.append(np.array(stft_combined))
        self.data.append(np.array(stft_clean))
        np.save(export_to + '/' + combined[:-4] + '.wav', self.data)  # saving as [combined, clean]
        self.data.clear()


class LogSpectrogram:
    def __init__(self, source_clean: str, source_combined: str, export_to: str, extract_files: Extract_Files
                 , audio_features: Dict[str, str]):
        self.source_clean = source_clean
        self.source_combined = source_combined
        self.files_clean: list = natsorted(os.listdir(self.source_clean))
        self.files_combined: list = natsorted(os.listdir(self.source_combined))
        self.base = export_to
        self.export_to = export_to
        self.extract = extract_files
        self.audio_features = audio_features

    # def normalize_tanh_estimator(self, spectrogram):
    #     # scalar=StandardScaler(copy=False)
    #     # scalar.fit(spectrogram)
    #     # normalized=scalar.transform(spectrogram, copy=True)
    #     zscore=stats.zscore(spectrogram)
    #     return zscore

    def _remove_silent_frames(self, audio) -> np.array:
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=round(0.25 * 256), top_db=20)  # hop_length->256
        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def show_spectrogram(self, array) -> None:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(array, ref=np.max), y_axis='linear', x_axis='time',
                                       ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def create_logspectrogram(self, file, source, frames=False) -> np.array:
        signal, sr = librosa.load(source + "/" + file, sr=self.audio_features['frame_rate'])
        #signal=self._remove_silent_frames(signal)
        # taking abs only keeps the magnitude informations and gets rid of complex numbers and phase
        stft: np.array = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=256, hop_length=round(256 * 0.25),
                                             win_length=256, window=scipy.signal.hamming(256, sym=False), center=True)))
        # normalize = MinMaxScaler(feature_range=(-1, 1))
        # normalized_stft = normalize.fit_transform(stft)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        stft=scaler.fit_transform(stft)
        if frames:  # This is added so when we extract the first 8 frames of combined it lands on 1 for clean
            zeros = np.zeros((129, 7))
            stft = np.append(zeros, stft, axis=1)
        return stft

    def multi_framed_logspectrogram(self) -> None:
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        # os.mkdir(self.export_to)
        # print("Running")
        # for folder in self.files_clean:
        #     combined_dir: str = self.source_combined + '/' + folder
        #     clean_dir: str = self.source_clean + '/' + folder
        #     self.export_to: str = self.base + '/' + folder
        os.mkdir(self.export_to)
        # zip_list = zip(natsorted(os.listdir(combined_dir)), natsorted(os.listdir(clean_dir)))
        zip_list = zip(self.files_combined, self.files_clean)
        for i, (combined, clean) in enumerate(zip_list):
            try:
                combined_log: np.array = self.create_logspectrogram(combined, self.source_combined, frames=True)
                clean_log: np.array = self.create_logspectrogram(clean, self.source_clean)
                print(f'Exporting spectrogram {i} of {len(self.files_combined)}')
                self.extract.extract_frames(combined_log, clean_log, self.export_to+'/'+combined[:-4], combined)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_tb.tb_lineno)
        print("Saving List...")

    def single_framed_logspectrogram(self):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        os.mkdir(self.export_to)
        zip_list = zip(self.files_combined, self.files_clean)
        for i, (combined, clean) in enumerate(zip_list):
            try:
                combined_log: np.array = self.create_logspectrogram(combined, self.source_combined)
                clean_log: np.array = self.create_logspectrogram(clean, self.source_clean)
                print(f'Exporting spectrogram {i} of {len(self.files_combined)}')
                self.extract.extract_single_frame(combined_log, clean_log, self.export_to+'/'+combined[:-4], combined)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_tb.tb_lineno)
        print("Saving List...")

class Unpack:
    def __init__(self, dir, export_to, count):
        # self.urbanSoundAudioPaths=natsorted(os.listdir(urbanSoundAudioPaths))
        self.dir=natsorted(os.listdir(dir))
        self.dir_base=dir
        self.export_to=export_to
        self.count=count

    def process_audio(self):
        for i, folder in enumerate(self.dir):
            print(f'Unpacking folder {i}')
            if i==self.count:
                break
            for file in natsorted(os.listdir(self.dir_base+folder)):
                try:
                    shutil.move(self.dir_base+folder+'/'+file, self.export_to)
                except:
                    print('Error')
            shutil.rmtree(self.dir_base+folder)


    def run(self):
        self.process_audio()


def main(arguments):
    """Main func."""
    audio_features: Dict[str, str] = {'frame_rate': 1000, 'channels': 1}

    source: str="/home/braden/Environments/Research/Audio/Research(Refactored)/Data/Mozilla_MP3"
    files: List[str]=os.listdir(source)
    destination: str="/home/braden/Environments/Research/Audio/Research(Refactored)/Data/Mozilla_WAV"
    CreateFolder: FolderSize=FolderSize(destination)
    folder_size: int=10
    convertToWave: ConvertToWav=ConvertToWav(files, source, destination, CreateFolder, folder_size, audio_features)
    convertToWave.run()
    
    Clean_folder: str = "/home/braden/Environments/Research/Audio/Research(Refactored)/Data/Mozilla_WAV"
    Background_folder: str = "/home/braden/Environments/Research/Audio/Research(Refactored)/Data/UrbanSound8K"
    Combined_folder: str = "/home/braden/Environments/Research/Audio/Research(Refactored)/Data/Combined"
    snr: SNR=SNR()
    start: CombineAudio = CombineAudio(Clean_folder, Background_folder, Combined_folder, snr, audio_features)
    start.run()

    source_clean: str = "/home/braden/Work/JayHear/MS-SNSD/low_sr/clean"
    source_combined: str = "/home/braden/Work/JayHear/MS-SNSD/low_sr/noisy"
    export_to: str = "/home/braden/Work/JayHear/MS-SNSD/low_sr/Spectrograms"
    extract_files: Extract_Files = Extract_Files()
    test: LogSpectrogram = LogSpectrogram(source_clean, source_combined, export_to, extract_files, audio_features)
    test.multi_framed_logspectrogram()

    dir="/home/braden/Environments/Spectrograms/"
    export_to="/home/braden/Environments/data/clean"
    count=420
    unpack=Unpack(dir, export_to, count)
    unpack.run()


if __name__ == "__main__":
    main(sys.argv[1:])