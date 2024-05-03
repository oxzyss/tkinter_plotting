import os
import math
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# raw_adc=['/home/frbusr/data/current/raw_acq/000000']
raw_adc = [
    '/Users/wayan/asiaa/tkinter_plotting/test_data/20240220T224640Z_corr434/raw_acq']
chan_one = 4
chan_two = 12
plot_type = "coeff"


class Plot():
    """
    Class for plotting data with matplotlib.
    Reads h5 files and plots cross correlation graphs
    """

    def __init__(self, plot_type, filepath, channels, console_frame):
        self.plt = plt
        self.plot_type = plot_type
        self.file_path = filepath
        self.channels = channels
        self.console_frame = console_frame
        self.rawtime = None
        self.rawfpgacount = None
        self.rawdata = None

    def update_data(self, plot_type, filepath, channels):
        self.plot_type = plot_type
        self.file_path = filepath
        self.channels = channels
 
    def plot_data(self):
        '''
        Function to plot data, reads data from class initalization and
        determines what plot to generate
        '''
        if os.path.isfile(self.file_path[0]):
            self.console_frame.log('file path is a file')
            self.rawtime, self.rawfpgacount, self.rawdata = self.read_raw_adc()
        else:
            self.console_frame.log(f'{self.file_path} is not a file')
            return

        self.console_frame.log(f'Plotting: {self.plot_type}')
        # Initial setup
        num_channels = len(self.channels)
        self.console_frame.log(f'num_channels: {num_channels}')
        num_rows = math.ceil(num_channels / 4)
        num_cols = math.ceil(num_channels / num_rows)

        fig = plt.figure(figsize=(12, 9))

        fig.suptitle(f"{self.plot_type}")

        if self.plot_type == "rms":
            plt.subplots_adjust(hspace=0.6, left=0.05, right=0.92,
                                wspace=0.3, top=0.935, bottom=0.08)
            for i in range(1, num_channels + 1):
                index = i - 1
                self.plot_rms(self.rawdata, self.rawtime, self.channels[index])
        else:
            plt.subplots_adjust(hspace=0.6, left=0.05, right=0.95,
                                wspace=0.3, top=0.935, bottom=0.08)
            for i in range(1, num_channels + 1):
                index = i - 1
                ax = fig.add_subplot(num_rows, num_cols, i)

                if self.plot_type == "spectrum":
                    self.plot_spectrum(
                            self.rawdata, self.rawtime, self.channels[index])

                elif self.plot_type == "waterfall":
                    self.plot_waterfall(self.rawdata, self.rawtime,
                                        self.channels[index])

                elif self.plot_type == "correlation_coefficient":
                    self.plot_corr_coeff(self.rawdata, self.rawtime,
                                         self.channels[index])

                elif self.plot_type == "correlation_phase":
                    self.plot_corr_phase(self.rawdata, self.rawtime,
                                         self.channels[index])

                elif self.plot_type == "correlation_magnitude":
                    self.plot_corr_magnitude(self.rawdata, self.rawtime,
                                             self.channels[index])


        self.plt.show()

    def read_raw_adc(self, verbose=1):
        """
        Read raw ADC data of one iceboard.
        """
        adc_input_to_sma = np.array(
            [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])
        inputmap_dict = {'FCA000000': 0, 'FCA000001': 1, 'FCA000002': 2, 'FCA000003': 3, 'FCA000004': 4, 'FCA000005': 5,
                         'FCA000006': 6, 'FCA000007': 7, 'FCA000008': 8, 'FCA000009': 9, 'FCA000010': 10, 'FCA000011': 11,
                         'FCA000012': 12, 'FCA000013': 13, 'FCA000014': 14, 'FCA000015': 15}

        FMT = 'FCA0000{:02d}'
        for rr, rf in enumerate(self.file_path):

            if verbose:
                print(rf)

            with h5py.File(rf, 'r') as handler:
                # Figure out the unique fpga frame numbers
                timestamp = handler['timestamp'][:, 0]

                uniq_fpga_count, iuniq, itime = np.unique(timestamp['fpga_count'],
                                                          return_index=True,
                                                          return_inverse=True)
                ctime = timestamp['ctime'][iuniq]
                ntime = ctime.size

                # Load in the full timestream
                timestream = handler['timestream'][:]

                # Repackage into an array that is nsample, ninput, ntime
                npacket, nsample = timestream.shape

                sn_to_id = inputmap_dict
                ninput = 16
                sma = adc_input_to_sma[handler["adc_input"][:, 0]]
                # sma = handler['adc_input'][:, 0]

                data = np.zeros((ninput, ntime, nsample),
                                dtype=timestream.dtype)

                for jj, (tt, ii) in enumerate(zip(itime, sma)):
                    sn = FMT.format(ii)
                    data[sn_to_id[sn], tt, :] = timestream[jj, :]

                if not rr:
                    rawtime = ctime
                    rawfpgacount = uniq_fpga_count.copy()
                    rawdata = data
                else:
                    rawtime = np.concatenate((rawtime, ctime))
                    rawfpgacount = np.concatenate(
                        (rawfpgacount, uniq_fpga_count))
                    rawdata = np.concatenate((rawdata, data), axis=1)

        return rawtime, rawfpgacount, rawdata

    def plot_corr_coeff(self, rawdata, rawtime, input_one, input_two):
        Ninput, Nframes, Framelength = rawdata.shape
        f_MHz = 800. - np.arange(1024)*400./1024.
        y1 = rawdata[input_one, :]
        y2 = rawdata[input_two, :]

        fft_y1 = np.fft.fft(np.hamming(Framelength) *
                            (y1-np.mean(y1, axis=1)[:, np.newaxis]), axis=1)
        fft_y2 = np.fft.fft(np.hamming(Framelength) *
                            (y2-np.mean(y2, axis=1)[:, np.newaxis]), axis=1)

        print("FFT: done")
        corr_y1y2 = fft_y1[:, :1024]*fft_y2[:, :1024].conj()
        corr_y1y1 = fft_y1[:, :1024]*fft_y1[:, :1024].conj()
        corr_y2y2 = fft_y2[:, :1024]*fft_y2[:, :1024].conj()
        print("Correlation: done")

        corr_coeff = corr_y1y2 / np.sqrt(corr_y1y1 * corr_y2y2)
        coeff_mean = np.mean(corr_coeff, axis=0)

        # correlation_matrix = np.corrcoef(fft_y1, fft_y2)

        self.plt.plot(f_MHz, np.abs(coeff_mean))
        # self.plt.plot(f_MHz[:min_length], correlation_coefficient[:min_length])
        # self.plt.plot(f_MHz, np.mean(correlation_coefficient, axis=0))

    def plot_corr_magnitude(self, rawdata, rawtime, input_one, input_two):
        Ninput, Nframes, Framelength = rawdata.shape
        f_MHz = 800. - np.arange(1024)*400./1024.
        y1 = rawdata[input_one, :]
        y2 = rawdata[input_two, :]

        fft_y1 = np.fft.fft(np.hamming(Framelength) *
                            (y1-np.mean(y1, axis=1)[:, np.newaxis]), axis=1)
        fft_y2 = np.fft.fft(np.hamming(Framelength) *
                            (y2-np.mean(y2, axis=1)[:, np.newaxis]), axis=1)

        print("FFT: done")
        corr_y1y2 = fft_y1[:, :1024]*fft_y2[:, :1024].conj()
        print("Correlation: done")

        # Calculate the magnitude of the cross-correlation
        magnitude_corr_y1y2 = np.abs(corr_y1y2)

        self.plt.plot(f_MHz, np.mean(magnitude_corr_y1y2, axis=0))

    def plot_corr_phase(self, rawdata, rawtime, input_one, input_two):
        Ninput, Nframes, Framelength = rawdata.shape
        f_MHz = 800. - np.arange(1024)*400./1024.
        y1 = rawdata[input_one, :]
        y2 = rawdata[input_two, :]

        fft_y1 = np.fft.fft(np.hamming(Framelength) *
                            (y1-np.mean(y1, axis=1)[:, np.newaxis]), axis=1)
        fft_y2 = np.fft.fft(np.hamming(Framelength) *
                            (y2-np.mean(y2, axis=1)[:, np.newaxis]), axis=1)

        print("FFT: done")
        corr_y1y2 = fft_y1[:, :1024]*fft_y2[:, :1024].conj()
        print("Correlation: done")

        # Calculate the phase shift of the cross-correlation
        phase_shift = np.angle(corr_y1y2)

        # min_length = min(len(f_MHz), len(phase_shift))

        # self.plt.plot(f_MHz[:min_length], phase_shift[:min_length])
        # self.plt.ylim(-4,4)
        self.plt.plot(f_MHz, np.mean(phase_shift, axis=0))

        # self.plt.ylim(-4,4)

    def plot_waterfall(self, rawdata, rawtime, channel):
        Ninput, Nframes, Framelength = rawdata.shape
        #f_MHz = 800. - np.arange(1024)*400./1024.
        y = rawdata[channel, :]

        fft_im = np.abs(np.fft.fft(np.hamming(Framelength) *
                        (y-np.mean(y, axis=1)[:, np.newaxis]), axis=1))**2
        vmin, vmax = np.percentile(fft_im, [1, 99])
        plt.imshow(fft_im[:, 1024:], vmin=vmin, vmax=vmax, extent=[400, 800, 0, rawtime[-1]-rawtime[0]],
                   aspect='auto')
        plt.xlabel('MHz')
        plt.ylabel('Time (s)')
        plt.title('input %d' % channel)

    def plot_spectrum(self, rawdata, rawtime, the_input):
        Ninput, Nframes, Framelength = rawdata.shape
        f_MHz = 800. - np.arange(1024)*400./1024.
        y = rawdata[the_input, :]

        fft = np.median(np.abs(np.fft.fft(np.hamming(Framelength)*(y-np.mean(y, axis=1)[:, np.newaxis]), axis=1))**2,
                        axis=0)
        self.plt.plot(f_MHz, fft[:1024], label='%d')
        self.plt.yscale('log')
        self.plt.ylabel('power')
        self.plt.xlabel('MHz')

        self.plt.title('input %d' % (the_input + 1))

    def plot_rms(self, rawdata, rawtime, channel):
        rawrms = np.std(rawdata, axis=2)
        rawrms.shape
        plt.plot(rawtime-rawtime[0], rawrms[channel,:], label=channel)

        plt.legend(loc=(1.02,0.0))
        plt.ylabel('RMS of each input')
        plt.xlabel('time (s)')

# plot RMS:
# rawrms = np.std(rawdata_all, axis=2)
# rawrms.shape
# for i in range(0,16):
#    plt.plot(rawtime-rawtime[0], rawrms[i,:], label=i)
# plt.legend(loc=(1.02,0.0))
# plt.ylabel('RMS of each input')
# plt.xlabel('time (s)')
# dt=datetime.fromtimestamp(np.mean(rawtime))
# plt.savefig('%s_rms_inputs.png'%dt.strftime("%Y%m%dT%H%M%S"), format='png',bbox_inches='tight',dpi=100)
#
# print ('Produced RMS plot: %s_rms_inputs.png'%dt.strftime("%Y%m%dT%H%M%S"))
#
# def plot_waterfall(rawdata, rawtime, the_input):
#    Ninput, Nframes, Framelength = rawdata.shape
#    f_MHz = 800. - np.arange(1024)*400./1024.
#    y = rawdata[the_input,:]
#
#    fft_im = np.abs(np.fft.fft(np.hamming(Framelength)*(y-np.mean(y, axis=1)[:, np.newaxis]), axis=1))**2
#    vmin, vmax=np.percentile(fft_im, [1,99])
#    plt.imshow(fft_im[:,1024:], vmin=vmin, vmax=vmax, extent=[400,800,0,rawtime[-1]-rawtime[0]],
#               aspect='auto')
#    plt.xlabel('MHz')
#    plt.ylabel('Time (s)')
#    plt.title('input %d'%the_input)
#
# fig=plt.figure(figsize=(10,10))
# for i in range(0,16):
#    rect=[0.27*(i%4), -0.27*(i//4), 0.2, 0.2]
#    fig.add_axes(rect)
#    frame1=plt.gca()
#    plot_spectrum(rawdata_all, rawtime, i)
#
# plt.savefig('%s_all_inputs_spectra.png'%dt.strftime("%Y%m%dT%H%M%S"), format='png',bbox_inches='tight',dpi=100)
#
# print ('Produced spectrum plot: %s_all_inputs_spectra.png'%dt.strftime("%Y%m%dT%H%M%S"))
#
#
# fig=plt.figure(figsize=(10,10))
# for i in range(0,16):
#    rect=[0.27*(i%4), -0.27*(i//4), 0.2, 0.2]
#    fig.add_axes(rect)
#    frame1=plt.gca()
#    plot_waterfall(rawdata_all, rawtime, i)
#
# plt.savefig('%s_all_inputs_waterfall.png'%dt.strftime("%Y%m%dT%H%M%S"), format='png',bbox_inches='tight',dpi=100)
# print ('Produced waterfall plot: %s_all_inputs_waterfall.png'%dt.strftime("%Y%m%dT%H%M%S"))


if __name__ == "__main__":
    path = ['/Users/wayan/asiaa/tkinter_plotting/test_data/20240220T224640Z_corr434/raw_acq/000000']
    plot_type = 'spectrum'
    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plot = Plot(plot_type, path, channels, None)
    plot.plot_data()
