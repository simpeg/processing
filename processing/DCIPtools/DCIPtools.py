#################################################
# Imports

import numpy as np
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial
from SimPEG.EM.Static import DC
import properties
##################################################
# Playing around


class baseKernel(object):

    kernel = properties.Array(
        "array containing filter kernal",
        default=np.zeros(1)
    )

    num_ensembles = properties.Integer(
        "number of ensembles",
        default=0)

    reject_std = properties.Float(
        "Tolerance for rejecting an ensemble or stat reject",
        default=1.,
        required=True)

    def __init__(self, filtershape=None, **kwargs):
        if filtershape is not None:
            if isinstance(filtershape, np.ndarray):
                self.filtershape = filtershape
                self.createFilterKernal()

    def createFilterKernal(self):
        """
        creates the filter kernal

        """
        if len(self.filtershape) > 1:
            # create the filter kernal
            tkernal = np.ones((3, 1))
            tkernal[1] = -2.0                    # 3 point kernal
            bsgn = np.ones((1, self.filtershape.size))
            bsgn[0, 1::2] = bsgn[0, 1::2] * -1   # alternate pol rem lin drift
            bwtd = np.matmul(tkernal,
                             bsgn * self.filtershape)  # filter weights
            tmp1 = np.arange(1, 4)
            tmp1 = np.reshape(tmp1, (3, 1))
            tmp2 = np.ones((1, self.filtershape.size))
            tmp3 = np.ones((3, 1))
            tmp4 = np.arange(self.filtershape.size)
            tmp4 = np.reshape(tmp4, (1, self.filtershape.size))
            knew = (np.matmul(tmp1, tmp2) +
                    np.matmul(tmp3, (tmp4 * (self.filtershape.size + 3))))
            btmp = np.zeros((self.filtershape.size + 2,
                             self.filtershape.size))  # create zero matrix
            shape_knew = knew.shape
            num_elements_kn = shape_knew[0] * shape_knew[1]
            knew = np.reshape(knew, num_elements_kn, order='F')
            shape = btmp.shape
            num_elements = shape[0] * shape[1]
            btmp = np.reshape(btmp, num_elements, order='F')
            shape_bwtd = bwtd.shape
            num_elements_b = shape_bwtd[0] * shape_bwtd[1]
            bwtd = np.reshape(bwtd, num_elements_b, order='F')
            for idx in range(knew.size):
                btmp[int(knew[idx]) - 1] = bwtd[idx]  # fill diag w/ weights
            btmp = np.reshape(btmp, shape, order='F')
            tHK = np.sum(btmp, 1)
            norm_tHK = np.sum(np.abs(tHK))
            tHK = tHK / norm_tHK
            tHK = np.reshape(tHK, (tHK.size, 1))
        else:
            raise Exception('filter size must be greater than 3!')
            tHK = np.zeros(1)
        self.kernel = tHK         # assign weighted kernal

    def __mul__(self, val):
        """
        stack results

        """
        if isinstance(val, np.ndarray):
            return self._transform(val)
        else:
            raise Exception("Input must be an numpy array")

    @property
    def sizeOfFilter(self):
        """
            :rtypr int
            :return: number of points in filter kernal
        """
        return self.filtershape.size

    @property
    def getFilterKernal(self):
        """
           returns the filter kernal
        """
        return self.kernel


class decayKernal(baseKernel):
    """
       Decay kernal for calculating a decay
       from a stack
    """
    # output_type = properties.StringChoice(
    #     "tells kernal the type of output (decay, std)",
    #     default="Vs",
    #     choices=["std", "Vs"]
    # )

    # window_starts = properties.Array(
    #     "array containing window start times",
    #     default=np.zeros(20)
    # )

    # window_ends = properties.Array(
    #     "array containing window end times",
    #     default=np.zeros(1)
    # )

    # window_weight = properties.Float(
    #     "attenuation",
    #     default=401,
    #     required=True
    # )

    # window_overlap = properties.Float(
    #     "overlap in windows",
    #     default=0.45,
    #     required=True
    # )

    def __init__(self, num_windows=None,
                 window_starts=None,
                 window_ends=None,
                 window_weight=None,
                 window_overlap=None,
                 output_type=None, **kwargs):
        baseKernel.__init__(self, None, **kwargs)

        if num_windows is not None:
            self.num_window = int(num_windows)
            self.window_starts = window_starts
            self.window_ends = window_ends
            self.window_weight = window_weight
            self.window_overlap = window_overlap
            self.output_type = output_type
        else:
            raise Exception("need number of windows: num_windows")
        # self.window_overlap = self.window_overlap / self.divsor

    def _transform(self, stack):
        """
        takes in stack data and returns decay
        Input:
        stack = Half period stacked Voltage data

        """
        # calculate weighted window decay data =============
        if isinstance(stack, np.ndarray):
            print(self.output_type)
            starts = self.getWindowStarts()
            # print(starts.size)
            timebase = np.round(
                starts[self.num_window - 1] / 1000.)
            timebase = timebase * 1000
            time = np.arange(0, timebase, (timebase / stack.size))
            vsDecay = np.zeros((self.num_window))
            vs_std = np.zeros((self.num_window))
            # loop through and do every window
            for win in range(self.num_window):
                self.window_starts[win]
                # find how many samples in first window
                cntr = 0
                # get time span for tail of windows
                start_tmp = self.window_starts[win] - (
                    self.window_overlap *
                    (self.window_ends[win] - self.window_starts[win]))
                if win == (vsDecay.size - 1):
                    end_tmp = self.window_ends[win]
                else:
                    end_tmp = self.window_ends[win] + (
                        self.window_overlap *
                        (self.window_ends[win] - self.window_starts[win]))
                # print start_tmp, end_tmp
                for i in range(stack.size):
                    if time[i] >= start_tmp and time[i] <= end_tmp:
                        cntr += 1
                # create window wieghts
                indx1 = np.arange(0, self.window_weight)
                weights = 0.5 - (0.5 *
                                 np.cos((2 * np.pi * indx1) /
                                        (indx1.size - 1)))
                # create new weights
                Wg = np.zeros(cntr)
                start_Wg = (indx1.size / 2.0 - 1.0) - (cntr / 2.0) + 1
                for r in range(Wg.size):
                    Wg[r] = weights[int(start_Wg) + r]
                # print Wg
                # create vector storing weighted values
                Vs_window = np.zeros(cntr)
                Vs_window_ave = np.zeros(cntr)
                # get window times
                w_idx = np.zeros(cntr)
                # assign total time and time step
                count = 0
                for i in range(time.size):
                    if time[i] >= start_tmp and time[i] <= end_tmp:
                        w_idx[count] = time[i]
                        Vs_window[count] = stack[i] * -1 * Wg[count]
                        Vs_window_ave[count] = stack[i] * -1
                        count += 1
                sumWin = np.sum(Vs_window)      # sum the values of the window
                vs_std[win] = np.std(Vs_window_ave)  # standard deviation of window
                # print Vs_window
                vsDecay[win] = sumWin / cntr
        else:
            raise Exception("input must be a stack numpy array!")
            vsDecay = np.zeros(1)
            # end decay =======================================
        output = vsDecay
        # output = vs_std
        if self.output_type == 'std':
            output = vs_std

        return output

    def getWindowStarts(self):
        return self.window_starts

    def getWindowCenters(self):
        """
           returns window centers
        """
        window_centers = (np.asarray(self.window_ends) +
                          np.asarray(self.window_starts)) / 2.
        return window_centers

    def getWindowWidths(self):
        """
           returns window widths
        """
        window_widths = (np.asarray(self.window_ends) -
                         np.asarray(self.window_starts))
        return window_widths


class filterKernal(baseKernel):
    """
        Filter Kernal for stacking a time-series
        raw signal
    """

    def __init__(self, filtershape, **kwargs):
        baseKernel.__init__(self, filtershape, **kwargs)
        # self.createFilterKernal()            # create filter kernal

    def _transform(self, signal):
        """
           performs the stacking calculation
        """
        size_of_stack = int(signal.size / (self.kernel.size))
        Ax = np.reshape(signal, (int(size_of_stack),
                        int(self.kernel.size)), order='F')
        shape_Ax = Ax.shape
        shape_tHK = self.kernel.shape
        if shape_Ax[1] == shape_tHK[0]:
            stack = np.matmul(Ax, self.kernel)  # create stack data
            return stack
        else:
            return 0


class ensembleKernal(baseKernel):
    """
    ensemble stacking kernel

    """

    def __init__(self, filtershape,
                 number_half_periods, **kwargs):
        filter_kernal = createHanningWindow(filtershape.size)   # mod for tappered overlap
        baseKernel.__init__(self, filter_kernal, **kwargs)          # create the base kernal
        self.number_half_periods = number_half_periods              # number of half t in signal
        self.kernal_ends = self.createFilterKernalEnds(filtershape.size)  # create end kernal

    def createFilterKernalEnds(self, ensemble_size):
        """
        creates the filter kernal

        """
        ends_window = createHanningWindow(ensemble_size - 2)
        print(ends_window.size)
        if ends_window.size > 1:
            # create the filter kernal
            tkernal = np.ones((3, 1))
            tkernal[1] = -2.0                    # 3 point kernal
            bsgn = np.ones((1, ends_window.size))
            bsgn[0, 1::2] = bsgn[0, 1::2] * -1   # alternate pol rem lin drift
            bwtd = np.matmul(tkernal,
                             bsgn * ends_window)  # filter weights
            tmp1 = np.arange(1, 4)
            tmp1 = np.reshape(tmp1, (3, 1))
            tmp2 = np.ones((1, ends_window.size))
            tmp3 = np.ones((3, 1))
            tmp4 = np.arange(ends_window.size)
            tmp4 = np.reshape(tmp4, (1, ends_window.size))
            knew = (np.matmul(tmp1, tmp2) +
                    np.matmul(tmp3, (tmp4 * (ends_window.size + 3))))
            btmp = np.zeros((ends_window.size + 2,
                             ends_window.size))  # create zero matrix
            shape_knew = knew.shape
            num_elements_kn = shape_knew[0] * shape_knew[1]
            knew = np.reshape(knew, num_elements_kn, order='F')
            shape = btmp.shape
            num_elements = shape[0] * shape[1]
            btmp = np.reshape(btmp, num_elements, order='F')
            shape_bwtd = bwtd.shape
            num_elements_b = shape_bwtd[0] * shape_bwtd[1]
            bwtd = np.reshape(bwtd, num_elements_b, order='F')
            for idx in range(knew.size):
                btmp[int(knew[idx]) - 1] = bwtd[idx]  # fill diag w/ weights
            btmp = np.reshape(btmp, shape, order='F')
            tHK = np.sum(btmp, 1)
            norm_tHK = np.sum(np.abs(tHK))
            tHK = tHK / norm_tHK
            tHK = np.reshape(tHK, (tHK.size, 1))
        else:
            raise Exception('filter size must be greater than 3!')
            tHK = np.zeros(1)
        return tHK

    def _transform(self, signal):
        """
           performs the stacking calculation using Ensembles
        """
        sub_signals = []
        sub_samples = []
        size_of_stack = int(signal.size / (self.number_half_periods))
        overlap = 2                                                        # hard code standard
        T_per_ensemble = (self.kernel.size - overlap)                 # desired half T
        new_signal_size = int((T_per_ensemble) * size_of_stack)            # size of the signal going in
        samples_per_ens = int((T_per_ensemble + overlap) * size_of_stack)
        number_of_ensembles = signal.size / (T_per_ensemble * size_of_stack)
        opt_number_of_samples = number_of_ensembles * T_per_ensemble * size_of_stack
        print(
            "num of ensembles: {0} & opt num: {1} & size of org sig: {2}".format(
                number_of_ensembles, T_per_ensemble, self.kernal_ends.size
            )
        )
        signal = signal[:opt_number_of_samples]
        ensembles = np.zeros((size_of_stack, number_of_ensembles))         # matrix holding all the stacks
        for index in range(number_of_ensembles):
            if index == 0:
                T_overlap = T_per_ensemble                                # get end overlap
                end_index = T_overlap * size_of_stack
                trim_signal = signal[:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(0, end_index))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernal_ends.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernal_ends.shape
                if shape_Ax[1] == shape_tHK[0]:
                    stack = np.matmul(Ax, self.kernal_ends)  # create stack data
                    ensembles[:, index] = stack.T
                else:
                    print("fail stack, wrong size")
            elif index == 1:
                T_overlap = T_per_ensemble + overlap
                start_index = size_of_stack * (T_per_ensemble - 2)
                end_index = start_index + (T_overlap * size_of_stack)
                trim_signal = signal[start_index:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, end_index))
                print(
                    "num of ensembles: {0} & opt num: {1} & size of org sig: {2}".format(
                        number_of_ensembles, T_per_ensemble, self.kernel.size
                    )
                )
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape
                if shape_Ax[1] == shape_tHK[0]:
                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T
                else:
                    print("fail stack, wrong size")
            elif index == (number_of_ensembles - 1):
                T_overlap = T_per_ensemble + overlap                   # get end overlap
                start_index = (index * (T_per_ensemble) - 2) * size_of_stack
                trim_signal = signal[start_index:]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, signal.size))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape
                print("size of last: {0}, with overlap: {1}".format(start_index, signal.size))
                if shape_Ax[1] == shape_tHK[0]:
                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T
                else:
                    print("fail stack, wrong size in last")
            else:
                T_overlap = T_per_ensemble + overlap
                start_index = (((index) * T_per_ensemble) - 2) * size_of_stack
                end_index = start_index + (T_overlap * size_of_stack)
                trim_signal = signal[start_index:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, end_index))
                print("size of first: {0}, with overlap: {1}".format(start_index, signal.size))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape
                if shape_Ax[1] == shape_tHK[0]:
                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T
                else:
                    print("fail stack, wrong size")
        ensembles[:, ::2] = ensembles[:, ::2] * -1   # make all stacks same polarity
        return ensembles, sub_signals, sub_samples


##################################################
# define methods
def getFrequnceyResponse(signal):
    """
       :rtype numpy array
       :return: frequeny response of the filter kernal
    """
    v_fft = fftpack.fft(signal)
    amplitude = np.sqrt(v_fft.real**2 + v_fft.imag**2)
    return amplitude[0:(amplitude.size / 2 - 1)] / np.max(amplitude)


def getPhaseResponse(signal):
    """
       :rtype numpy array
       :return: Rhase response of the filter kernal
    """
    v_fft = fftpack.fft(signal)
    phase = np.arctan2(v_fft.imag, v_fft.real)
    return phase[0:(phase.size / 2 - 1)]


def getColeCole(mx_decay,
                init_cond,
                init_tau,
                step_factor,
                window_widths,
                window_weights):
    """
    takes in a decay and returns best fitting cole-cole
    note:
    A tool to calculate the time - domain voltage response[V(t) / Vo] for a
    homogenous Cole - Cole model of the earth using the digital linear filter
    formulation given by Guptasarma(Geophys, vol 47, pg 1575, 1982)
    """
    time = np.zeros(window_widths.size)  # initiates time
    # convert window widths to accumlative times specific to algorithm
    for i in range(time.size):
        time[i] = (np.sum(window_widths[0:i + 1]) / 2.0) / 1000.0
    c = np.zeros(12)                      # conductivity array
    v_cole = np.zeros(window_widths.size)  # best fit cole-cole
    tau = np.zeros(12)                    # time constant array
    err = np.zeros((12, 12))               # error matrix
    cole_m = np.zeros((12, 12))            # matrix of chargeabilities
    radius = step_factor                 # radius of array fill
    c[7] = init_cond                     # center of cond. array
    tau[7] = init_tau                    # center of tau array
    tau10 = np.log10(tau[7])             # log of time constant
    idx = np.arange(0, 12)
    c = c[7] + radius * (idx - 7) / 40.0  # fill cond. array
    tau = np.power(10.0,
                   (tau10 + radius * (idx - 7) /
                    2.))  # fill tau array
    # create filter
    areg = np.asarray([-3.82704, -3.56608, -3.30512, -3.04416,
                       -2.78320, -2.52224, -2.26128, -2.00032,
                       -1.73936, -1.47840, -1.21744, -.95648,
                       -.69552, -0.43456, -0.17360, 0.08736,
                       0.34832, 0.60928, 0.87024, 1.13120, 1.39216])
    # create 2nd filter
    preg = np.asarray([0.000349998, -0.000418371, 0.000772828,
                      -0.000171356, 0.001022172, 0.000897638,
                      0.002208974, 0.003844944, 0.006809040,
                      0.013029162, 0.022661391, 0.042972904,
                      0.075423603, 0.139346367, 0.234486236,
                      0.366178323, 0.284615486, -0.235691746,
                      0.046994188, -0.005901946, 0.000570165])
    v_cole = np.zeros(time.size)      # initiate decay array
    minErr = 0.5                      # signify initial Low error
    c_idx = 0                         # index of cond. of min err
    tau_idx = 0                       # index of cond. of min err

    # loop through the arrays of cond. and tau
    for i in range(c.size):
        for j in range(tau.size):
            ax = c[i] * np.pi / 2.0
            for win in range(mx_decay.size):
                v_temp = 0.0
                for n in range(areg.size):
                    w = np.power(10.0, (areg[n] - np.log10(time[win])))
                    ex = (w * tau[j])**c[i]
                    y = np.complex(ex * np.cos(ax), ex * np.sin(ax))
                    z = 1.0 - 1.0 / (1.0 + y)
                    # print(ex * np.cos(ax), np.real(z))
                    v_temp = v_temp + preg[n] * np.real(z)
                v_cole[win] = v_temp

            # calculate error
            norm_weights = np.sum(window_weights) / window_weights.size
            # serr = (np.sum(np.power((mx_decay - v_cole), 2) *
            #         window_weights) / norm_weights)
            err = np.sqrt(np.sum(np.power(np.abs(mx_decay - v_cole) *
                          window_weights, 2)) / window_weights.size)
            # np.sqrt(serr / (window_weights.size))
            if err < minErr:
                c_idx = i
                tau_idx = j
                minErr = (err)

            mx_conversion = 1000.                 # conversion for mV/V
            cole_m[i, j] = (np.sum(v_cole * window_widths) /
                            np.sum(window_widths)) * mx_conversion  # calcs Mx

            # go back and calculate best fit cole-cole curve and save it
            for win in range(mx_decay.size):
                v_temp = 0.0
                for n in range(areg.size):
                    w = np.power(10.0, (areg[n] - np.log10(time[win])))
                    ex = np.power(w * tau[tau_idx], c[c_idx])
                    y = np.complex(ex * np.cos(ax), ex * np.sin(ax))
                    z = 1.0 - 1.0 / (1.0 + y)
                    v_temp = v_temp + preg[n] * np.real(z)
                v_cole[win] = v_temp

    # calculate the percent diff
    # percent_diff = (
    #     np.mean((np.abs((v_cole - mx_decay) /
    #             ((v_cole + mx_decay) / 2.))) * window_weights))

    return c[c_idx], tau[tau_idx], cole_m[c_idx, tau_idx], minErr, v_cole


def getBestFitColeCole(mx_decay,
                       init_cond,
                       init_tau,
                       step_factor,
                       window_widths,
                       iterations,
                       timeCenter,
                       window_weights):
    c = init_cond
    tau = init_tau
    r = step_factor
    stored_error = 0
    min_iter = 3                               # min amount of iterations
    for iter in range(iterations):
        c, tau, M, error, vs = getColeCole(mx_decay,
                                           c,
                                           tau,
                                           r,
                                           timeCenter,
                                           window_weights)
        delta_error = abs(stored_error - error) / ((stored_error + error) / 2.)
        print("iter: %i | c: %f | tau: %f | M: %f | error: %f | delta: %f" %
              (iter, c, tau, M, error, delta_error))
        stored_error = error
        # r = r / 2.
        if delta_error > 0.002 or iter < min_iter:
            r = r / 2.
        elif delta_error < 0.002:
            print("convergence accomplished! DONE")
            break



def getWeightedVs(stack, window_start, window_end, attenuation):
    """
    takes in data and returns decay
    Input:
    stack = Half period stacked Voltage data
    window_start = an array of window start times
    window_end = an array of window end times

    """
    timebase = np.round(window_end[len(window_end) - 1] / 1000.0)
    timebase = timebase * 1000
    time = np.arange(0, timebase, (timebase / stack.size))
    vsDecay = np.zeros((len(window_end)))
    # loop through and do every window
    for win in range(len(window_start)):
        # find how many samples in first window
        cntr = 0
        # get time span for tail of windows
        start_tmp = window_start[win] - (
            0.45 * (window_end[win] - window_start[win]))
        if win == (vsDecay.size - 1):
            end_tmp = window_end[win]
        else:
            end_tmp = window_end[win] + (
                0.45 * (window_end[win] - window_start[win]))
        # print start_tmp, end_tmp
        for i in range(stack.size):
            if time[i] >= start_tmp and time[i] <= end_tmp:
                cntr += 1
        # create window wieghts
        indx1 = np.arange(0, attenuation)
        weights = 0.5 - (0.5 * np.cos((2 * np.pi * indx1) / (indx1.size - 1)))
        # create new weights
        Wg = np.zeros(cntr)
        start_Wg = (indx1.size / 2.0 - 1.0) - (cntr / 2.0) + 1
        for r in range(Wg.size):
            Wg[r] = weights[int(start_Wg) + r]
        # print Wg
        # create vector storing weighted values
        Vs_window = np.zeros(cntr)
        Vs_window_ave = np.zeros(cntr)
        # get window times
        w_idx = np.zeros(cntr)
        # assign total time and time step
        count = 0
        for i in range(time.size):
            if time[i] >= start_tmp and time[i] <= end_tmp:
                w_idx[count] = time[i]
                Vs_window[count] = stack[i] * -1 * Wg[count]
                Vs_window_ave[count] = stack[i] * -1
                count += 1
        sumWin = np.sum(Vs_window)
        # print Vs_window
        vsDecay[win] = sumWin / cntr

    return vsDecay


def createBruteStackWindow(num_points):
    num_points = num_points - 6
    tmp = np.ones(num_points)
    tmp = tmp * 4

    # create full filter kernal
    f1 = np.zeros(tmp.size + 4)
    f1[0] = 1
    f1[1] = 3
    f1[f1.size - 2] = 3
    f1[f1.size - 1] = 1

    for j in range(tmp.size):
        f1[j + 2] = tmp[j]

    f1 = f1 / (4.0 * (num_points - 2))

    return f1


def createKaiserWindow(num_taps, attenuation):
    """
    creates a Kaiser window
    Input:
    num_taps = number of taps for the requested window

    """
    x = factorial(2)
    return x


def mbessel(position, max_iter):
    seq_m = np.arange(0, max_iter)
    fact_m = factorial(seq_m, exact=False)
    summation = 0.0
    for i in range(max_iter):
        inc = np.power(1.0 / fact_m * np.power(position * 0.5, i), 2)
        frac = inc / summation
        summation += inc
        if frac < 0.001:
            break
    return summation


def cheby_poly(n, x):
    pos = 0.0
    if np.abs(x) <= 1.0:
        pos = np.cos(n * np.arccos(x))
    else:
        pos = np.cosh(n * np.arccosh(x))
    return pos


def createHanningWindow(num_points):
    """
    creates a Hanning window filter kernal
    Input:
    num_points = number of taps for the requested window

    """
    # num_points = num_points - 2
    indx1 = np.arange(1, num_points + 1).T          # create  sequence array
    filterWindow = 0.5 * (1 - np.cos((2 * np.pi /
                          (indx1.size - 1)) * indx1))  # creates window

    return filterWindow


def createChebyshevWindow(num_taps, attenuation):
    """
    creates a Chebyshev window
    Input:
    num_taps = number of taps for the requested window

    """
    weights = np.zeros(num_taps)
    summation = 0.0
    max_value = 0.0
    tg = np.power(10, (attenuation / 20))
    x0 = np.cosh(1.0 / (num_taps - 1)) * np.arccosh(tg)
    M = (num_taps - 1) / 2
    if (num_taps % 2) == 0:
        M = M + 0.5
    for nn in range(num_taps / 2 + 1):
        n = nn - M
        summation = 0.0
        for i in range(int(M)):
            summation += cheby_poly(num_taps - 1.0,
                                    x0 * np.cos((2.0 *
                                                n * np.pi * i) / num_taps))
        weights[nn] = tg + 2 * summation
        weights[num_taps - nn - 1] = weights[nn]
        if weights[nn] > max_value:
            max_value = weights[nn]
    for ii in range(num_taps):
        weights[ii] /= max_value
    return weights


def getPrimaryVoltage(start, end, stack):
        """
        Extracts the Vp of the signal.
        Input:
        start = percent of the on time to start calculation
        end = percent of on time to end calculation
        e.g 50% to 90%
        """
        sumStart = int((start / 100.0) * (stack.size / 2))  # start Vp calc
        sumEnd = int((end / 100.0) * (stack.size / 2))     # end of Vp calc
        Vp = np.sum(stack[sumStart:sumEnd]) / (sumEnd - sumStart)

        return Vp


def loadDias(fileName):
    """
    Function for loading a dias file and returns a
    "Patch" class complete with sources and recievers

    Input:
    fileName = complete path to data file

    """
    lines = 0
    text_file = open(fileName, "r")

    # determin how many lines in the file
    while text_file.readline():
            lines += 1
    text_file.close()

    # initiate a patch
    patch = Jpatch()
    # read header information
    text_file = open(fileName, "r")
    # initiate reading control variables
    currRdg = 0
    for i, line in enumerate(text_file):
        if i == 4:
            Varinfo = line.split()
            header4 = line
            # print(Varinfo)
        elif i == 0:
                header1 = line
        elif i == 1:
                header2 = line
                id_info = line.split()
                patch.assignPatchID(id_info[1])
        elif i == 2:
                header3 = line
        elif i > 3:
            # try:
                    datatxt = line.split()
                    # do some Jdatamanagment stuff
                    varFields = Jreadtxtline(Varinfo, datatxt)
                    # verify if line is a new reading
                    if varFields.RDG == currRdg:
                        # add the dipoles
                        # Idp = Jdata.JinDipole(varFields)
                        Vpdp = JvoltDipole(varFields)
                        Rdg.addVoltageDipole(Vpdp)
                    else:
                        # create a reading
                        Rdg = Jreading(varFields.RDG)
                        Idp = JinDipole(varFields)
                        Vpdp = JvoltDipole(varFields)
                        Rdg.addVoltageDipole(Vpdp)
                        Rdg.addInDipole(Idp)
                        # add reading to the patch
                        patch.addreading(Rdg)
                        currRdg = varFields.RDG
            # except:
                    # pass

    text_file.close()
    headers = [header1, header2, header3, header4]
    patch.assignHeaderInfo(headers)
    return patch

# ===================================================
# Dias Data specific class
class JinDipole:
    """
    Class containing Source information

    Initiate with a structure containing location
    and Current value + error

    """

    def __init__(self, InDpInfo):
        self.Tx1East = float(InDpInfo.Tx1East)
        self.Tx1North = float(InDpInfo.Tx1North)
        self.Tx1Elev = float(InDpInfo.Tx1Elev)
        self.Tx2East = float(InDpInfo.Tx2East)
        self.Tx2North = float(InDpInfo.Tx2North)
        self.Tx2Elev = float(InDpInfo.Tx2Elev)
        self.Tx1x = float(InDpInfo.Tx1x)
        self.Tx1y = float(InDpInfo.Tx1y)
        self.Tx2x = float(InDpInfo.Tx2x)
        self.Tx2y = float(InDpInfo.Tx2y)
        self.In = np.abs(float(InDpInfo.In))
        self.In_err = float(InDpInfo.In_err)

    def getTxStack(self, stack_dir):
        rec_num = self.reading
        bin_file = stack_dir + "P1_R"+ rec_num + "_TX.raw"

        f = open(bin_file, "rb")          # open the file
        lines = f.read()                  # read data as string (can modify this to get header info
        return_char = 0.                  # beginning of binary data idex
        for idx in range(100):
            if lines[idx] == "\r":
                return_char = idx
        bin_start = return_char + 2       # increment 2 spaces to start of binary data
        data = []                         # initiate array for data
        with open(bin_file, 'rb') as f:   # open file for data extraction
            f.seek(bin_start,
                   os.SEEK_SET)          # seek to beginning of binary data
            while True:
                b = f.read(8)             # read 8 bytes at a time
                if not b:                 # break out if end of file
                    # eof
                    break
                data.append(
                    struct.unpack('d',
                                  b))  # store data

        return data


class JvoltDipole:
    """
    object containing voltage information

    """

    def __init__(self, VoltDpinfo):
        self.dipole = VoltDpinfo.DIPOLE
        self.reading = VoltDpinfo.RDG
        self.Rx1x = float(VoltDpinfo.Rx1x)
        self.Rx1y = float(VoltDpinfo.Rx1y)
        self.Rx2x = float(VoltDpinfo.Rx2x)
        self.Rx2y = float(VoltDpinfo.Rx2y)
        self.Rx1File = VoltDpinfo.Rx1File
        self.Rx1East = float(VoltDpinfo.Rx1East)
        self.Rx1North = float(VoltDpinfo.Rx1North)
        self.Rx1Elev = float(VoltDpinfo.Rx1Elev)
        self.Rx2File = VoltDpinfo.Rx2File
        self.Rx2East = float(VoltDpinfo.Rx2East)
        self.Rx2North = float(VoltDpinfo.Rx2North)
        self.Rx2Elev = float(VoltDpinfo.Rx2Elev)
        self.K = float(VoltDpinfo.k)
        try:
            self.Vp = float(VoltDpinfo.Vp)
        except:
            self.Vp = -999.9

        self.Vp_err = float(VoltDpinfo.Vp_err)
        self.Rho = float(VoltDpinfo.Rho)
        self.flagRho = VoltDpinfo.Rho_QC
        self.Stack = float(VoltDpinfo.Stack)
        try:
            self.Mx = float(VoltDpinfo.Mx)
        except:
            self.Mx = -99.9
        self.Mx_err = float(VoltDpinfo.Mx_err)
        self.flagMx = VoltDpinfo.Mx_QC
        self.flagBad = VoltDpinfo.Status
        self.TimeBase = VoltDpinfo.TimeBase
        self.Vs = np.asarray(VoltDpinfo.Vs)

    def getDipoleStack(self, stack_dir):
        node1_id = self.Rx1File[:2]
        node2_id = self.Rx2File[:2]
        rec_num = self.reading
        bin_file = stack_dir + "P1_R"+ rec_num + "_" + node1_id + "_" + node2_id + ".stk"

        f = open(bin_file, "rb")          # open the file
        lines = f.read()                  # read data as string (can modify this to get header info
        return_char = 0.                  # beginning of binary data idex
        for idx in range(100):
            if lines[idx] == "\r":
                return_char = idx
        bin_start = return_char + 2       # increment 2 spaces to start of binary data
        data = []                         # initiate array for data
        with open(bin_file, 'rb') as f:   # open file for data extraction
            f.seek(bin_start,
                   os.SEEK_SET)           # seek to beginning of binary data
            while True:
                b = f.read(8)             # read 8 bytes at a time
                if not b:                 # break out if end of file
                    # eof
                    break
                data.append(
                    struct.unpack('d',
                                  b))  # store data
                # if len(b) < 8:
                #     break
        return data

    def getXplotpoint(self, Idp):
        if (self.Rx1x > Idp.Tx1x):
            x = Idp.Tx1x + ((self.Rx1x - Idp.Tx1x) / 2.0)
        elif (self.Rx1x < Idp.Tx1x):
            x = Idp.Tx1x - ((Idp.Tx1x - self.Rx1x) / 2.0)
        return x

    def getYplotpoint(self, Idp):
        if (self.Rx1y > Idp.Tx1y):
            y = Idp.Tx1y + ((self.Rx1y - Idp.Tx1y) / 2.0)
        elif (self.Rx1y < Idp.Tx1y):
            y = Idp.Tx1y - ((Idp.Tx1y - self.Rx1y) / 2.0)
        return y

    def getZplotpoint(self, Idp):
        r = np.sqrt((Idp.Tx1x - self.Rx2x)**2 +
                    (Idp.Tx1y - self.Rx2y)**2 +
                    (Idp.Tx1Elev - self.Rx2Elev)**2)
        # z = -(abs(Idp.Tx1 - self.Rx1)) / 2.0
        z = -(r / 3.)
        return z

    def calcGeoFactor(self, Idp):
        r1 = ((self.Rx1East - Idp.Tx1East)**2 +
              (self.Rx1North - Idp.Tx1North)**2 +
              (self.Rx1Elev - Idp.Tx1Elev)**2)**0.5
        r2 = ((self.Rx2East - Idp.Tx1East)**2 +
              (self.Rx2North - Idp.Tx1North)**2 +
              (self.Rx2Elev - Idp.Tx1Elev)**2)**0.5
        r3 = ((self.Rx1East - Idp.Tx2East)**2 +
              (self.Rx1North - Idp.Tx2North)**2 +
              (self.Rx1Elev - Idp.Tx2Elev)**2)**0.5
        r4 = ((self.Rx2East - Idp.Tx2East)**2 +
              (self.Rx2North - Idp.Tx2North)**2 +
              (self.Rx2Elev - Idp.Tx2Elev)**2)**0.5
        gf = 1 / ((1 / r1 - 1 / r2) - (1 / r3 - 1 / r4))
        return 2 * np.pi * gf

    def calcRho(self, Idp):
        r1 = ((self.Rx1East - Idp.Tx1East)**2 +
              (self.Rx1North - Idp.Tx1North)**2 +
              (self.Rx1Elev - Idp.Tx1Elev)**2)**0.5
        r2 = ((self.Rx2East - Idp.Tx1East)**2 +
              (self.Rx2North - Idp.Tx1North)**2 +
              (self.Rx2Elev - Idp.Tx1Elev)**2)**0.5
        r3 = ((self.Rx1East - Idp.Tx2East)**2 +
              (self.Rx1North - Idp.Tx2North)**2 +
              (self.Rx1Elev - Idp.Tx2Elev)**2)**0.5
        r4 = ((self.Rx2East - Idp.Tx2East)**2 +
              (self.Rx2North - Idp.Tx2North)**2 +
              (self.Rx2Elev - Idp.Tx2Elev)**2)**0.5
        gf = 1 / ((1 / r1 - 1 / r2) - (1 / r3 - 1 / r4))
        Vp = np.abs(self.Vp)
        if gf < 0:
            Vp = Vp * -1
        # print("Vp: {0}".format(self.Vp))
        rho = (Vp / Idp.In) * 2 * np.pi * gf
        self.Rho = rho
        return rho


class Jreading:
    """
    Class to handle current and voltage dipole
    information for a given source

    """
    def __init__(self, mem):
        self.MemNumber = mem
        self.Vdp = []
    # method for adding voltage dipoles

    def addVoltageDipole(self, JVdp):
        self.Vdp.append(JVdp)

    # method for assigning Current dipole
    def addInDipole(self, JIdp):
        self.Idp = JIdp


class Jpatch:
    """
    Class to hold source information for a given data patch

    """

    def __init__(self):
        self.readings = []

    def addreading(self, Jrdg):
        self.readings.append(Jrdg)

    def assignPatchID(self, id):
        self.ID = id

    def assignHeaderInfo(self, headerLines):
        """
        Method for processing the header lines
        of a Dias data file. (e.g IP times, project name, etc.)

        Input: an array of header lines from file

        """

        self.headers = headerLines         # assigns headers to patch class
        # process IP times from header
        timing_string = self.headers[2].split(' ')[1]
        timings = timing_string.split(';')[0]
        timing = timings.split(',')
        self.window_start = np.zeros(len(timing))
        self.window_end = np.zeros(len(timing))
        self.window_center = np.zeros(len(timing))
        self.window_width = np.zeros(len(timing))
        # loops each available IP window and calculate width and centre
        for i in range(len(timing)):
            wintime = timing[i].split(':')
            self.window_start[i] = float(wintime[0])
            self.window_end[i] = float(wintime[1])
            self.window_center[i] = (self.window_start[i] +
                                     self.window_end[i]) / 2.0
            self.window_width[i] = (self.window_end[i] -
                                    self.window_start[i])

    def getApparentResistivity(self):
        """
        Exports all the apparent resistivity data

        Output:
        numpy array [value]

        """
        resistivity_list = []
        num_rdg = len(self.readings)
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            for j in range(num_dipole):
                if self.readings[k].Vdp[j].flagRho == "Accept":
                    rho_a = self.readings[k].Vdp[j].calcRho(self.readings[k].Idp)
                    resistivity_list.append(rho_a)
        return np.asarray(resistivity_list)

    def getGeometricFactor(self):
        """
        Exports all the geometry factor data

        Output:
        numpy array [value]

        """
        k_list = []
        num_rdg = len(self.readings)
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            for j in range(num_dipole):
                if (self.readings[k].Vdp[j].flagRho == "Accept"):
                    # k_a = self.readings[k].Vdp[j].K
                    k_a = self.readings[k].Vdp[j].calcGeoFactor(self.readings[k].Idp)
                    k_list.append(1 / k_a)
        return np.asarray(k_list)

    def getVoltages(self):
        """
        Exports all the geometry factor data

        Output:
        numpy array [value]

        """
        k_list = []
        num_rdg = len(self.readings)
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            for j in range(num_dipole):
                if self.readings[k].Vdp[j].flagRho == "Accept":
                    k_a = self.readings[k].Vdp[j].Vp
                    k_list.append(k_a)
        return np.asarray(k_list)

    def getApparentChageability(self):
        """
        Exports all the apparent chargeability data

        Output:
        numpy array [value]

        """
        chargeability_list = []
        num_rdg = len(self.readings)
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            for j in range(num_dipole):
                mx_a = self.readings[k].Vdp[j].Mx
                chargeability_list.append(mx_a)
        return np.asarray(chargeability_list)

    def getSources(self, dipole=False):
        """
        Exports all the tx locations to a numpy array

        Output:
        numpy array [x0,y0,z0,x1,y1,z1] (c1,c2)

        """

        src_list = []
        num_rdg = len(self.readings)
        if not dipole:
            for k in range(num_rdg):
                tx = np.array([self.readings[k].Idp.Tx1East,
                              self.readings[k].Idp.Tx1North,
                              self.readings[k].Idp.Tx1Elev,
                              self.readings[k].Idp.Tx2East,
                              self.readings[k].Idp.Tx2North,
                              self.readings[k].Idp.Tx2Elev])
                src_list.append(tx)
        else:
            for k in range(num_rdg):
                num_dipole = len(self.readings[k].Vdp)
                for j in range(num_dipole):
                    tx = np.array([self.readings[k].Idp.Tx1East,
                                  self.readings[k].Idp.Tx1North,
                                  self.readings[k].Idp.Tx1Elev,
                                  self.readings[k].Idp.Tx2East,
                                  self.readings[k].Idp.Tx2North,
                                  self.readings[k].Idp.Tx2Elev])
                    src_list.append(tx)
        # number_of_src = len(src_list)
        # src = np.zeros((number_of_src, 6))
        # for tx in range(number_of_src):
        #     src[tx, :] = src_list[tx]
        return np.asarray(src_list)

    def getSources2(self, dipole=False):
        """
        Exports all the tx locations to a numpy array

        Output:
        numpy array [x0,y0,z0,x1,y1,z1] (c1,c2)

        """

        src_list = []
        num_rdg = len(self.readings)
        if not dipole:
            for k in range(num_rdg):
                tx = np.array([self.readings[k].Idp.Tx1East,
                              self.readings[k].Idp.Tx1North,
                              self.readings[k].Idp.Tx1Elev])
                src_list.append(tx)
        else:
            for k in range(num_rdg):
                num_dipole = len(self.readings[k].Vdp)
                for j in range(num_dipole):
                    tx = np.array([self.readings[k].Idp.Tx1East,
                                  self.readings[k].Idp.Tx1North,
                                  self.readings[k].Idp.Tx1Elev])
                    src_list.append(tx)
        # number_of_src = len(src_list)
        # src = np.zeros((number_of_src, 6))
        # for tx in range(number_of_src):
        #     src[tx, :] = src_list[tx]
        return np.asarray(src_list)

    def getDipoles(self):
        """
        Exports all the tx locations to a numpy array

        Output:
        numpy array [x0,y0,z0,x1,y1,z1] (p1,p2)

        """

        dipole_list = []
        num_rdg = len(self.readings)
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            for j in range(num_dipole):
                rx = np.array([self.readings[k].Vdp[j].Rx1East,
                              self.readings[k].Vdp[j].Rx1North,
                              self.readings[k].Vdp[j].Rx1Elev,
                              self.readings[k].Vdp[j].Rx2East,
                              self.readings[k].Vdp[j].Rx2North,
                              self.readings[k].Vdp[j].Rx2Elev])
                dipole_list.append(rx)
        number_of_dipoles = len(dipole_list)
        dipoles = np.zeros((number_of_dipoles, 6))
        for rx in range(number_of_dipoles):
            dipoles[rx, :] = dipole_list[rx]
        return dipoles

    def createDcSurvey(self, data_type, ip_type=None):
        """
        Loads a dias data file to a SimPEG "srcList" class

        Input:
        datatype = Choose either IP or DC

        Note: elevation is +ve for simPEG inversion

        """
        doff = 0                                      # in case offset is require
        srcLists = []                                 # Injections + dipoles
        data = []                                     # data from file
        d_weights = []                                # weights for the data
        num_rdg = len(self.readings)
        minE = self.readings[0].Idp.Tx2East
        minN = self.readings[0].Idp.Tx2North
        maxN = minN
        maxE = minE
        for k in range(num_rdg):
            num_dipole = len(self.readings[k].Vdp)
            num_dipole_count = 0
            for i in range(num_dipole):
                if data_type == "DC":
                    if (self.readings[k].Vdp[i].flagRho == "Accept"):
                        num_dipole_count += 1
                if data_type == "IP":
                    if self.readings[k].Vdp[i].flagMx == "Accept":
                        num_dipole_count += 1
            rx = np.zeros((num_dipole_count, 6))
            tx = np.array([self.readings[k].Idp.Tx1East,
                          self.readings[k].Idp.Tx1North,
                          self.readings[k].Idp.Tx1Elev - doff,
                          # 0,
                          self.readings[k].Idp.Tx2East,
                          self.readings[k].Idp.Tx2North,
                          self.readings[k].Idp.Tx2Elev - doff])
                          # 0])
            if self.readings[k].Idp.Tx1East > maxE:
                maxE = self.readings[k].Idp.Tx1East
            if self.readings[k].Idp.Tx1East < minE:
                minE = self.readings[k].Idp.Tx1East
            if self.readings[k].Idp.Tx1North > maxN:
                maxN = self.readings[k].Idp.Tx1North
            if self.readings[k].Idp.Tx1North < minN:
                minN = self.readings[k].Idp.Tx1North
            cnt = 0
            for i in range(num_dipole):
                if data_type == "DC":
                    if (self.readings[k].Vdp[i].flagRho == "Accept"):
                        rx[cnt, :] = [self.readings[k].Vdp[i].Rx1East,
                                      self.readings[k].Vdp[i].Rx1North,
                                      self.readings[k].Vdp[i].Rx1Elev - doff,
                                      # 0,
                                      self.readings[k].Vdp[i].Rx2East,
                                      self.readings[k].Vdp[i].Rx2North,
                                      self.readings[k].Vdp[i].Rx2Elev - doff]
                                      # 0]
                        Vp = np.abs(self.readings[k].Vdp[i].Vp)
                        if self.readings[k].Vdp[i].K < 0:
                            Vp = Vp * -1
                        data.append(Vp / self.readings[k].Idp.In)
                        d_weights.append((self.readings[k].Vdp[i].Vp_err +
                                          self.readings[k].Idp.In_err) / 100.)
                        cnt += 1
                if data_type == "IP":
                    if self.readings[k].Vdp[i].flagMx == "Accept":
                        rx[cnt, :] = [self.readings[k].Vdp[i].Rx1East,
                                      self.readings[k].Vdp[i].Rx1North,
                                      self.readings[k].Vdp[i].Rx1Elev - doff,
                                      self.readings[k].Vdp[i].Rx2East,
                                      self.readings[k].Vdp[i].Rx2North,
                                      self.readings[k].Vdp[i].Rx2Elev - doff]
                        if ip_type is None:
                            data.append(self.readings[k].Vdp[i].Mx)
                        elif ip_type == "decay":
                            Vs = np.abs(self.readings[k].Vdp[i].Vs)
                            if self.readings[k].Vdp[i].K < 0:
                                Vs = Vs * -1
                            data.append(self.readings[k].Vdp[i].Vs /
                                        self.readings[k].Idp.In)
                        else:
                            Vp = self.Vp
                            if self.readings[k].K < 0:
                                Vp = Vp * -1
                            data.append(self.readings[k].Vdp[i].Mx *
                                       ((Vp / 1e3) / self.readings[k].Idp.In))

                        d_weights.append((self.readings[k].Vdp[i].Mx *
                                         (self.readings[k].Vdp[i].Mx_err /
                                          100.0)))
                        cnt += 1

            Rx = DC.Rx.Dipole(rx[:, :3], rx[:, 3:])    # create dipole list
            # srcLists.append(DC.Src.Pole([Rx], tx[:3]))
            srcLists.append(DC.Src.Dipole([Rx], tx[:3], tx[3:]))

        survey = DC.SurveyDC.Survey(srcLists)          # creates the survey
        survey.dobs = np.float64(np.asarray(data))                 # assigns data
        survey.std = np.asarray(d_weights)             # assign data weights
        survey.eps = 0.001

        return survey


class Jreadtxtline:
    """
    Class specifically for reading a line of text from a dias file

    """

    def __init__(self, hdrLine, dataLine):
        # make a structure with the header inputs
        self.Vs = []
        for n in range(len(hdrLine)):
            if hdrLine[n].find("Vs") == 0:
                self.Vs.append(float(dataLine[n]))
            else:
                setattr(self, hdrLine[n], dataLine[n])
