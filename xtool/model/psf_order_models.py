from collections import OrderedDict

import numpy as np
import numexpr as ne
from astropy import modeling
from astropy import units as u

from scipy import sparse

from xtool.model.base import SIGMA_TO_FWHM, FWHM_TO_SIGMA



class LinearLeastSquaredModel(modeling.Model):

    def _initialize_lls_model(self, pixel_table, wavelength_pixels):
        self.pixel_table = pixel_table.copy()
        self.standard_broadcasting = False
        self.matrix_parameter_slices = self._generate_matrix_parameter_slices()
        self.wavelength_pixels = wavelength_pixels

    def _generate_matrix_parameter_slices(self):
        """
        Generate slices for the different matrix parameters

        Returns
        -------
        OrderedDict
            a OrderedDict with matrix parameter names as keys and slices as
            values
        """
        matrix_parameter_slices = OrderedDict([])
        for matrix_param in self.matrix_parameter:
            current_param = getattr(self, matrix_param).value
            matrix_parameter_slices[matrix_param] = slice(
                0, current_param.shape[0])
        return matrix_parameter_slices

    def update_pixel_table(self, pixel_table):
        """
        Update the pixel table that defines the virtual pixels

        Parameters
        ----------
        pixel_table : pandas.DataFrame


        """

        self.pixel_table = pixel_table

class GenericPSFModel(modeling.Model):

    def _initialize_model(self, pixel_to_wavelength , wavelength):

        self.pixel_to_wavelength = pixel_to_wavelength
        self.wavelength = wavelength
        self.standard_broadcasting = False
        self.matrix_parameter_slices = self._generate_matrix_parameter_slices()

    @staticmethod
    def _initialize_matrix_coordinates(pixel_to_wavelength, wavelength,
                                       impact_range):
        """
        Calculates the which PSF wavelength centers (`wavelength`) contribute
         to which pixels using `impact_range`

        Parameters
        ----------
        pixel_to_wavelength : ~numpy.ndarray
            an array of length number of pixels mapping pixel_id to wavelength
            (derived from LUT WCS)
        wavelength : ~numpy.ndarray
            an array of wavelength psf centers
        impact_range : ~float
            the number of nm or angstrom to go on either side of the
            pixel wavelength to look for contributing PSF wavelength centers

        Returns
        -------
            row_id : ~numpy.ndarray
                array of row_indices (pixel ids) for the matrix
            column_id: ~numpy.ndarray
                array column indices (PSF wavelength center ids) for the design
                matrix

        """
        wavelength_start_id = np.empty_like(pixel_to_wavelength, dtype=np.int64)
        wavelength_stop_id = np.empty_like(pixel_to_wavelength, dtype=np.int64)
        for i, pixel_wavelength in enumerate(pixel_to_wavelength):
            wavelength_start_id[i] = wavelength.searchsorted(
                pixel_wavelength - impact_range)
            wavelength_stop_id[i] = wavelength.searchsorted(
                pixel_wavelength + impact_range)

        number_of_coord = (wavelength_stop_id - wavelength_start_id).sum()

        row_id = -np.ones(number_of_coord, dtype=np.int64)
        column_id = -np.ones(number_of_coord, dtype=np.int64)

        start_id = 0

        for i, pixel_wavelength in enumerate(pixel_to_wavelength):
            stop_id = (
                start_id + wavelength_stop_id[i] - wavelength_start_id[i])
            row_id[start_id:stop_id] = i
            column_id[start_id:stop_id] = np.arange(wavelength_start_id[i],
                                                    wavelength_stop_id[i],
                                                    dtype=np.int64)
            start_id = stop_id

        return row_id, column_id

    def _generate_matrix_parameter_slices(self):
        """
        Generate slices for the different matrix parameters

        Returns
        -------
        OrderedDict
            a OrderedDict with matrix parameter names as keys and slices as
            values
        """
        matrix_parameter_slices = OrderedDict([])
        for matrix_param in self.matrix_parameter:
            current_param = getattr(self, matrix_param).value
            matrix_parameter_slices[matrix_param] = slice(
                0, current_param.shape[0])
        return matrix_parameter_slices


class GenericPSFBackground(GenericPSFModel):

    background_psf_amplitude = modeling.Parameter(bounds=(0, np.inf))
    resolution = modeling.Parameter(default=10000)
    matrix_parameter = ['background_psf_amplitude']
    inputs = ()
    outputs = ('row_id', 'column_id', 'value')

    def __init__(self, pixel_to_wavelength, wavelength, sigma_impact_width=10.):
        background_psf_amplitude = np.empty_like(wavelength) * np.nan
        super(GenericPSFBackground, self).__init__(
            background_psf_amplitude=background_psf_amplitude)
        self._initialize_model(pixel_to_wavelength, wavelength)
        impact_range = (
            (wavelength.mean() / self.resolution) * FWHM_TO_SIGMA *
            sigma_impact_width)
        self.row_ids, self.column_ids = self._initialize_matrix_coordinates(
            pixel_to_wavelength, wavelength, impact_range)

        self.pixel_wavelength = pixel_to_wavelength[self.row_ids]
        self.virt_pixel_wavelength = self.wavelength[self.column_ids]


    def generate_design_matrix_coordinates(self, resolution):
        row_ids = self.row_ids
        column_ids = self.column_ids
        sigma = (self.virt_pixel_wavelength / resolution) * FWHM_TO_SIGMA

        norm_factor = 1 / (sigma * np.sqrt(2 * np.pi))
        pixel_wavelength = self.pixel_wavelength
        virt_pixel_wavelength = self.virt_pixel_wavelength
        matrix_values = norm_factor * ne.evaluate(
            'exp(-0.5 * '
            '(pixel_wavelength - virt_pixel_wavelength)**2 '
            '/ sigma**2)', )

        return row_ids, column_ids, matrix_values

    def generate_design_matrix(self, resolution):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(resolution))
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))

    def evaluate(self, background_level):
        row_ids = self.pixel_table.pixel_id.values.astype(np.int64)
        column_ids = self.pixel_table.column_ids.values.astype(np.int64)
        matrix_values = self.pixel_table.sub_x.values
        return row_ids, column_ids, matrix_values * 1.

class PSFMoffatTrace(GenericPSFModel):

    psf_amplitude = modeling.Parameter(bounds=(0, np.inf))
    resolution_trace = modeling.Parameter(default=10000)

    trace_pos = modeling.Parameter(default=0.0, bounds=(-6, 6))
    sigma = modeling.Parameter(default=1.0, bounds=(0, 99))
    beta = modeling.Parameter(default=1.5, fixed=True, bounds=(1.1, 3))
    matrix_parameter = ['psf_amplitude']

    def __init__(self, pixel_to_wavelength, pixel_to_slit_pos, wavelength, sigma_impact_width=10.):
        psf_amplitude = np.empty_like(wavelength) * np.nan
        super(PSFMoffatTrace, self).__init__(
            psf_amplitude=psf_amplitude)
        self._initialize_model(pixel_to_wavelength, wavelength)
        impact_range = (
            (wavelength.mean() / self.resolution_trace) * FWHM_TO_SIGMA *
            sigma_impact_width)
        self.row_ids, self.column_ids = self._initialize_matrix_coordinates(
            pixel_to_wavelength, wavelength, impact_range)

        self.pixel_wavelength = pixel_to_wavelength[self.row_ids]
        self.virt_pixel_wavelength = self.wavelength[self.column_ids]
        self.slit_pos = pixel_to_slit_pos[self.row_ids]

    def generate_design_matrix_coordinates(self, resolution_trace, trace_pos,
                                           sigma, beta):
        row_ids = self.row_ids
        column_ids = self.column_ids
        psf_sigma = (
            (self.virt_pixel_wavelength / resolution_trace) * FWHM_TO_SIGMA)

        norm_factor = 1 / (psf_sigma * np.sqrt(2 * np.pi))
        pixel_wavelength = self.pixel_wavelength
        virt_pixel_wavelength = self.virt_pixel_wavelength
        matrix_values = norm_factor * ne.evaluate(
            'exp(-0.5 * '
            '(pixel_wavelength - virt_pixel_wavelength)**2 '
            '/ sigma**2)', )

        moffat_profile = self._moffat(self.slit_pos, trace_pos, sigma, beta)

        return row_ids, column_ids, matrix_values * moffat_profile

    def generate_design_matrix(self, resolution_trace, trace_pos, sigma, beta):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(
                resolution_trace, trace_pos, sigma, beta))
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))

    def evaluate(self, amplitude, trace_pos, sigma, beta):
        row_ids = self.pixel_table.pixel_id.values.astype(np.int64)
        column_ids = self.pixel_table.wavelength_pixel_id.values.astype(np.int64)
        matrix_values = self.pixel_table.sub_x.values.copy()
        moffat_profile = self._moffat(self.pixel_table.slit_pos.values, trace_pos, sigma, beta)
        return row_ids, column_ids, matrix_values * moffat_profile

    def to_spectrum(self):
        try:
            from specutils import Spectrum1D
        except ImportError:
            raise ImportError('specutils needed for this functionality')
        from xtool.fix_spectrum1d import Spectrum1D

        if getattr(self, 'amplitude_uncertainty', None) is None:
            uncertainty = None
        else:
            uncertainty = self.amplitude_uncertainty

        spec = Spectrum1D.from_array(
            self.wavelength * u.nm, self.psf_amplitude.value,
            uncertainty=uncertainty)

        return spec

    @staticmethod
    def _moffat(s, s0, sigma, beta=1.5):
        """
        Calculate the moffat profile
        Parameters
        ----------
        s : ndarray
            slit position
        s0 : float
            center of the Moffat profile
        sigma : float
            sigma of the Moffat profile
        beta : float, optional
            beta parameter of the Moffat profile (default = 1.5)

        Returns
        -------

        """
        fwhm = sigma * SIGMA_TO_FWHM
        alpha = fwhm / (2 * np.sqrt(2**(1.0 / float(beta)) - 1.0))
        norm_factor = (beta - 1.0) / (np.pi * alpha**2)
        return norm_factor * (1.0 + ((s - s0) / alpha)**2)**(-beta)

class PSFSlopedMoffatTrace(PSFMoffatTrace):
    psf_amplitude = modeling.Parameter(bounds=(0, np.inf))
    resolution_trace = modeling.Parameter(default=10000, bounds=(1000, 20000))

    trace_pos = modeling.Parameter(default=0.0, bounds=(-6, 6))
    trace_slope = modeling.Parameter(default=0.0, bounds=(-0.5, 0.5))
    sigma = modeling.Parameter(default=1.0, bounds=(0, 99))
    sigma_slope = modeling.Parameter(default=0.0, bounds=(-0.5, 0.5))
    beta = modeling.Parameter(default=1.5, fixed=True, bounds=(1.1, 3.))
    matrix_parameter = ['psf_amplitude']

    def __init__(self, pixel_to_wavelength, pixel_to_slit_pos, wavelength,
                 sigma_impact_width=10.):
        psf_amplitude = np.empty_like(wavelength) * np.nan
        super(PSFMoffatTrace, self).__init__(
            psf_amplitude=psf_amplitude)
        self._initialize_model(pixel_to_wavelength, wavelength)
        impact_range = (
            (wavelength.mean() / self.resolution_trace) * FWHM_TO_SIGMA *
            sigma_impact_width)
        self.row_ids, self.column_ids = self._initialize_matrix_coordinates(
            pixel_to_wavelength, wavelength, impact_range)

        self.pixel_wavelength = pixel_to_wavelength[self.row_ids]
        self.normed_pixel_wavelength = (
            (self.pixel_to_wavelength - self.pixel_wavelength.min()) /
            (self.pixel_wavelength.max() - self.pixel_wavelength.min()))
        self.normed_pixel_wavelength = (
            (self.normed_pixel_wavelength - 0.5) * 2)[self.row_ids]
        self.virt_pixel_wavelength = self.wavelength[self.column_ids]
        self.slit_pos = pixel_to_slit_pos[self.row_ids]

    def generate_design_matrix_coordinates(
            self, resolution_trace, trace_pos, trace_slope,
            sigma, sigma_slope, beta):
        row_ids = self.row_ids
        column_ids = self.column_ids
        psf_sigma = (
            (self.virt_pixel_wavelength / resolution_trace) * FWHM_TO_SIGMA)

        norm_factor = 1 / (psf_sigma * np.sqrt(2 * np.pi))
        pixel_wavelength = self.pixel_wavelength
        virt_pixel_wavelength = self.virt_pixel_wavelength
        matrix_values = norm_factor * ne.evaluate(
            'exp(-0.5 * '
            '(pixel_wavelength - virt_pixel_wavelength)**2 '
            '/ sigma**2)', )


        normed_wavelength = self.normed_pixel_wavelength
        varying_trace_pos = (trace_pos + trace_slope * normed_wavelength)
        varying_sigma = (sigma + sigma_slope * 0.5 * (normed_wavelength + 1))
        moffat_profile = self._moffat(self.slit_pos,
                                      varying_trace_pos, varying_sigma, beta)
        return row_ids, column_ids, matrix_values * moffat_profile

    def generate_design_matrix(
            self, resolution_trace, trace_pos, trace_slope,
            sigma, sigma_slope, beta):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(resolution_trace, trace_pos, trace_slope,
            sigma, sigma_slope, beta))
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))


class PSFPolynomialMoffatTrace(PSFMoffatTrace):

    psf_amplitude = modeling.Parameter(bounds=(0, np.inf))
    resolution_trace = modeling.Parameter(default=10000, bounds=(1000, 20000))

    trace0 = modeling.Parameter(default=1.0, bounds=(-6, 6))
    trace1 = modeling.Parameter(default=0.0, bounds=(-6, 6))
    trace2 = modeling.Parameter(default=0.0, bounds=(-6, 6))
    trace3 = modeling.Parameter(default=0.0, bounds=(-6, 6))
    trace4 = modeling.Parameter(default=0.0, bounds=(-6, 6))
    sigma0 = modeling.Parameter(default=1.0, bounds=(0, 99))
    sigma1 = modeling.Parameter(default=0.0, bounds=(0, 99))
    sigma2 = modeling.Parameter(default=0.0, bounds=(0, 99))
    sigma3 = modeling.Parameter(default=0.0, bounds=(0, 99))
    sigma4 = modeling.Parameter(default=0.0, bounds=(0, 99))


    beta = modeling.Parameter(default=1.5, fixed=True, bounds=(1.1, 3.))

    matrix_parameter = ['psf_amplitude']

    def __init__(self, pixel_to_wavelength, pixel_to_slit_pos, wavelength,
                 sigma_impact_width=10.):
        psf_amplitude = np.empty_like(wavelength) * np.nan
        super(PSFMoffatTrace, self).__init__(
            psf_amplitude=psf_amplitude)
        self._initialize_model(pixel_to_wavelength, wavelength)
        impact_range = (
            (wavelength.mean() / self.resolution_trace) * FWHM_TO_SIGMA *
            sigma_impact_width)
        self.row_ids, self.column_ids = self._initialize_matrix_coordinates(
            pixel_to_wavelength, wavelength, impact_range)

        self.pixel_wavelength = pixel_to_wavelength[self.row_ids]
        self.normed_pixel_wavelength = (
            (self.pixel_to_wavelength - self.pixel_wavelength.min()) /
            (self.pixel_wavelength.max() - self.pixel_wavelength.min()))
        self.normed_pixel_wavelength = (
            (self.normed_pixel_wavelength - 0.5) * 2)[self.row_ids]
        self.virt_pixel_wavelength = self.wavelength[self.column_ids]
        self.slit_pos = pixel_to_slit_pos[self.row_ids]

    def generate_design_matrix_coordinates(
            self, resolution_trace, trace0, trace1, trace2, trace3, trace4,
                                           sigma0, sigma1, sigma2, sigma3, sigma4, beta):
        row_ids = self.row_ids
        column_ids = self.column_ids
        psf_sigma = (
            (self.virt_pixel_wavelength / resolution_trace) * FWHM_TO_SIGMA)

        norm_factor = 1 / (psf_sigma * np.sqrt(2 * np.pi))
        pixel_wavelength = self.pixel_wavelength
        virt_pixel_wavelength = self.virt_pixel_wavelength
        matrix_values = norm_factor * ne.evaluate(
            'exp(-0.5 * '
            '(pixel_wavelength - virt_pixel_wavelength)**2 '
            '/ psf_sigma**2)', )


        normed_wavelength = self.normed_pixel_wavelength
        varying_trace_pos = np.polyval([trace4, trace3, trace2, trace1, trace0],
                                       normed_wavelength)
        varying_sigma = np.abs(np.polyval([sigma4, sigma3, sigma2, sigma1, sigma0],normed_wavelength))
        moffat_profile = self._moffat(self.slit_pos,
                                      varying_trace_pos, varying_sigma, beta)
        return row_ids, column_ids, matrix_values * moffat_profile

    def generate_design_matrix(
            self, resolution_trace, trace0, trace1, trace2, trace3, trace4,
            sigma0, sigma1, sigma2, sigma3, sigma4, beta):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(
                resolution_trace, trace0, trace1, trace2, trace3, trace4,
                sigma0, sigma1, sigma2, sigma3, sigma4, beta))
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))
