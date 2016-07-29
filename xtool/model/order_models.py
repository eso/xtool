from collections import OrderedDict

import numpy as np
from astropy import modeling
from astropy import units as u
from astropy.nddata import StdDevUncertainty

from scipy import sparse

from xtool.model.base import SIGMA_TO_FWHM



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

class GenericBackground(LinearLeastSquaredModel):

    background_level = modeling.Parameter(bounds=(0, np.inf))
    matrix_parameter = ['background_level']
    inputs = ()
    outputs = ('row_id', 'column_id', 'value')

    def __init__(self, pixel_table, wavelength_pixels):
        background_level = np.empty_like(
            pixel_table.wavelength_pixel_id.unique())
        super(GenericBackground, self).__init__(background_level)
        self._initialize_lls_model(pixel_table, wavelength_pixels)


    def generate_design_matrix_coordinates(self):
        row_ids = self.pixel_table.pixel_id.values
        column_ids = self.pixel_table.wavelength_pixel_id.values
        matrix_values = self.pixel_table.sub_x.values
        return row_ids, column_ids, matrix_values * 1.

    def generate_design_matrix(self):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates())
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))

    def evaluate(self, background_level):
        row_ids = self.pixel_table.pixel_id.values.astype(np.int64)
        column_ids = self.pixel_table.column_ids.values.astype(np.int64)
        matrix_values = self.pixel_table.sub_x.values
        return row_ids, column_ids, matrix_values * 1.

class PolynomialBackground(LinearLeastSquaredModel):
    background_level = modeling.Parameter(bounds=(0, np.inf))
    matrix_parameter = ['background_level']
    inputs = ()
    outputs = ('row_id', 'column_id', 'value')

    def __init__(self, pixel_table, wavelength_pixels):
        background_level = np.empty_like(
            pixel_table.wavelength_pixel_id.unique())
        super(GenericBackground, self).__init__(background_level)
        self._initialize_lls_model(pixel_table, wavelength_pixels)


    def generate_design_matrix_coordinates(self):
        row_ids = self.pixel_table.pixel_id.values
        column_ids = self.pixel_table.wavelength_pixel_id.values
        matrix_values = self.pixel_table.sub_x.values
        return row_ids, column_ids, matrix_values * 1.

    def generate_design_matrix(self):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates())
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))

    def evaluate(self, background_level):
        row_ids = self.pixel_table.pixel_id.values.astype(np.int64)
        column_ids = self.pixel_table.column_ids.values.astype(np.int64)
        matrix_values = self.pixel_table.sub_x.values
        return row_ids, column_ids, matrix_values * 1.


class MoffatTrace(LinearLeastSquaredModel):

    amplitude = modeling.Parameter(bounds=(0, np.inf))
    trace_pos = modeling.Parameter(default=0.0, bounds=(-6, 6))
    sigma = modeling.Parameter(default=1.0, bounds=(0, 99))
    beta = modeling.Parameter(default=1.5, fixed=True, bounds=(1.1, 3))
    matrix_parameter = ['amplitude']

    def __init__(self, pixel_table, wavelength_pixels):
        amplitude = np.empty_like(pixel_table.wavelength_pixel_id.unique())
        super(MoffatTrace, self).__init__(amplitude=amplitude * np.nan)
        self._initialize_lls_model(pixel_table, wavelength_pixels)


    def generate_design_matrix_coordinates(self, trace_pos, sigma, beta):
        row_ids = self.pixel_table.pixel_id.values
        column_ids = self.pixel_table.wavelength_pixel_id.values
        matrix_values = self.pixel_table.sub_x.values.copy()
        moffat_profile = self._moffat(self.pixel_table.slit_pos.values,
                                      trace_pos, sigma, beta)

        return row_ids, column_ids, matrix_values * moffat_profile

    def generate_design_matrix(self, trace_pos, sigma, beta):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(trace_pos, sigma, beta))
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
            self.wavelength_pixels * u.nm, self.amplitude.value,
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


class SlopedMoffatTrace(MoffatTrace):
    amplitude = modeling.Parameter(bounds=(0, np.inf))
    trace_pos = modeling.Parameter(default=0.0, bounds=(-6, 6))
    trace_slope = modeling.Parameter(default=0.0, bounds=(-0.5, 0.5))
    sigma = modeling.Parameter(default=1.0, bounds=(0, 99))
    beta = modeling.Parameter(default=1.5, fixed=True, bounds=(1.1, 3.))
    matrix_parameter = ['amplitude']


    def __init__(self, pixel_table, wavelength_pixels):
        amplitude = np.empty_like(pixel_table.wavelength_pixel_id.unique())
        super(MoffatTrace, self).__init__(amplitude=amplitude * np.nan)
        self._initialize_lls_model(pixel_table, wavelength_pixels)


    def generate_design_matrix_coordinates(self, trace_pos, trace_slope, sigma,
                                           beta):
        row_ids = self.pixel_table.pixel_id.values
        column_ids = self.pixel_table.wavelength_pixel_id.values
        matrix_values = self.pixel_table.sub_x.values
        moffat_profile = self._moffat(self.pixel_table.slit_pos.values,
                                      trace_pos, sigma, beta)

        return row_ids, column_ids, matrix_values * moffat_profile

    def generate_design_matrix(self, trace_pos, trace_slope, sigma, beta):
        row_ids, column_ids, matrix_values = (
            self.generate_design_matrix_coordinates(trace_pos, trace_slope,
                                                    sigma, beta))
        return sparse.coo_matrix((matrix_values, (row_ids, column_ids)))

    def evaluate(self, amplitude, trace_pos, trace_slope, sigma, beta):
        row_ids = self.pixel_table.pixel_id.values.astype(np.int64)
        column_ids = self.pixel_table.wavelength_pixel_id.values.astype(
            np.int64)
        matrix_values = self.pixel_table.sub_x.values
        varying_trace_pos = (
            trace_pos + trace_slope * self.pixel_table.normaled_wavelength)
        moffat_profile = self._moffat(self.pixel_table.slit_pos.values,
                                      varying_trace_pos, sigma, beta)
        return row_ids, column_ids, matrix_values * moffat_profile