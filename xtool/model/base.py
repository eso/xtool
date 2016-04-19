from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy import modeling
from scipy import sparse
from scipy.spatial import cKDTree

from xtool.wcs import PolynomialOrderWCS

FWHM_TO_SIGMA = 1. / (8 * np.log(2))**0.5
SIGMA_TO_FWHM = 1 / FWHM_TO_SIGMA

class VirtualPixelWavelength(modeling.Model):

    inputs = ()
    outputs = ('pixel_table', )

    wave_transform_coef = modeling.Parameter()

    wavelength_sampling_defaults = {'UVB' : 0.04,
                                    'VIS' : 0.04,
                                    'NIR' : 0.1}

    @classmethod
    def from_order(cls, order, poly_order=(2, 3), wavelength_sampling=None):
        """
        Instantiate a Virtualpixel table from the order raw Lookup Table WCS
        and then fitting a Polynomial WCS with given orde
        Parameters
        ----------
        order : xtool.data.Order
            order data or object
        poly_order : tuple or int
        wavelength_sampling: float, optional
            float for wavelength spacing. If `None` will use for different arms
            * UVB - 0.04 nm
            * VIS - 0.04 nm
            * NIR - 0.1 nm

        Returns
        -------
        VirtualPixelWavelength
        """
        polynomial_wcs = PolynomialOrderWCS.from_lut_order_wcs(order.wcs,
                                                               poly_order)

        if wavelength_sampling is None:
            wavelength_sampling = cls.wavelength_sampling_defaults[
                order.instrument_arm]

        wavelength_bins = np.arange(
            order.wcs.pix_to_wave.min(),
            order.wcs.pix_to_wave.max() + wavelength_sampling,
            wavelength_sampling)

        return VirtualPixelWavelength(polynomial_wcs, order.wcs,
                                      wavelength_bins)

    def __init__(self, polynomial_wcs, lut_wcs, wavelength_pixels,
                 sub_sampling=5):
        """
        A model that represents the virtual pixels of the model

        Parameters
        ----------
        polynomial_wcs : xtool.wcs.PolynomialOrderWCS
        lut_wcs : xtool.wcs.LUTOrderWCS
        wavelength_pixels : numpy.ndarray
        sub_sampling : int
            how many subsamples to create in each dimension
        """

        super(VirtualPixelWavelength, self).__init__(
            polynomial_wcs.wave_transform_coef)


        self.lut_wcs = lut_wcs
        self.polynomial_wcs = polynomial_wcs
        self.wavelength_pixels = wavelength_pixels
        self.standard_broadcasting = False
        self.sub_sampling = sub_sampling

    @staticmethod
    def _initialize_sub_pixel_table(x, y, sub_sampling):
        """
        Generate a subgrid for x and y with sampling `sub_sampling`.
        For example, for a subsampling of 5 each real x y coordinate will have
        25 additional entries.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray
        sub_sampling : int

        Returns
        -------
        pandas.DataFrame
            subpixel table with pixel index, sub_x, sub_y
        """
        sub_edges = sub_sampling + 1
        sub_y_delta, sub_x_delta = np.mgrid[-0.5:0.5:sub_edges * 1j,
                                   -0.5:0.5:sub_edges * 1j]

        sub_pixel_size = 1 / np.float(sub_sampling)
        sub_x_delta = sub_x_delta[:-1, :-1] + 0.5 * sub_pixel_size
        sub_y_delta = sub_y_delta[:-1, :-1] + 0.5 * sub_pixel_size

        sub_x = np.add.outer(x.flatten(), sub_x_delta.flatten()).flatten()
        sub_y = np.add.outer(y.flatten(), sub_y_delta.flatten()).flatten()
        pixel_id = np.multiply.outer(
            np.arange(len(x.flatten())),
                      np.ones_like(sub_x_delta.flatten(),
                                   dtype=np.int64)).flatten()


        sub_pixel_table = pd.DataFrame(columns=['pixel_id', 'sub_x', 'sub_y', ],
                                   data={'pixel_id': pixel_id,
                                         'sub_x': sub_x,
                                         'sub_y': sub_y})
        return sub_pixel_table

    @staticmethod
    def _add_wavelength_sub_pixel_table(sub_pixel_table, wavelength_bins,
                                        wave_transform_coef):
        """
        Add a 'sub_wavelength' column that is calculated for each of the
        sub-pixel locations using the PolynomialWCS and add a
        'wavelength_bin_id' column that identifies the wavelength
        bin that this sub pixel corresponds to.
        Parameters
        ----------
        sub_pixel_table : pandas.DataFrame
        wavelength_bins : numpy.ndarray
        wave_transform_coef : numpy.ndarray

        Returns
        -------

        """
        kdt = cKDTree(wavelength_bins[np.newaxis].T)

        sub_pixel_table['sub_wavelength'] = np.polynomial.polynomial.polyval2d(
            sub_pixel_table.sub_x.values, sub_pixel_table.sub_y.values,
            wave_transform_coef)

        _, sub_pixel_table['wavelength_pixel_id'] = kdt.query(
            sub_pixel_table.sub_wavelength.values[np.newaxis].T)

        return sub_pixel_table

    @staticmethod
    def _generate_pixel_table(sub_pixel_table, sub_sampling):
        pixel_table = sub_pixel_table.groupby(
            ['pixel_id', 'wavelength_pixel_id']).sub_x.count()
        pixel_table /= sub_sampling ** 2
        pixel_table = pixel_table.reset_index()

        return pixel_table

    def evaluate(self, wave_transform_coef):
        sub_pixel_table = self._initialize_sub_pixel_table(
            self.lut_wcs.x, self.lut_wcs.y, self.sub_sampling)
        sub_pixel_table = self._add_wavelength_sub_pixel_table(
            sub_pixel_table, self.wavelength_pixels,
            np.squeeze(wave_transform_coef))
        pixel_table = self._generate_pixel_table(sub_pixel_table,
                                                 self.sub_sampling)

        pixel_table['wavelength'] = self.wavelength_pixels[
            pixel_table.wavelength_pixel_id]

        pixel_table['slit_pos'] = self.lut_wcs.pix_to_slit[
            pixel_table.pixel_id]

        pixel_table['normed_wavelength'] = (
                (pixel_table.wavelength - pixel_table.wavelength.min()) /
                (pixel_table.wavelength.max() - pixel_table.wavelength.min())) * 2 - 1


        if (pixel_table.wavelength_pixel_id.unique().size
                != self.wavelength_pixels.size):
            raise ValueError("Not all wavelength pixels covered by real pixels")
        return pixel_table


class OrderModel(object):

    def __init__(self, model_list):
        self.parameter_dict = self._generate_parameter_dict(model_list)
        self.model_list = model_list


    def __getitem__(self, item):
        return self.model_list[item]

    @property
    def fittable_parameter_names(self):
        return self._get_variable_normal_parameters()

    @property
    def fittable_parameter_dict(self):
        return OrderedDict([(param_name, getattr(self, param_name).value)
                            for param_name in self.fittable_parameter_names])

    def _get_variable_normal_parameters(self):
        """
        Get all parameters that are not fixed and not matrix parameters

        Returns
        -------
        list
        """

        parameter_names = []
        for model in self.model_list:
            for parameter_name in model.param_names:
                if parameter_name in model.matrix_parameter:
                    continue
                elif getattr(model, parameter_name).fixed:
                    continue
                else:
                    parameter_names.append(parameter_name)
        return parameter_names


    def __getattr__(self, item):
        if 'parameter_dict' in self.__dict__ and item in self.parameter_dict:
            return self.parameter_dict[item]
        else:
            return super(OrderModel, self).__getattribute__(item)

    def __setattr__(self, key, value):
        if key in getattr(self, 'parameter_dict', {}):
            setattr(self.parameter_dict[key]._model, key, value)
        else:
            super(OrderModel, self).__setattr__(key, value)

    def __repr__(self):
        parameter_str = [
            '{0}={1.value}{2}'.format(param_name,
                                   getattr(param._model, param_name),
                                      ' [f]' if getattr(
                                          param._model, param_name).fixed
                                      else '')
            for param_name, param in self.parameter_dict.items()]
        return "<OrderModel({0})>".format(', '.join(parameter_str))


    @staticmethod
    def _generate_parameter_dict(model_list):
        """
        Generate a dictionary mapping parameter name to model instance. This
        excludes matrix parameters.

        Parameters
        ----------
        model_list : list of astropy.modeling.Model

        Returns
        -------
        OrderedDict
        """
        parameter_dict = OrderedDict()

        for model in model_list:
            for parameter_name in model.param_names:
                if parameter_name == model.matrix_parameter:
                    continue
                else:
                    if parameter_name in parameter_dict:
                        raise NotImplementedError(
                            'Currently only one model of each type is allowed')
                    else:
                        parameter_dict[parameter_name] = getattr(model,
                                                                 parameter_name)

        return parameter_dict

    @property
    def result_dict(self):
        result_dict = {}
        for i, model in enumerate(self.model_list):
            result_dict[model.__class__.__name__] = self.result[0][self.model_idx[i]:self.model_idx[i+1]]
        return result_dict

    def generate_design_matrix(self, order, **kwargs):
        design_matrices = []
        for model in self.model_list:
            model_call_dict = self._generate_model_call_dict(model, kwargs)
            design_matrices.append(
                model.generate_design_matrix(**model_call_dict))
        model_widths = [item.shape[1] for item in design_matrices]

        dmatrix = sparse.hstack(design_matrices)
        dmatrix.data /= order.uncertainty.compressed()[dmatrix.row]
        return dmatrix, model_widths


    @staticmethod
    def _generate_model_call_dict(model, kwargs):
        model_call_dict = {}

        for parameter_name in model.param_names:
            if parameter_name in model.matrix_parameter:
                continue
            elif getattr(model, parameter_name).fixed:
                model_call_dict[parameter_name] = getattr(
                    model, parameter_name).value
            else:
                if parameter_name not in kwargs:
                    raise ValueError(
                        'Fittable parameter {0} has no assigned value'.format(
                            parameter_name))
                else:
                    model_call_dict[parameter_name] = kwargs[parameter_name]

        return model_call_dict

    def solve_design_matrix(self, dmatrix, order, solver='lsmr',
                            solver_dict={}):
        b = order.data.compressed() / order.uncertainty.compressed()
        if solver == 'lsmr':
            result = sparse.linalg.lsmr(dmatrix.tobsr(), b, **solver_dict)
        else:
            raise NotImplementedError('Solver {0} is not implemented'.format(
                solver))
        return result

    def set_matrix_parameters(self, b, model_widths):
        matrix_model_columns = np.cumsum(model_widths)
        for i, model in enumerate(self.model_list):
            current_matrix_column = (
                matrix_model_columns[i] - matrix_model_columns[0])
            for (matrix_parameter,
                 matrix_slice) in model.matrix_parameter_slices.items():
                setattr(model, matrix_parameter,
                        b[matrix_slice.start + current_matrix_column:
                        matrix_slice.stop + current_matrix_column])

    def set_matrix_uncertainties(self, order):

        dmatrix, model_widths = self.generate_design_matrix(
            order, **self.fittable_parameter_dict)
        dmatrix_inv = sparse.linalg.inv(dmatrix.T * dmatrix)
        uncertainties = np.sqrt(dmatrix_inv.diagonal())

        matrix_model_columns = np.cumsum(model_widths)
        for i, model in enumerate(self.model_list):
            current_matrix_column = (
                matrix_model_columns[i] - matrix_model_columns[0])
            for (matrix_parameter,
                 matrix_slice) in model.matrix_parameter_slices.items():
                setattr(
                    model, matrix_parameter + '_uncertainty',
                    uncertainties[matrix_slice.start + current_matrix_column:
                    matrix_slice.stop + current_matrix_column])

    def evaluate(self, order, solver='lsmr', solver_dict={}, **kwargs):
        dmatrix, model_widths = self.generate_design_matrix(order, **kwargs)
        result = self.solve_design_matrix(dmatrix, order, solver=solver,
                                          solver_dict=solver_dict)

        self.set_matrix_parameters(result[0], model_widths)
        return (dmatrix * result[0]) * order.uncertainty.compressed()

    def evaluate_to_frame(self, order, solver='lsmr', solver_dict={}, **kwargs):
        result = self.evaluate(order, solver, solver_dict, **kwargs)
        result_frame = np.ma.MaskedArray(
            np.empty_like(order.data, dtype=np.float64), mask=order.data.mask)
        result_frame.data[order.wcs.y, order.wcs.x] = result
        return result_frame

    def evaluate_residuals(self, order, solver='lsmr', solver_dict={},
                           **kwargs):
        result = self.evaluate(order, solver, solver_dict, **kwargs)
        b = order.data.compressed()
        return ((result - b) / order.uncertainty.compressed())


    def evaluate_chi2(self, order, solver='lsmr', solver_dict={}, **kwargs):
        return np.sum(
            self.evaluate_residuals(order, solver, solver_dict, **kwargs)**2)

    def calculate_to_frame(self, order, solver='lsmr', solver_dict={}):
        return self.evaluate_to_frame(
            order, solver, solver_dict,
            **{param_name:self.parameter_dict[param_name].value
               for param_name in self.fittable_parameter_names})