import numpy as np
from astropy import modeling
from numpy.polynomial import polynomial


class LUTOrderWCS(object):
    def __init__(self, transform_pixel_to_wavelength, transform_pixel_to_slit,
                 mask):
        """
        A Look Up Table (LUT) World Coordinate System (WCS) that can transform
        from input

        Parameters
        ----------
        transform_pixel_to_wavelength :
        transform_pixel_to_slit :
        mask :
        """

        self.x_grid, self.y_grid = self._generate_coordinate_grid(mask)

        self.pix_to_wave = np.ma.MaskedArray(
            transform_pixel_to_wavelength.value, ~mask, fill_value=np.nan)
        self.pix_to_slit = np.ma.MaskedArray(
            transform_pixel_to_slit.value, ~mask, fill_value=np.nan)

        self.pix_to_wave_unit = transform_pixel_to_wavelength.unit
        self.pix_to_slit_unit = transform_pixel_to_slit.unit


    @property
    def x(self):
        return self.x_grid.compressed()

    @property
    def y(self):
        return self.y_grid.compressed()

    @staticmethod
    def _generate_coordinate_grid(mask):
        """
        generate a coordinate grid for the order slice
        Parameters
        ----------
        mask : numpy.ndarray
            boolean masked with True for valid data and false for invalid data

        Returns
        -------
        x_grid : numpy.ma.MaskedArray
        y_grid : numpy.ma.MaskedArray

        """
        y_grid, x_grid = np.mgrid[:mask.shape[0], :mask.shape[1]]

        return (np.ma.MaskedArray(x_grid, ~mask, fill_value=-1),
                np.ma.MaskedArray(y_grid, ~mask, fill_value=-1))


class PolynomialOrderWCS(modeling.Model):

    wave_transform_coef = modeling.Parameter()
    slit_model_coef = modeling.Parameter()

    def __init__(self, wave_model_coef, slit_model_coef):
        super(PolynomialOrderWCS, self).__init__(wave_model_coef,
                                                 slit_model_coef)


    @staticmethod
    def polyfit2d(x, y, f, deg):
        x = np.asarray(x)
        y = np.asarray(y)
        f = np.asarray(f)
        deg = np.asarray(deg)
        vander = polynomial.polyvander2d(x, y, deg)
        vander = vander.reshape((-1, vander.shape[-1]))
        f = f.reshape((vander.shape[0],))
        c = np.linalg.lstsq(vander, f)[0]
        return c.reshape(deg + 1)