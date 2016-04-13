import numpy as np
from astropy import modeling
from numpy.polynomial import polynomial


class LUTOrderWCS(modeling.Model):
    inputs = ('x', 'y')
    outputs = ('wave', 'slit')
    pix_to_wave_grid = modeling.Parameter()
    pix_to_slit_grid = modeling.Parameter()

    def __init__(self, transform_pixel_to_wavelength, transform_pixel_to_slit,
                 mask):
        """
        A Look Up Table (LUT) World Coordinate System (WCS) that can transform
        from pixel coordinate input to wavelength and slit coordinates

        Parameters
        ----------
        transform_pixel_to_wavelength : astropy.units.Quantity
        transform_pixel_to_slit : astropy.units.Quantity
        mask : np.ndarray
        """

        self.mask = mask
        self.x_grid, self.y_grid = self._generate_coordinate_grid(mask)

        self.pix_to_wave_ma = np.ma.MaskedArray(
            transform_pixel_to_wavelength.value, ~mask, fill_value=np.nan)
        self.pix_to_slit_ma = np.ma.MaskedArray(
            transform_pixel_to_slit.value, ~mask, fill_value=np.nan)

        self.pix_to_wave_unit = transform_pixel_to_wavelength.unit
        self.pix_to_slit_unit = transform_pixel_to_slit.unit

        super(LUTOrderWCS, self).__init__(self.pix_to_wave_ma.filled(),
                                          self.pix_to_slit_ma.filled())
        self.standard_broadcasting = False


    @property
    def x(self):
        return self.x_grid.compressed()

    @property
    def y(self):
        return self.y_grid.compressed()

    @property
    def pix_to_wave(self):
        return self.pix_to_wave_ma.compressed()

    @property
    def pix_to_slit(self):
        return self.pix_to_slit_ma.compressed()


    def evaluate(self, x, y, pix_to_wave, pix_to_slit):
        x = np.int64(x)
        y = np.int64(y)
        return np.squeeze(pix_to_wave)[y, x], np.squeeze(pix_to_slit)[y, x]

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

    def _update_mask(self, mask):
        self.pix_to_wave_ma.mask = mask
        self.pix_to_slit_ma.mask = mask
        self.x_grid.mask = mask
        self.y_grid.mask = mask
        self.pix_to_wave_grid = self.pix_to_wave_ma.filled()
        self.pix_to_slit_grid = self.pix_to_slit_ma.filled()

class PolynomialOrderWCS(modeling.Model):
    inputs = ('x', 'y')
    outputs = ('wave', 'slit')

    wave_transform_coef = modeling.Parameter()
    slit_transform_coef = modeling.Parameter()


    @classmethod
    def from_lut_order_wcs(cls, lut_order_wcs, poly_order=(2, 3)):
        """
        Fits a polynomial on n-th degree to the WCS transform
        Parameters
        ----------
        lut_order_wcs : LUTOrderWCS
            Lookup Table WCS to be fit
        poly_order : int or tuple, optional
            tuple consisting of two integers describing the polynomial in
            wave and slit default=(2, 3)

        Returns
        -------
        polynomial_order_wcs: PolynomialOrderWCS
        """

        if not hasattr(poly_order, '__iter__'):
            poly_order = (poly_order, poly_order)

        wave_model_coef = cls.polyfit2d(lut_order_wcs.x, lut_order_wcs.y,
                                        lut_order_wcs.pix_to_wave, poly_order)

        slit_model_coef = cls.polyfit2d(lut_order_wcs.x, lut_order_wcs.y,
                                        lut_order_wcs.pix_to_slit, poly_order)

        return cls(wave_model_coef, slit_model_coef)

    def __init__(self, wave_transform_coef, slit_transform_coef):
        super(PolynomialOrderWCS, self).__init__(wave_transform_coef,
                                                 slit_transform_coef)
        self.standard_broadcasting = False


    @staticmethod
    def polyfit2d(x, y, f, deg):
        """
        Fits a 2D polynomial and returns the coefficients
        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray
        f : numpy.ndarray
        deg : tuple

        Returns
        -------
        poly_2d_coef : numpy.ndarray

        """
        deg = np.asarray(deg)
        vander = polynomial.polyvander2d(x, y, deg)
        vander = vander.reshape((-1, vander.shape[-1]))
        f = f.reshape((vander.shape[0],))
        c = np.linalg.lstsq(vander, f)[0]
        return c.reshape(deg + 1)

    def evaluate(self, x, y, wave_transform_coef, slit_transform_coef):
        x = np.squeeze(x)
        y = np.squeeze(x)
        wave_transform_coef = np.squeeze(wave_transform_coef)
        slit_transform_coef = np.squeeze(slit_transform_coef)
        return (polynomial.polyval2d(x, y, wave_transform_coef),
                polynomial.polyval2d(x, y, slit_transform_coef))

