import os
from glob import glob

import matplotlib as mpl
import matplotlib.cm
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import xtool
from xtool.wcs import LUTOrderWCS


class XShooterData(object):
    data_fname = 'SCI_SLIT_DIVFF'

    transform_pixel_to_wave_fname = 'SCI_SLIT_WAVE_MAP'
    transform_pixel_to_slit_fname = 'SCI_SLIT_SLIT_MAP'

    spectral_format_fname = 'SPECTRAL_FORMAT_TAB'

    def __init__(self, data_dir, data_fname=None):
        """

        Parameters
        ----------
        data_dir : numpy.ndarray

        """

        if data_fname is not None:
            self.data_fname = data_fname

        data_full_fname = glob(os.path.join(
            data_dir, '{0}*'.format(self.data_fname)))
        assert len(data_full_fname) == 1, (
            "More than one datafile found {0}".format(data_full_fname))


        science_data_full_fname = data_full_fname[0]
        self.instrument_arm = fits.getval(science_data_full_fname,
                                          'HIERARCH ESO SEQ ARM')
        self.xshooter_dir = os.path.dirname(science_data_full_fname)

    def __getattr__(self, item):
        read_item = 'read_{0}'.format(item)
        if hasattr(self, read_item):
            if not hasattr(self, '_' + item):
                setattr(self, '_' + item, getattr(self, read_item)())
            return getattr(self, '_' + item)
        else:
            raise AttributeError()

    def __getitem__(self, item):
        if item not in self.order_table.index:
            raise IndexError('Dataset only contains orders {0} - {1}'.format(
                self.order_table.index.min(), self.order_table.index.max()))
        else:
            return Order.from_xshooter_data(self, item)


    @property
    def xbin(self):
        """
        Returns
        -------
            : int
        """

        if "HIERARCH ESO DET WIN1 BINX" not in self.science_header:
            print "WARNING: XBin not known defaulting to 1"
            return 1
        else:
            return self.science_header.get("HIERARCH ESO DET WIN1 BINX")

    @property
    def ybin(self):
        """
        Returns
        -------
            : int
        """
        
        if "HIERARCH ESO DET WIN1 BINY" not in self.science_header:
            print "WARNING: YBin not known defaulting to 1"
            return 1
        else:
            return self.science_header.get("HIERARCH ESO DET WIN1 BINY")

    def read_data(self, xshooter_dir, data_fname, ext=0):
        """
        This is a helper function to read fits data from an xshooter directory


        Parameters
        ----------
        xshooter_dir : str
          directory for the XShooter science data
        data_fname : str
            name prefix for the datafile
        ext : int
            FITS extension (default = 0)

        Returns
        -------
        data_array : numpy.ndarray
            fits data from that file

        """

        fname = os.path.join(xshooter_dir, "{0}_{1}.fits".format(
            data_fname, self.instrument_arm))

        return fits.getdata(fname, ext=ext)

    def read_science_data(self):
        """
        Utility function to read the science data

        Returns
        -------
            : numpy.ndarray
            science data array
        """

        return self.read_data(self.xshooter_dir, self.data_fname)

    def read_science_header(self):
        """
        Utility function to read the science header
        Returns
        -------
            : astropy.io.fits.header.Header
            Header for the science dataset
        """

        fname = os.path.join(self.xshooter_dir, "{0}_{1}.fits".format(
            self.data_fname, self.instrument_arm))

        return fits.getheader(fname)

    def read_uncertainty(self):
        """
        Utility function to read the science uncertainty
        Returns
        -------
            : numpy.ndarray
        """
        return self.read_data(self.xshooter_dir, self.data_fname, ext=1)

    def read_flags(self):
        """
        Utility function to read the science flags and convert it to an
        integer
        Returns
        -------
            : numpy.ndarray
        """

        return np.int64(
            self.read_data(self.xshooter_dir, self.data_fname, ext=2))

    def read_transform_pix_to_wave(self):
        """
        Utility function to read the transform and convert it to a quantity

        Returns
        -------
        pix_to_wave : astropy.units.Quantity
        """

        return self.read_data(self.xshooter_dir,
                              self.transform_pixel_to_wave_fname) * u.nm

    def read_transform_pix_to_slit(self):
        """
        Utility function to read the transform and convert it to a quantity

        Returns
        -------
        pix_to_wave : astropy.units.Quantity
        """

        return self.read_data(
            self.xshooter_dir,
            self.transform_pixel_to_slit_fname) * u.arcsecond


    def read_spectral_format_table(self):
        """

        Utility function to read the spectral format table


        Returns
        -------
        spectral_format_table : pandas.DataFrame
        """

        spectral_format = Table.read(
            os.path.join(xtool_data_path, '{0}_{1}.fits'.format(
                self.spectral_format_fname, self.instrument_arm))).to_pandas()
        spectral_format['ORDER'] = spectral_format['ORDER'].astype(int)

        spectral_format = spectral_format.set_index('ORDER')
        spectral_format.index = spectral_format.index.rename('ABSORDER')
        return spectral_format


    def read_order_table(self):
        """
        Utility function to read the order table

        Returns
        -------
            : pandas.DataFrame
        """

        order_table = Table(self.read_data(self.xshooter_dir,
                              'ORDER_TAB_AFC_SLIT'))
        return order_table.to_pandas().set_index('ABSORDER')

    def __repr__(self):
        return "<XShooterData {0} {1}>".format(self.science_header['ARCFILE'],
                                               self.instrument_arm)

    def get_order_coefficients(self, order_id):
        """
        Getting the polynomial coefficients for a specific order

        Parameters
        ----------
        order_id : int

        Returns
        -------
        slice_y : tuple
            tuple of starty and endy

        edg_up_coef: list
            list of polynomial coefficients for upper edge

        edg_lo_coef: list
            list of polynomial coefficients for lower edge


        """

        order_data = self.order_table.ix[order_id]
        spectral_format_table = self.spectral_format_table.ix[order_id]
        deg = np.int(order_data['DEGY'])
        if self.instrument_arm == 'NIR' and order_id > 15:
            slice_y = slice(np.int(spectral_format_table['YMIN']) - 1 + 30,
                            np.int(spectral_format_table['YMAX']) - 1 - 30)
        elif 'YMIN' in spectral_format_table:
            slice_y = slice(np.int(spectral_format_table.get('YMIN') - 1),
                            np.int(spectral_format_table.get('YMAX') - 1))
        else:
            slice_y = slice(None, None)

        edg_up_coef = [order_data['EDGUPCOEF{0}'.format(i)]
                       for i in xrange(deg + 1)]
        edg_lo_coef = [order_data['EDGLOCOEF{0}'.format(i)]
                       for i in xrange(deg + 1)]
        return slice_y, edg_up_coef, edg_lo_coef

    def calculate_edges(self, order_id):
        """
        Calculate the edges for a given order
        Parameters
        ----------
        order_id : int
            absolute order id

        Returns
        -------
        yrange : numpy.ndarray
            y coordinate for order polynomial

        lower_edges : numpy.ndarray
            x coordinate for lower edge of the order

        upper_edges : numpy.ndarray
            x coordinate for lower edge of the order

        """

        slice_y, edg_up_coef, edg_lo_coef = self.get_order_coefficients(order_id)
        yrange = (np.arange(self.science_data.shape[0]) * self.ybin)


        lower_edges = np.polynomial.polynomial.polyval(
            yrange, edg_lo_coef)
        upper_edges = np.polynomial.polynomial.polyval(
            yrange, edg_up_coef)

        return slice_y, lower_edges - 1, upper_edges - 1


    def cutout_rectangular_order(self, order_id, data_array, extra_sample=5):
        """
        Cutting out the data in a rectangular fashion for each order without
        masking the other orders

        Parameters
        ----------
        order_id : int
            absolute order id
        data_array : numpy.ndarray
            any numpy.ndarray coming from this dataset
        extra_sample :
            extra marging to cut left and right (defaults to 5 pixels)

        Returns
        -------
        cutout_data_array: numpy.ndarray
            cutout of the data array
        edge_slice: slice
            slice of the whole data_array

        """

        slice_y, lower_edge, upper_edge = self.calculate_edges(order_id)

        edge_slice_start = np.max(
            [0, np.int(lower_edge[slice_y].min() - extra_sample)])
        edge_slice_end = np.int(upper_edge[slice_y].max() + extra_sample)
        edge_slice = slice(edge_slice_start, edge_slice_end)

        return data_array[:, edge_slice], edge_slice

    def cutout_and_mask_order(self, order_id, data_array, extra_sample=5):
        """
        Cutout the order and generate a boolean mask for the data

        Parameters
        ----------
        order_id : int
            Order id
        data_array : numpy.ndarray
            numpy data array to be cut
        extra_sample : int
            pixel to extend either side (default=5)

        Returns
        -------
        cutout_data_array : numpy.ndarray
            cut data array

        mask : numpy.ndarray
            mask only masking the actual order data - dtype = bool

        """
        cutout_data_array, edge_slice = self.cutout_rectangular_order(
            order_id, data_array, extra_sample=extra_sample)

        slice_y, lower_edge, upper_edge = self.calculate_edges(order_id)
        y, x = np.mgrid[
               :cutout_data_array.shape[0], :cutout_data_array.shape[1]]

        mask = ((x > (lower_edge - edge_slice.start)[None].T) &
                (x < (upper_edge - edge_slice.start)[None].T))

        if slice_y.start is not None and slice_y.stop is not None:
            mask = mask & (y > slice_y.start) & (y < slice_y.stop)

        return cutout_data_array, mask

    def get_mask(self, data_array):
        mask = np.zeros_like(data_array).astype(bool)

        y, x = np.mgrid[:data_array.shape[0], :data_array.shape[1]]

        for order_id in self.order_table.index:
            slice_y, lower_edge, upper_edge = self.calculate_edges(order_id)
            mask = mask | ((x > lower_edge[None].T) &
                           (x < upper_edge[None].T) &
                           (y > slice_y.start) & (y < slice_y.stop))
        return mask


    def display_data(self, figure, vmin=0, vmax=500,
                     cmap=matplotlib.cm.gray):
        """
        Displau the XShooter data in

        Parameters
        ----------
        figure : matplotlib.Figure
        vmin : minimum cut [default=0]
        vmax : maximum cut [default=500]
        cmap : matplotlib colormaps [default 'gray']

        """
        ax = figure.add_subplot(111)
        ax.imshow(self.science_data, vmin=vmin, vmax=vmax, cmap=cmap)
        for order_id in self.order_table.index:
            slice_y, low_edge, up_edge = self.calculate_edges(order_id)
#            slice_y2 = slice(
#                self.spectral_format_table.loc[order_id, 'YMIN'] - 1 + 50,
#                self.spectral_format_table.loc[order_id, 'YMAX'] - 1 - 50)

            ax.plot(low_edge, np.arange(len(low_edge)), color='blue', lw=2)
            ax.plot(up_edge, np.arange(len(up_edge)), color='red', lw=2)
            if slice_y.start is not None and slice_y.stop is not None:
                ax.plot([low_edge[slice_y.start], up_edge[slice_y.start]],
                     [slice_y.start, slice_y.start], color='purple', lw=2)
                ax.plot([low_edge[slice_y.stop], up_edge[slice_y.stop]],
                     [slice_y.stop, slice_y.stop], color='purple', lw=2)
            ax.text(0.5 * (low_edge[len(low_edge)/2] +
                           up_edge[len(up_edge)/2]), len(low_edge)/2, str(order_id),
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(facecolor='white'))


class Order(object):
    @classmethod
    def from_xshooter_data(cls, xshooter_data, order_id):
        """
        Construct an xtool.model.Order instance from the an XShooter Data
        Parameters
        ----------
        xshooter_data : xtool.data.XShooterData
            input X-shooter data
        order_id : int
            absolute order ID

        Returns
        -------
            : Order
            order object
        """

        data, mask = xshooter_data.cutout_and_mask_order(
            order_id, xshooter_data.science_data)
        uncertainty, _ = xshooter_data.cutout_rectangular_order(
            order_id, xshooter_data.uncertainty)
        flags, _ = xshooter_data.cutout_rectangular_order(
            order_id, xshooter_data.flags)

        transform_pix_to_wave, _ = xshooter_data.cutout_rectangular_order(
            order_id, xshooter_data.transform_pix_to_wave)

        transform_pix_to_slit, _ = xshooter_data.cutout_rectangular_order(
            order_id, xshooter_data.transform_pix_to_slit)

        order_wcs = LUTOrderWCS(transform_pix_to_wave, transform_pix_to_slit,
                                mask)

        return cls(data, uncertainty, flags, mask, order_id,
                   xshooter_data.instrument_arm,
                   xshooter_data.science_header['ARCFILE'], order_wcs)

    def __init__(self, data, uncertainty, flags, order_mask, order_id,
                 instrument_arm, data_set_id, wcs):
        """

        Parameters
        ----------
        data : numpy.ndarray
        uncertainty : numpy.ndarray
        flags : numpy.ndarray
        order_mask : numpy.ndarray
        order_id : int
        instrument_arm : str
            can be 'UVB', 'VIS', 'NIR'
        data_set_id : str
        wcs : LUTOrderWCS
        """
        self.order_mask = order_mask

        self.data = self._generate_masked_array(data)
        self.uncertainty = self._generate_masked_array(uncertainty)
        self.flags = self._generate_masked_array(flags, fill_value=-1)

        self.order_id = order_id
        self.wcs = wcs
        self.data_set_id = data_set_id
        self.instrument_arm = instrument_arm

    def _generate_masked_array(self, data_set, fill_value=np.nan):
        """
        Make a masked array using numpy.ma.MaskedArray
        Parameters
        ----------
        data_set : numpy.ndarray

        Returns
        -------
        masked_data_set : numpy.ma.MaskedArray
        """
        return np.ma.MaskedArray(data_set, ~self.order_mask, fill_value=fill_value)


    def _update_mask(self, mask):
        """
        Setting a new mask in the masked arrays in both this class and the wcs
        class
        Parameters
        ----------
        mask : numpy.ndarray
        """
        updated_mask = self.data.mask | mask

        self.data.mask = updated_mask
        self.uncertainty.mask = updated_mask
        self.flags.mask = updated_mask
        self.wcs._update_mask(updated_mask)



    def enable_flags_as_mask(self):
        flags_mask = (self.flags.filled() == 0) & self.order_mask
        self._update_mask(~flags_mask)

    def enable_instrument_model_masking(self):
        instrument_model_mask = ((self.wcs.pix_to_wave_ma.data > 0.0) &
                                 self.order_mask)
        self._update_mask(~instrument_model_mask)

    def standard_masking(self):
        """
        Flags everything with QUAL flag != 0 and everything less
        than 0.

        """

        self.enable_flags_as_mask()
        self.enable_instrument_model_masking()

        self._update_mask(~((self.data.filled() > 0) & self.order_mask))


    def __repr__(self):
        repr_str = "<Order {num} {dataset} {arm}>".format(
            num=self.order_id, dataset=self.data_set_id, arm=self.instrument_arm)
        return repr_str


class MultiOrder(Order):

    @classmethod
    def from_xshooter_data(cls, xshooter_data):
        """
        Construct an xtool.model.Order instance from the an XShooter Data
        Parameters
        ----------
        xshooter_data : xtool.data.XShooterData
            input X-shooter data
        order_id : int
            absolute order ID

        Returns
        -------
            : Order
            order object
        """

        mask = xshooter_data.get_mask(xshooter_data.science_data)
        mask[xshooter_data.transform_pix_to_wave == 0.0] = False


        order_wcs = LUTOrderWCS(
            xshooter_data.transform_pix_to_wave,
            xshooter_data.transform_pix_to_slit, mask)

        return cls(xshooter_data.science_data, xshooter_data.uncertainty,
                   xshooter_data.flags, mask, None,
                   xshooter_data.instrument_arm,
                   xshooter_data.science_header['ARCFILE'], order_wcs)

xtool_data_path = os.path.join(xtool.__path__[0], 'data')