class InstrumentModel(object):
    pass

class XShooterInstrumentModel(InstrumentModel):

    data_fname = 'SCI_SLIT_DIVFF_NIR.fits'

    transform_pixel_to_wave_fname = 'SCI_SLIT_WAVE_MAP_NIR.fits'

    def read_reduced_data(self, data_directory):

        pass


class PolynomialXShooterInstrumentModel(XShooterInstrumentModel):
    pass
