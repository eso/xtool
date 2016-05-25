Generating Data for XTool
=========================

(provided by Sabine Möhler)

1. download science data with associated raw calibrations
    (UVB XSHOO.2013-03-04T07:06:20.957, VIS XSHOO.2013-03-04T07:06:26.098)
2. process with reflex as STARE
3. look for SCI_SLIT_DIVFF, SCI_SLIT_WAVE_MAP, SCI_SLIT_SLIT_MAP in
    <reflex_data_dir>/reflex_tmp_products/xshooter/xsh_scired_slit_stare_1/<date-time>
4. the slit wavelength calibration raw data are delivered together with
    the other raw data, but their processing is not supported by reflex. I
    used the calSelector version on my disk to get the master calibrations
    used by QC to process these data and processed them manually. This  way
    I got the "ON" frame in the correct orientation.
