# Define which detector/subarray is going to be fit:
detector = 'SUBSTRIP256'
frequency_filename = 'data/frequencies.npy'
indexes_filename = 'data/indexes.npy'

# Set pixel and jump times (in microseconds):
pixel_time, jump_time = 10., 120.

if detector == 'SUBSTRIP256':

    plot_name = 'NIRISS, SUBSTRIP256'
    psd_filename = 'data/median_power256_otis.npy'
    ncolumns = 2048
    nrows = 256
    ngroups = 199

elif detector == 'SUBSTRIP96':

    plot_name = 'NIRISS, SUBSTRIP96'
    psd_filename = 'data/median_power96_otis.npy'
    ncolumns = 2048
    nrows = 96
    ngroups = 239

elif detector == 'NRS1':

    plot_name = 'NIRSpec, NIS1 (512)'
    psd_filename = 'data/median_NRS1_88.npy'
    ncolumns = 2048
    nrows = 512
    ngroups = 88

elif detector == 'NRS2':

    plot_name = 'NIRSpec, NIS2 (512)'
    psd_filename = 'data/median_NRS2_88.npy'
    ncolumns = 2048
    nrows = 512
    ngroups = 88

elif detector == 'NIRCAM':

    plot_name = 'NIRCam (512)'
    psd_filename = 'data/median_nircam_108.npy'
    ncolumns = 2048
    nrows = 512
    ngroups = 108
