import pickle
from astropy.io import fits
import numpy as np

try:
    hdu = fits.ImageHDU(data=np.ones((10, 10)))
    hdu.header['TEST_KEY'] = 'VAL'
    pickled = pickle.dumps(hdu)
    unpickled = pickle.loads(pickled)
    print("HDU pickle success:", isinstance(unpickled.data, np.ndarray), unpickled.header['TEST_KEY'])
except Exception as e:
    print("HDU pickle failed:", e)

try:
    header = fits.Header()
    header['TEST_KEY'] = 'VAL'
    pickled = pickle.dumps(header)
    unpickled = pickle.loads(pickled)
    print("Header pickle success:", unpickled['TEST_KEY'])
except Exception as e:
    print("Header pickle failed:", e)
