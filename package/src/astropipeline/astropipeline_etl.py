import pandas as pd
import requests
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

natroot = "https://astroarchive.noirlab.edu"
baseurl = f"{natroot}/api/sia"
adsurl = f"{natroot}/api/adv_search"


class PipeStudy:
    """
    PipeStudy is a class for storing top-level details of astronomy pipeline
        results.

    Attributes:
        telescope (str): Name of telescope used for capturing images.
        instrument (str): Name of instrument used for capturing images.
        filter (str): Name of filter used for capturing images.
        exposure (float): Exposure time in seconds for images.
    """
    telescope = ""
    instrument = ""
    filter = ""
    exposure = ""

    def __init__(self, telescope, instrument, exposure, filter,
                 max_returns=10):
        # Method to instantiate PipeStudy object
        self.telescope = telescope
        self.instrument = instrument
        self.exposure = exposure
        self.filter = filter
        self.index = 0
        self.max_returns = max_returns

    def find_instcals(self):
        # Method to find images that resulted from a processing pipeline
        jj = {
            "outfields": [
                "telescope",
                "md5sum",
                "archive_filename",
                "original_filename",
                "instrument",
                "proc_type",
                "prod_type",
                "ifilter",
                "obs_type",
                "release_date",
                "proposal",
                "caldat",
                "EXPTIME",
                "AIRMASS",
                "CORN1DEC",
                "CORN2DEC",
                "CORN3DEC",
                "CORN4DEC",
                "CORN1RA",
                "CORN2RA",
                "CORN3RA",
                "CORN4RA",
                "dec_max",
                "OBJECT",
                "SEQID",
                "url",
            ],
            "search": [
                ["telescope", self.telescope],
                ["instrument", self.instrument],
                ["proc_type", "instcal"],
                ["prod_type", "image"],
                ["ifilter", self.filter],
                ["exposure", self.exposure, self.exposure],
            ],
        }
        # complete the search and return results as pandas dataframe
        instcal_fits_df = pd.DataFrame(
            requests.post(
                f"{adsurl}/find/?limit={self.max_returns*10}", json=jj
            ).json()[1:]
        )

        # reduce df by removing skyflats (sflats) then only taking the
        #  requested amount
        instcal_fits_df = (
            instcal_fits_df[
                ~instcal_fits_df["OBJECT"].str.contains("sflat",
                                                        case=False,
                                                        na="True")
            ]
            .head(self.max_returns)
            .reset_index(drop=True)
        )

        # initialize the fields for the output path and pipeline file path
        instcal_fits_df["out_path"] = None
        instcal_fits_df["pipe_path"] = None

        # store the number of collected instant calibration images as an
        #   attribute of the PipeStudy object
        self.num_instcals = len(instcal_fits_df)

        return instcal_fits_df

    def __iter__(self):
        return self  # Return the iterator object itself

    def __next__(self):
        if self.index < len(self.instcal_fits_df):
            result = self.instcal_fits_df.iloc[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration  # Signal end of iteration


class PipeFilePaths:
    """
    PipeFilePaths builds and stores strings for pipeline data paths to
        the user's local directory.

    Attributes:
        raw_row (pandas dataframe series): Single row from pipeline dataframe.
        output_folder (str): Directory for storing the pipeline dataframe and
                                the re-processed FITS file.
        local_fits_path (str): Filepath for the re-processed FITS file.
        pipe_file_path (str): Filepath for the dataframe (saved as a CSV) that
                                details which files were used in the
                                reprocessing.
    """
    def __init__(self, raw_row, output_folder, study_name):
        self.raw_url = raw_row["url"].iloc[0]
        raw_filename = raw_row["archive_filename"].iloc[0]
        self.local_fits_path = output_folder + \
            raw_filename.split("/")[-1].replace("ori", study_name)
        self.pipe_file_path = output_folder + \
            raw_filename.split("/")[-1].replace("ori.fits.fz",
                                                study_name + "_pipe.csv")


def find_precal_match(pipe_study_row, product_name):
    """
    Given a processed ("instcal") image, find_precal_match retrieves raw and
    calibration files associated with the processed image that can be used in
    a processing pipeline.

    Args:
        pipe_study_row (pandas dataframe series): Single row from pipeline
                                                    study.
        product_name (str): The name of the specific type of product/file to
                                find (examples: "dark", "flat")

    Returns:
        results_df (pandas dataframe): Dataframe storing all data products
        that are related to the pipeline process for the specific processed
        image from the pipe_study_row.
    """
    hdu = fits.open(pipe_study_row["url"])
    hdr = hdu[0].header
    this_instr = hdr["INSTRUME"]
    this_rawfile = hdr["RAWFILE"]
    this_propid = hdr["PROPID"]
    this_caldat = hdr["DTCALDAT"]
    this_filter = hdr["FILTER"]
    this_dqmask = hdu[1].header["DQMASK"].split("[")[0]

    match product_name:
        case "raw":
            searchParameters = [
                ["instrument", this_instr],
                ["proc_type", "raw"],
                ["prod_type", "image"],
                ["obs_type", "object"],
                ["ifilter", this_filter],
                ["original_filename", this_rawfile, "contains"],
            ]
        case "flat":
            searchParameters = [
                ["instrument", this_instr],
                ["proc_type", "raw"],
                ["PROPID", this_propid],
                ["prod_type", "image"],
                ["ifilter", this_filter],
                ["obs_type", "flat", "dome flat"],
                ["caldat", this_caldat, this_caldat],
            ]
        case "dark":
            searchParameters = [
                ["instrument", this_instr],
                ["proc_type", "raw"],
                ["PROPID", this_propid],
                ["prod_type", "image"],
                ["obs_type", "dark"],
                ["caldat", this_caldat, this_caldat],
            ]
        case "dqmask":
            searchParameters = [
                ["instrument", this_instr],
                ["proc_type", "instcal"],
                ["PROPID", this_propid],
                ["FILENAME", this_dqmask],
                ["prod_type", "dqmask"],
                ["obs_type", "object"],
                ["caldat", this_caldat, this_caldat],
            ]

    jj = {
        "outfields": [
            "telescope",
            "md5sum",
            "archive_filename",
            "original_filename",
            "instrument",
            "proc_type",
            "prod_type",
            "ifilter",
            "obs_type",
            "release_date",
            "proposal",
            "caldat",
            "DATE-OBS",
            "EXPTIME",
            "AIRMASS",
            "ra_min",
            "ra_max",
            "dec_min",
            "dec_max",
            "OBJECT",
            "SEQID",
            "url",
        ],
        "search": searchParameters,
    }

    result_limit = 200
    request_response = requests.post(
        f"{adsurl}/find/?limit={result_limit}", json=jj
    ).json()

    if not isinstance(request_response, list):
        print(request_response["errorMessage"])
        return -1
    else:
        results_df = pd.DataFrame(request_response[1:])

    if len(results_df) == 0:
        print("No " + product_name + "File(s) Found")
        return -1
    elif product_name == ("raw", "dqmask") and len(results_df) > 1:
        print("Ambiguous " + product_name + " Files Found (more than 1)")
        return -1
    else:
        return results_df


def get_pipeline_df(pipe_study_row):
    """
    Given a processed ("instcal") image, build a dataframe where each row
    represents a different file used in the processing.

    Args:
        pipe_study_row (pandas dataframe series): Single row from pipeline
                                                    study.

    Returns:
        pipe_df (pandas dataframe): Pipeline data frame storing details about
        which files were used to create the processed version of the image.
    """
    pipe_df = find_precal_match(pipe_study_row, "raw")

    if isinstance(pipe_df, pd.DataFrame):
        for prod_type in ("dark", "flat", "dqmask"):
            prod_df = find_precal_match(pipe_study_row, prod_type)
            if isinstance(prod_df, pd.DataFrame):
                pipe_df = pd.concat([pipe_df, prod_df], ignore_index=True)
            else:
                return -1

        return pipe_df


def query_2mass(center_ra, center_dec, width_ra, width_dec):
    """
    Given a pair of FK5 sky coordinates and bidirectional angular subtense,
    query the 2MASS catalog for objects within the field of regard.

    Args:
        center_ra (float): Right Ascension angle in degrees.
        center_dec (float): Declination angle in degrees.
        width_ra (float): Right Ascension field of view (degrees).
        width_dec (float): Declination field of view (degrees).

    Returns:
        stars_df (pandas dataframe): Dataframe of stars and their coordinates.
    """
    vizier = Vizier(columns=["*", "Ksnr"], catalog="II/246")
    result = vizier.query_region(
            SkyCoord(ra=center_ra, dec=center_dec, frame="fk5", unit="deg"),
            width=width_ra * u.deg,
            height=width_dec * u.deg,
            )

    # this name is the designation for 2MASS in Vizier
    stars_df = result["II/246/out"].to_pandas()

    # convert the RA and Dec names for easier typing
    stars_df = stars_df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})

    # only return significantly contrasting stars
    stars_df = stars_df[stars_df["Ksnr"] > 3]

    return stars_df


def get_catalog_stars(pipeline_inst, catalog="SIMBAD"):
    """
    Given a pipeline for processing a raw image, query the
    specified catalog for objects within the image's field of regard.

    Args:
        pipeline_inst (pandas data series): A single row of a PipeStudy
                                                instcal_df.
        catalog (str): Name of the star catalog or database to use.

    Returns:
        stars_df (pandas dataframe): Dataframe of stars and their coordinates.
    """
    corner_ras = (
        pipeline_inst.CORN1RA,
        pipeline_inst.CORN2RA,
        pipeline_inst.CORN3RA,
        pipeline_inst.CORN4RA,
    )
    corner_decs = (
        pipeline_inst.CORN1DEC,
        pipeline_inst.CORN2DEC,
        pipeline_inst.CORN3DEC,
        pipeline_inst.CORN4DEC,
    )

    min_ra = min(corner_ras)
    max_ra = max(corner_ras)

    min_dec = min(corner_decs)
    max_dec = max(corner_decs)

    width_ra = max_ra - min_ra
    width_dec = max_dec - min_dec

    center_ra = (width_ra) / 2 + min_ra
    center_dec = (width_dec) / 2 + min_dec

    if catalog == "SIMBAD":
        sql = """SELECT TOP 50 oid, main_id, ra, dec
                    FROM basic
                    WHERE CONTAINS(POINT('ICRS', ra, dec), BOX('ICRS', {0},
                                                        {1}, {2}, {3})) = 1
                    AND ra IS NOT NULL
                    AND dec IS NOT NULL;
            """.format(
            center_ra, center_dec, width_ra, width_dec
        )

        stars_df = Simbad.query_tap(query=sql, maxrec=10000).to_pandas()

    elif catalog == "2MASS":

        stars_df = query_2mass(center_ra, center_dec, width_ra, width_dec)

    return stars_df


def get_study_file(study_path):
    """
    Load a PipeStudy from a saved CSV

    Args:
        study_path (str): The filepath to a saved pipeline study
                            stored as a CSV.

    Returns:
        study_df (pandas dataframe): Dataframe where each row represents a
                                        single processed image.
    """
    study_df = pd.read_csv(study_path)

    return study_df


def get_pipeline_file(pipeline_path):
    """
    Load a Pipeline Dataframe (pipe_df) from a saved CSV

    Args:
        pipeline_path (str): The filepath to a saved pipeline instance
                                stored as a CSV.

    Returns:
        pipeline_df (pandas dataframe): Dataframe where each row represents a
                                        file used to reprocess a raw image.
    """
    pipeline_df = pd.read_csv(pipeline_path)

    return pipeline_df
