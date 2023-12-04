import gdown
import os
from loguru import logger
import zipfile

def clone_data():
    gg_drive_url = 'https://drive.google.com/uc?id=1jYxSQV5QVDTalFDbVSDBr1I4SPB-FTVb'
    output = 'data.zip'

    if (os.path.exists('./data')):
        logger.info('Data is already found in ./data, no action is needed.')
        return

    if (not os.path.exists(output)):
        logger.info('Downloading data from Google Drive...')
        gdown.download(gg_drive_url, output, quiet=False)
        logger.info('Data is downloaded.')
    else:
        logger.info('Data is already downloaded.')

    logger.info('Extracting data...')
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()
    logger.info('Data is extracted into ./data')
    os.remove(output)
clone_data()