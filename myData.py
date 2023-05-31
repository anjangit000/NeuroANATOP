import os
import numpy as np
import collections.abc
import contextlib
import fnmatch
import hashlib
import pickle
import shutil
import time
import sys
import tarfile
import urllib
import warnings
import zipfile
import json
import requests
from sklearn.utils import Bunch, deprecated

def md5_hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def _format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with open(path, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def _read_md5_sum_file(path):
    """ Reads a MD5 checksum file and returns hashes as a dictionary.
    """
    with open(path, "r") as f:
        hashes = {}
        while True:
            line = f.readline()
            if not line:
                break
            h, name = line.rstrip().split('  ', 1)
            hashes[name] = h
    return hashes


def readlinkabs(link):
    """
    Return an absolute path for the destination
    of a symlink
    """
    path = os.readlink(link)
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(link), path)
    
def get_data_dirs(data_dir=None):
    """Returns the directories in which nilearn looks for data.
    This is typically useful for the end-user to check where the data is
    downloaded and stored.
    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    Returns
    -------
    paths : list of strings
        Paths of the dataset directories.
    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder
    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(data_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv('NILEARN_SHARED_DATA')
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv('NILEARN_DATA')
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        #paths.append(os.path.expanduser('~/nilearn_data')) F:\Research\Dat_new
        paths.append(os.path.expanduser('~/nilearn_data/'))
    return paths


def _get_dataset_dir(dataset_name, data_dir=None, default_paths=None, verbose=1):
    """Creates if necessary and returns data directory of given dataset.
    Parameters
    ----------
    dataset_name : string
        The unique name of the dataset.
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    default_paths : list of string, optional
        Default system paths in which the dataset may already have been
        installed by a third party software. They will be checked first.
    verbose : int, optional
        Verbosity level (0 means no message). Default=1.
    Returns
    -------
    data_dir : string
        Path of the given dataset directory.
    Notes
    -----
    This function retrieves the datasets directory (or data directory) using
    the following priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder
    """
    paths = []
    # Search possible data-specific system paths
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(os.pathsep)])

    paths.extend([(d, False) for d in get_data_dirs(data_dir=data_dir)])

    if verbose > 2:
        print('Dataset search paths: %s' % paths)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if os.path.islink(path):
            # Resolve path
            path = readlinkabs(path)
        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print('\nDataset found in %s\n' % path)
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print('\nDataset created in %s\n' % path)
                return path
            except Exception as exc:
                short_error_message = getattr(exc, 'strerror', str(exc))
                errors.append('\n -{0} ({1})'.format(
                    path, short_error_message))

    raise OSError('Nilearn tried to store the dataset in the following '
                  'directories, but:' + ''.join(errors))

def _get_dataset_descr(ds_name):
    module_path = os.path.dirname(os.path.abspath(__file__))

    fname = ds_name

    try:
        with open(os.path.join(module_path, 'description', fname + '.rst'),
                  'rb') as rst_file:
            descr = rst_file.read()
    except IOError:
        descr = ''

    if descr == '':
        print("Warning: Could not find dataset description.")

    return descr

def movetree(src, dst):
    """Move an entire tree to another directory. Any existing file is
    overwritten"""
    names = os.listdir(src)

    # Create destination dir if it does not exist
    if not os.path.exists(dst):
        os.makedirs(dst)
    errors = []

    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname) and os.path.isdir(dstname):
                movetree(srcname, dstname)
                os.rmdir(srcname)
            else:
                shutil.move(srcname, dstname)
        except (IOError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive movetree so that we can
        # continue with other files
        except Exception as err:
            errors.extend(err.args[0])
    if errors:
        raise Exception(errors)

def _fetch_files(data_dir, files, resume=True, verbose=1, session=None):
    """Load requested dataset, downloading it if needed or requested.
    This function retrieves files from the hard drive or download them from
    the given urls. Note to developpers: All the files will be first
    downloaded in a sandbox and, if everything goes well, they will be moved
    into the folder of the dataset. This prevents corrupting previously
    downloaded data. In case of a big dataset, do not hesitate to make several
    calls if needed.
    Parameters
    ----------
    data_dir : string
        Path of the data directory. Used for data storage in a specified
        location.
    files : list of (string, string, dict)
        List of files and their corresponding url with dictionary that contains
        options regarding the files. Eg. (file_path, url, opt). If a file_path
        is not found in data_dir, as in data_dir/file_path the download will
        be immediately cancelled and any downloaded files will be deleted.
        Options supported are:
            * 'move' if renaming the file or moving it to a subfolder is needed
            * 'uncompress' to indicate that the file is an archive
            * 'md5sum' to check the md5 sum of the file
            * 'overwrite' if the file should be re-downloaded even if it exists
    resume : bool, optional
        If true, try resuming download if possible. Default=True.
    verbose : int, optional
        Verbosity level (0 means no message). Default=1.
    session : `requests.Session`, optional
        Session to use to send requests.
    Returns
    -------
    files : list of string
        Absolute paths of downloaded files on disk.
    """
    if session is None:
        with requests.Session() as session:
            session.mount("ftp:", _NaiveFTPAdapter())
            return _fetch_files(
                data_dir, files, resume=resume,
                verbose=verbose, session=session)
    # There are two working directories here:
    # - data_dir is the destination directory of the dataset
    # - temp_dir is a temporary directory dedicated to this fetching call. All
    #   files that must be downloaded will be in this directory. If a corrupted
    #   file is found, or a file is missing, this working directory will be
    #   deleted.
    files = list(files)
    files_pickle = pickle.dumps([(file_, url) for file_, url, _ in files])
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    # Create destination dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Abortion flag, in case of error
    abort = None

    files_ = []
    for file_, url, opts in files:
        # 3 possibilities:
        # - the file exists in data_dir, nothing to do.
        # - the file does not exists: we download it in temp_dir
        # - the file exists in temp_dir: this can happen if an archive has been
        #   downloaded. There is nothing to do

        # Target file in the data_dir
        target_file = os.path.join(data_dir, file_)
        # Target file in temp dir
        temp_target_file = os.path.join(temp_dir, file_)
        # Whether to keep existing files
        overwrite = opts.get('overwrite', False)
        if (abort is None and (overwrite or (not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)))):

            # We may be in a global read-only repository. If so, we cannot
            # download files.
            if not os.access(data_dir, os.W_OK):
                raise ValueError('Dataset files are missing but dataset'
                                 ' repository is read-only. Contact your data'
                                 ' administrator to solve the problem')

            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            md5sum = opts.get('md5sum', None)

            dl_file = _fetch_file(url, temp_dir, resume=resume,
                                  verbose=verbose, md5sum=md5sum,
                                  username=opts.get('username', None),
                                  password=opts.get('password', None),
                                  session=session, overwrite=overwrite)
            if 'move' in opts:
                # XXX: here, move is supposed to be a dir, it can be a name
                move = os.path.join(temp_dir, opts['move'])
                move_dir = os.path.dirname(move)
                if not os.path.exists(move_dir):
                    os.makedirs(move_dir)
                shutil.move(dl_file, move)
                dl_file = move
            if 'uncompress' in opts:
                try:
                    _uncompress_file(dl_file, verbose=verbose)
                except Exception as e:
                    abort = str(e)

        if (abort is None and not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):
            warnings.warn('An error occured while fetching %s' % file_)
            abort = ("Dataset has been downloaded but requested file was "
                     "not provided:\nURL: %s\n"
                     "Target file: %s\nDownloaded: %s" %
                     (url, target_file, dl_file))
        if abort is not None:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise IOError('Fetching aborted: ' + abort)
        files_.append(target_file)
    # If needed, move files from temps directory to final directory.
    if os.path.exists(temp_dir):
        # XXX We could only moved the files requested
        # XXX Movetree can go wrong
        movetree(temp_dir, data_dir)
        shutil.rmtree(temp_dir)
    return files_


class _NaiveFTPAdapter(requests.adapters.BaseAdapter):
    def send(self, request, timeout=None, **kwargs):
        try:
            timeout, _ = timeout
        except Exception:
            pass
        try:
            data = urllib.request.urlopen(request.url, timeout=timeout)
        except Exception as e:
            raise requests.RequestException(e.reason)
        data.release_conn = data.close
        resp = requests.Response()
        resp.url = data.geturl()
        resp.status_code = data.getcode() or 200
        resp.raw = data
        resp.headers = dict(data.info().items())
        return resp

    def close(self):
        pass

def _fetch_file(url, data_dir, resume=True, overwrite=False,
                md5sum=None, username=None, password=None,
                verbose=1, session=None):
    """Load requested file, downloading it if needed or requested.
    Parameters
    ----------
    url : string
        Contains the url of the file to be downloaded.
    data_dir : string
        Path of the data directory. Used for data storage in the specified
        location.
    resume : bool, optional
        If true, try to resume partially downloaded files.
        Default=True.
    overwrite : bool, optional
        If true and file already exists, delete it. Default=False.
    md5sum : string, optional
        MD5 sum of the file. Checked if download of the file is required.
    username : string, optional
        Username used for basic HTTP authentication.
    password : string, optional
        Password used for basic HTTP authentication.
    verbose : int, optional
        Verbosity level (0 means no message). Default=1.
    session : requests.Session, optional
        Session to use to send requests.
    Returns
    -------
    files : string
        Absolute path of downloaded file.
    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.
    """
    if session is None:
        with requests.Session() as session:
            session.mount("ftp:", _NaiveFTPAdapter())
            return _fetch_file(
                url, data_dir, resume=resume, overwrite=overwrite,
                md5sum=md5sum, username=username, password=password,
                verbose=verbose, session=session)
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Determine filename using URL
    parse = urllib.parse.urlparse(url)
    file_name = os.path.basename(parse.path)
    if file_name == '':
        file_name = md5_hash(parse.path)

    temp_file_name = file_name + ".part"
    full_name = os.path.join(data_dir, file_name)
    temp_full_name = os.path.join(data_dir, temp_file_name)
    if os.path.exists(full_name):
        if overwrite:
            os.remove(full_name)
        else:
            return full_name
    if os.path.exists(temp_full_name):
        if overwrite:
            os.remove(temp_full_name)
    t0 = time.time()
    local_file = None
    initial_size = 0

    try:
        # Download data
        headers = {}
        auth = None
        if username is not None and password is not None:
            if not url.startswith('https'):
                raise ValueError(
                    'Authentication was requested on a non  secured URL (%s).'
                    'Request has been blocked for security reasons.' % url)
            auth = (username, password)
        if verbose > 0:
            displayed_url = url.split('?')[0] if verbose == 1 else url
            print('Downloading data from %s ...' % displayed_url)
        if resume and os.path.exists(temp_full_name):
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_full_name)
            # If the file exists, then only download the remainder
            headers["Range"] = "bytes={}-".format(local_file_size)
            try:
                req = requests.Request(
                    method="GET", url=url, headers=headers, auth=auth)
                prepped = session.prepare_request(req)
                with session.send(prepped, stream=True,
                                  timeout=_REQUESTS_TIMEOUT) as resp:
                    resp.raise_for_status()
                    content_range = resp.headers.get('Content-Range')
                    if (content_range is None or not content_range.startswith(
                            'bytes {}-'.format(local_file_size))):
                        raise IOError('Server does not support resuming')
                    initial_size = local_file_size
                    with open(local_file, "ab") as fh:
                        _chunk_read_(
                            resp, fh, report_hook=(verbose > 0),
                            initial_size=initial_size, verbose=verbose)
            except Exception:
                if verbose > 0:
                    print('Resuming failed, try to download the whole file.')
                return _fetch_file(
                    url, data_dir, resume=False, overwrite=overwrite,
                    md5sum=md5sum, username=username, password=password,
                    verbose=verbose, session=session)
        else:
            req = requests.Request(
                method="GET", url=url, headers=headers, auth=auth)
            prepped = session.prepare_request(req)
            with session.send(
                    prepped, stream=True, timeout=_REQUESTS_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(temp_full_name, "wb") as fh:
                    _chunk_read_(resp, fh, report_hook=(verbose > 0),
                                 initial_size=initial_size, verbose=verbose)
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0
        if verbose > 0:
            # Complete the reporting hook
            sys.stderr.write(' ...done. ({0:.0f} seconds, {1:.0f} min)\n'
                             .format(dt, dt // 60))
    except (requests.RequestException):
        sys.stderr.write("Error while fetching file %s; dataset "
                         "fetching aborted." % (file_name))
        raise
    if md5sum is not None:
        if (_md5_sum_file(full_name) != md5sum):
            raise ValueError("File %s checksum verification has failed."
                             " Dataset fetching aborted." % local_file)
    return full_name

def fetch_adhd_data(n_subjects=None, data_dir=None, url=None, resume=True, verbose=1, university=None):
    print('&&&&&&&&&')
    dataset_name = 'adhd'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,  verbose=verbose)
    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)
    # First, get the metadata
    opts = dict(uncompress=True)
    phenotypic = ('ADHD200_40subs_motion_parameters_and_phenotypics.csv','', opts)
    print(data_dir)
    phenotypic = _fetch_files(data_dir, [phenotypic], resume=resume, verbose=verbose)[0]
    # Load the full data from the csv file
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',', dtype=None)
    #filter out the particular university subjects
    if n_subjects is None:
        n_subjects = len(phenotypic['site']) 
    if university == '"All"' or university == None:
        filtered_phenotypic = phenotypic if n_subjects is None else  phenotypic[:n_subjects]
    else:
        filtered_phenotypic = phenotypic[[i for i in range(len(phenotypic['site'])) if phenotypic['site'][i].decode("utf-8").split('_')[0]==university]]
        
    print(university, n_subjects, phenotypic[n_subjects-1])
    #collect the subject ids
    ids = filtered_phenotypic['Subject']
    #collect the functional names from the ids
    functionals = ['data/%s/%s_rest_tshift_RPI_voreg_mni.nii.gz' %(i,i) for i in ids]
    confounds = ['data/%s/%s_regressors.csv' % (i, i) for i in ids]
    archives = [url for ii in ids]
    print(functionals)
    functionals = _fetch_files(
        data_dir, zip(functionals, archives, (opts,) * n_subjects),
        resume=resume, verbose=verbose)
    print(functionals)
    confounds = _fetch_files(
        data_dir, zip(confounds, archives, (opts,) * n_subjects),
        resume=resume, verbose=verbose)
    print('>>>',len(ids))
    return Bunch(func=functionals, phenotypic=phenotypic, confounds=confounds)
    
def fetch_adni_data(n_subjects=None, data_dir=None, url=None, resume=True, verbose=1, research_group='AD'):
    print('$$$$$$$$$$$$$')
    dataset_name = 'alzhimer'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,  verbose=verbose)
    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)
    # First, get the metadata
    opts = dict(uncompress=True)
    phenotypic = ('phenotypic_info.csv','', opts)
    print(data_dir, phenotypic)
    phenotypic = _fetch_files(data_dir, [phenotypic], resume=resume, verbose=verbose)[0]
    # Load the full data from the csv file
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',', dtype=None)
    '''
    #filter out the particular university subjects
    print('#############')
    if n_subjects is None:
        n_subjects = len(phenotypic['RESEARCH_GROUP']) 
    if university == '"All"' or university == None:
        filtered_phenotypic = phenotypic if n_subjects is None else  phenotypic[:n_subjects]
    else:
        filtered_phenotypic = phenotypic[[i for i in range(len(phenotypic['RESEARCH_GROUP'])) if phenotypic['RESEARCH_GROUP'][i].decode("utf-8").split('_')[0]==university]]
        
    print(university, n_subjects, phenotypic[n_subjects-1])
    '''
    #filter out the particular research groups
    phenotypic = phenotypic[[i for i in range(len(phenotypic['RESEARCH_GROUP'])) if phenotypic['RESEARCH_GROUP'][i].decode("utf-8").split('_')[0]==research_group]]
    print(phenotypic)
    #collect the subject ids
    ids = phenotypic['DPARSF_SUB_ID']
    #collect the functional names from the ids
    functionals = ['/home/swarup/nilearn_data/alzhimer/data/'+research_group+'/%s/Filtered_4DVolume.nii' %i.decode("utf-8") for i in ids]
    #confounds = ['data/%s/%s_regressors.csv' % (i, i) for i in ids]
    archives = [url for ii in ids]
    print(functionals)
    #functionals = _fetch_files(data_dir, zip(functionals, archives, (opts,) * n_subjects), resume=resume, verbose=verbose)
    #confounds = _fetch_files(data_dir, zip(confounds, archives, (opts,) * n_subjects), resume=resume, verbose=verbose)
    print('>>>',len(ids))
    return Bunch(func=functionals, phenotypic=phenotypic)

