import os
import struct
import errno
from .const import CONSOLE_STRING
from .const import ALPHABET
import pickle
import numpy as np
import h5py


console_string = 'cosmoconvert'
verbose = False


def check_and_create_path(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_analysis_object(data, data_file, force_overwrite = False):
    check_and_create_path(os.path.dirname(data_file))
    
    if not force_overwrite:
        if os.path.isfile(data_file):
            info('%s already exists! Skipping.' % data_file)
            return

    with open(data_file, 'wb') as file_handle:
        pickle.dump(data, file_handle, pickle.HIGHEST_PROTOCOL)


def info(output, final = False):
    global verbose
    if verbose:
        global console_string

        print('[%s] %s' % (console_string, output))
        
        if final:
            console_string = CONSOLE_STRING


# This is for GADGET IO operations
def gadget_write_dummy(f, values_list):
    for i in values_list:
        dummy = [i]
        f.write(struct.pack('i', *dummy))


def gadget_write_block(f, block_data, data_type, block_name, gfmt = 2, 
                        shape = False):
    if shape:
        block_data.shape = (1, -1)
        block_data = block_data[0]

    if gfmt == 1:
        if block_name == 'HEAD':
            assert len(block_data) == 256

            gadget_write_dummy(f, [256])
            f.write(block_data)
            gadget_write_dummy(f, [256])
        else:
            fmt = data_type * len(block_data)
            nbytes = len(block_data) * 4

            gadget_write_dummy(f, [nbytes])
            f.write(struct.pack(fmt, *block_data))
            gadget_write_dummy(f, [nbytes])
    else:
        gadget_write_dummy(f, [8])
        f.write(struct.pack('c' * 4, *block_name))
    
        if block_name == 'HEAD':
            nbytes = 256
        else:
            fmt = data_type * len(block_data)
            nbytes = len(block_data) * 4
        
        gadget_write_dummy(f, [nbytes + 8, 8, nbytes]) 
    
        if block_name == 'HEAD':
            f.write(block_data)
        else:
            f.write(struct.pack(fmt, *block_data))
    
        gadget_write_dummy(f, [nbytes])

    f.flush()
    os.fsync(f.fileno())


def read_tipsy_aux(file_name, ngas, nstars, gas_str, star_str, byte_swap = False):
    """
    This function reads a TIPSY auxiliary file with the specified format.
    A common format is below:

        struct aux_gas_data
        {
          float metal[NMETALS_TIPSY];
          float sfr;
          float tmax;
          float delaytime;
          float ne;
          float nh;
          int nspawn;
          short int nrec;
        };


        struct aux_star_data
        {
          float metal[NMETALS_TIPSY];
          float age;
          float tmax;
          int nspawn;
          short int nrec;
        };


    This function is very memory intensive, I had to break apart some
    of the reshaping in order to minimize the damage.

    Specify the struct format using gas_str and star_str, they change frequently.

    :param file_name: Auxiliary file
    :param ngas: Number of gas particles at this timestep
    :param nstars: Number of star particles at this timestep
    :param num_metals: Number of metals in the simulation.
    :return: A tuple (gas_data, star_data) in the form of numpy arrays.
    """
    gas_dtypes_list = []
    labels = iter(ALPHABET)
    for char in gas_str:
        if byte_swap:
            char = '>' + char

        gas_dtypes_list.append((next(labels), char))

    gas_dtypes = np.dtype(gas_dtypes_list)

    star_dtypes_list = []
    labels = iter(ALPHABET)
    for char in star_str:
        if byte_swap:
            char = '>' + char

        star_dtypes_list.append((next(labels), char))

    star_dtypes = np.dtype(star_dtypes_list)

    with open(file_name, 'rb') as f:
        gas_data = np.fromfile(f, dtype = gas_dtypes, count = ngas)

    with open(file_name, 'rb') as f:
        # Need to go to the start of the star records
        f.seek(struct.calcsize(gas_str) * ngas)

        star_data = np.fromfile(f, dtype = star_dtypes, count = nstars)

    return gas_data, star_data


def read_tipsy(file_name, header_only = False, has_pad = True):
    """
     struct gas_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real rho;
        Real temp;
        Real hsmooth;
        Real metals ;
        Real phi ;
    } ;

     struct dark_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real eps;
        Real phi ;
    } ;

     struct star_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real metals ;
        Real tform ;
        Real eps;
        Real phi ;
    } ;

     struct dump {
        double time ;
        int nbodies ;
        int ndim ;
        int nsph ;
        int ndark ;
        int nstar ;
        int pad;
    } ;

    :param file_name:
    :param header_only:
    :param has_pad:
    :return:
    """
    byte_swap = False
    head_str = 'diiiiii'
    float_str = 'f'

    # How many values does each standard structure have?
    std_gas_vals = 12
    std_dark_vals = 9
    std_star_vals = 11

    if not has_pad:
        head_str = 'diiiii'

    # This has to come after all modifications to the header string
    head_bytes = struct.calcsize(head_str)

    f = open(file_name, 'rb')

    if has_pad:
        time, total_num_part, ndim, ngas, ndark, nstar, pad = struct.unpack(head_str, f.read(head_bytes))
    else:
        time, total_num_part, ndim, ngas, ndark, nstar = struct.unpack(head_str, f.read(head_bytes))

    if ndim > 3 or ndim < 1:
        f.seek(0)
        head_str = '>' + head_str

        if has_pad:
            time, total_num_part, ndim, ngas, ndark, nstar, pad = struct.unpack(head_str, f.read(head_bytes))
        else:
            time, total_num_part, ndim, ngas, ndark, nstar = struct.unpack(head_str, f.read(head_bytes))

        byte_swap = True

    if header_only:
        f.close()
        return {'time':             time,
                'total_num_part':   total_num_part,
                'ndim':             ndim,
                'ngas':             ngas,
                'ndark':            ndark,
                'nstar':            nstar}

    if byte_swap:
        float_str = '>f'

    gas_dtypes_list = []
    labels = iter(ALPHABET)
    for i in range(std_gas_vals):
        gas_dtypes_list.append((next(labels), float_str))

    gas_dtypes = np.dtype(gas_dtypes_list)

    dark_dtypes_list = []
    labels = iter(ALPHABET)
    for i in range(std_dark_vals):
        dark_dtypes_list.append((next(labels), float_str))

    dark_dtypes = np.dtype(dark_dtypes_list)

    star_dtypes_list = []
    labels = iter(ALPHABET)
    for i in range(std_star_vals):
        star_dtypes_list.append((next(labels), float_str))

    star_dtypes = np.dtype(star_dtypes_list)

    f.seek(head_bytes)
    gas_data = np.fromfile(f, dtype = gas_dtypes, count = ngas)

    f.seek(4 * len(gas_dtypes_list) * ngas + head_bytes)
    dark_data = np.fromfile(f, dtype = dark_dtypes, count = ndark)

    f.seek(4 * (len(gas_dtypes_list) * ngas + len(dark_dtypes_list) * ndark) + head_bytes)
    star_data = np.fromfile(f, dtype = star_dtypes, count = nstar)

    f.close()

    return gas_data, dark_data, star_data, byte_swap


def write_swift(file_name, header, data_dict):
    with h5py.File(file_name, 'w') as f:
        h = f.create_group('Header')
        h.attrs['NumPart_ThisFile'] = header['npart']
        h.attrs['NumPart_Total'] = header['npart']
        h.attrs['NumPart_Total_HighWord'] = 0 * header['npart']
        h.attrs['MassTable'] = np.zeros(7)
        h.attrs['NumPartTypes'] = np.array(7)
        h.attrs['Time'] = header['time']
        h.attrs['Redshift'] = header['redshift']
        h.attrs['BoxSize'] = header['box_size']
        h.attrs['Flag_Entropy_ICs'] = header['flag_entropy_ics']

        p = f.create_group('Parameters')
        p.attrs['Cosmology:Omega_b'] = header['omega_baryon']
        p.attrs['Cosmology:Omega_cdm'] = header['omega_matter'] - header['omega_baryon']
        p.attrs['Cosmology:Omega_lambda'] = header['omega_lambda']
        p.attrs['Cosmology:Omega_m'] = header['omega_matter']
        p.attrs['Cosmology:h'] = header['hubble_constant']

        u = f.create_group('Units')
        u.attrs['Unit current in cgs (U_I)'] = header['unit_current']
        u.attrs['Unit length in cgs (U_L)'] = header['unit_length']
        u.attrs['Unit mass in cgs (U_M)'] = header['unit_mass']
        u.attrs['Unit temperature in cgs (U_T)'] = header['unit_temperature']
        u.attrs['Unit time in cgs (U_t)'] = header['unit_time']

        for part_type in data_dict:
            p = f.create_group(part_type)

            for key in data_dict[part_type]:
                p.create_dataset(key, data = data_dict[part_type][key])

def write_gizmo(file_name, header, data_dict):
    with h5py.File(file_name, 'w') as f:
        h = f.create_group('Header')

        h.attrs['NumPart_ThisFile'] = header['npart']
        h.attrs['NumPart_Total'] = header['npart']
        h.attrs['NumPart_Total_HighWord'] = 0 * header['npart']
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = header['time']
        h.attrs['Redshift'] = header['redshift']
        h.attrs['BoxSize'] = header['box_size']
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['Omega0'] = header['omega0']
        h.attrs['OmegaLambda'] = header['omega_lambda']
        h.attrs['HubbleParam'] = header['hubble_constant']
        h.attrs['Flag_Sfr'] = header['flag_sfr']
        h.attrs['Flag_Cooling'] = header['flag_cooling']
        h.attrs['Flag_StellarAge'] = header['flag_stellarage']
        h.attrs['Flag_Metals'] = header['flag_metals']
        h.attrs['Flag_Feedback'] = header['flag_feedback']
        h.attrs['Flag_DoublePrecision'] = header['flag_doubleprecision']
        h.attrs['Flag_IC_Info'] = header['flag_ic_info']

        for part_type in data_dict:
            p = f.create_group(part_type)

            for key in data_dict[part_type]:
                p.create_dataset(key, data = data_dict[part_type][key])


def write_gadget(file_name, header, data_dict):
    with h5py.File(file_name, "w") as f:
        h = f.create_group("Header")

        h.attrs["NumPart_ThisFile"] = header["npart"]
        h.attrs["NumPart_Total"] = header["npart"] & 0xFFFFFFFF  # lower 32 bits
        h.attrs["NumPart_Total_HighWord"] = header["npart"] >> 32  # upper 32 bits
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = header["time"]
        h.attrs["Redshift"] = header["redshift"]
        h.attrs["BoxSize"] = header["box_size"]
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = header["omega0"]
        h.attrs["OmegaLambda"] = header["omega_lambda"]
        h.attrs["HubbleParam"] = header["hubble_constant"]
        h.attrs["Flag_Sfr"] = header["flag_sfr"]
        h.attrs["Flag_Cooling"] = header["flag_cooling"]
        h.attrs["Flag_StellarAge"] = header["flag_stellarage"]
        h.attrs["Flag_Metal"] = header["flag_metals"]
        h.attrs["Flag_Feedback"] = header["flag_feedback"]
        h.attrs["Flag_DoublePrecision"] = header["flag_doubleprecision"]
        h.attrs["Flag_IC_Info"] = header["flag_ic_info"]

        for part_type in data_dict:
            p = f.create_group(part_type)

            for key in data_dict[part_type]:
                p.create_dataset(key, data=data_dict[part_type][key])
