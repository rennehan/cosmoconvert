import numpy as np
from . import util
import pygadgetreader as pygr

class gadget2gizmo(object):
   
    # It is always safer to assume double precision when writing out the HDF5 file 
    def __init__(self, snapshot_file, omega_baryon = 0.0, doubleprecision = True, verbose = True):

        util.io.verbose = verbose

        self.redshift = pygr.readheader(snapshot_file, 'redshift')
        self.hubble_constant = pygr.readheader(snapshot_file, 'h')
        self.box_size = pygr.readheader(snapshot_file, 'boxsize')
        self.ngas = pygr.readheader(snapshot_file, 'gascount')
        self.ndm = pygr.readheader(snapshot_file, 'dmcount')
        self.ndisk = pygr.readheader(snapshot_file, 'diskcount')
        self.nbulge = pygr.readheader(snapshot_file, 'bulgecount')
        self.nstar = pygr.readheader(snapshot_file, 'starcount')
        self.nbh = pygr.readheader(snapshot_file, 'bndrycount')

        # Calculate the necessary units for conversion.
        # The GIZMO units are default:
        # L = 1.0 kpc/h
        # M = 1e10 Msun/h
        # V = 1e5 km/s
        self._set_gadget_units()
        self._set_gizmo_units()

        # This is the standard format for GIZMO files.
        gizmo_dict = dict(PartType0 = {}, PartType1 = {}, PartType2 = {}, PartType3 = {}, PartType4 = {})

        gas_dict = gizmo_dict['PartType0']
        dark_dict = gizmo_dict['PartType1']
        disk_dict = gizmo_dict['PartType2']
        bulge_dict = gizmo_dict['PartType3']
        star_dict = gizmo_dict['PartType4']

        header = {}

        # Add on GIZMO header data
        header['gizmo'] = {}
        header['gizmo']['time'] = pygr.readheader(snapshot_file, 'time')
        header['gizmo']['redshift'] = self.redshift
        header['gizmo']['npart'] = np.array([self.ngas, self.ndm, self.ndisk, self.nbulge, self.nstar, self.nbh])
        header['gizmo']['box_size'] = self.box_size * self.gadget_length / self.gizmo_length
        header['gizmo']['omega0'] = pygr.readheader(snapshot_file, 'O0')
        header['gizmo']['omega_baryon'] = omega_baryon
        header['gizmo']['omega_lambda'] = pygr.readheader(snapshot_file, 'Ol')
        header['gizmo']['hubble_constant'] = self.hubble_constant
        header['gizmo']['flag_sfr'] = 1
        header['gizmo']['flag_cooling'] = 1
        header['gizmo']['flag_stellarage'] = 1
        header['gizmo']['flag_metals'] = 11
        header['gizmo']['flag_feedback'] = 1
        header['gizmo']['flag_doubleprecision'] = doubleprecision # TODO: pygr.readheader(snapshot_file, 'doubleprecision')
        header['gizmo']['flag_ic_info'] = 0


        util.io.info('Reading %s' % snapshot_file)
        util.io.info('GIZMO Header:')

        print(header)

        mass_factor = self.gadget_mass / self.gizmo_mass
        length_factor = self.gadget_length / self.gizmo_length
        time_factor = self.gadget_time / self.gizmo_time
        density_factor = self.gadget_density / self.gizmo_density
        specific_energy_factor =  self.gadget_specific_energy / self.gizmo_specific_energy
        velocity_factor = self.gadget_velocity / self.gizmo_velocity

        # Read in all of the snapshot file data and check if it matches
        # what we read in the header. If it doesn't match what is in the
        # header, we have a big problem! In that case, stop execution
        # immediately and tell the user.

        # Gas properties
        if self.ngas > 0:
            util.io.info('Building GIZMO gas dictionary.')

            gas_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'gas', suppress = 1)
            gas_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'gas', suppress = 1) * mass_factor
            gas_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'gas', suppress = 1) * length_factor
            gas_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'gas', suppress = 1) * velocity_factor
            # estimate smoothing length for 64 neighbors in a uniform grid; this will be recomputed in GIZMO
            hsm = header['gizmo']['box_size'] * np.cbrt(64.0 / self.ngas) 
            gas_dict['SmoothingLength'] = hsm * np.ones(self.ngas) * length_factor
            gas_dict['Density'] = gas_dict['Masses'] / ((4.0 *np.pi / 3.0) * gas_dict['SmoothingLength']**3)

            # Assume neutral primordial gas at CMB temperature
            int_energies = 2.73 * (1.0 + header['gizmo']['redshift'])  # initial temperature
            XH = 0.75  # H mass fraction
            mu = 4.0 / (1.0 + 3.0 * XH)  
            temp_to_u = util.const.KBOLTZ / (mu * util.const.PROTON_MASS * (util.const.GAMMA - 1.0))
            gas_dict['InternalEnergy'] = int_energies * temp_to_u * np.ones(self.ngas) * specific_energy_factor 

        # Dark properties
        if self.ndm > 0:
            util.io.info('Building GIZMO dark matter dictionary.')

            dark_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'dm', suppress = 1)
            dark_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'dm', suppress = 1) * mass_factor
            dark_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'dm', suppress = 1) * length_factor
            dark_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'dm', suppress = 1) * velocity_factor

        # Disk properties
        if self.ndisk > 0:
            util.io.info('Building GIZMO disk dictionary.')

            disk_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'disk', suppress = 1)
            disk_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'disk', suppress = 1) * mass_factor
            disk_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'disk', suppress = 1) * length_factor
            disk_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'disk', suppress = 1) * velocity_factor

        if self.nbulge > 0:
            util.io.info('Building GIZMO bulge dictionary.')

            bulge_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'bulge', suppress = 1)
            bulge_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'bulge', suppress = 1) * mass_factor
            bulge_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'bulge', suppress = 1) * length_factor
            bulge_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'bulge', suppress = 1) * velocity_factor

        if self.nstar > 0:
            # Star properties
            util.io.info('Building GIZMO star dictionary.')

            star_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'star', suppress = 1)
            star_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'star', suppress = 1) * mass_factor
            star_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'star', suppress = 1) * length_factor
            star_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'star', suppress = 1) * velocity_factor

        if self.nbh > 0:
            util.io.info('Building GIZMO black hole dictionary.')

            dark_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'bh', suppress = 1)
            dark_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'bh', suppress = 1) * mass_factor
            dark_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'bh', suppress = 1) * length_factor
            dark_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'bh', suppress = 1) * velocity_factor

        snapshot_hdf5 = snapshot_file + '.hdf5'

        util.io.info('Writing GIZMO file to disk %s' % snapshot_hdf5)

        # Finally! Write the GIZMO file out :)
        util.io.write_gizmo(snapshot_hdf5, header['gizmo'], gizmo_dict)


    def _set_gadget_units(self):
        self.gadget_length = 3.085678e21 / self.hubble_constant  # 1 kpc/h
        self.gadget_mass = 1.989e43 / self.hubble_constant       # 1e10 Msun/h
        self.gadget_velocity = 1.0e5

        self.gadget_specific_energy = self.gadget_velocity**2
        self.gadget_density = self.gadget_mass / self.gadget_length**3
        self.gadget_time = self.gadget_length / self.gadget_velocity

    def _set_gizmo_units(self):
        self.gizmo_length = 3.08567758e21 / self.hubble_constant    # 1 kpc/h
        self.gizmo_mass = 1.98841e43 / self.hubble_constant         # 1e10 Msun/h
        self.gizmo_velocity = 1.0e5

        self.gizmo_specific_energy = self.gizmo_velocity**2
        self.gizmo_density = self.gizmo_mass / self.gizmo_length**3
        self.gizmo_time = self.gizmo_length / self.gizmo_velocity

