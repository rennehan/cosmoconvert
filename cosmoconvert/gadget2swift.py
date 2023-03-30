import numpy as np
from . import util
import pygadgetreader as pygr

class gadget2swift(object):
    
    def __init__(self, snapshot_file, omega_baryon=0., verbose=True):

        util.io.verbose = verbose

        self.redshift = pygr.readheader(snapshot_file, 'redshift')
        self.hubble_constant = pygr.readheader(snapshot_file, 'h')
        self.box_size = pygr.readheader(snapshot_file, 'boxsize')
        self.ngas = pygr.readheader(snapshot_file, 'gascount')
        self.ndm = pygr.readheader(snapshot_file, 'dmcount')
        self.nstar = pygr.readheader(snapshot_file, 'starcount')

        # Calculate the necessary units for conversion.
        # The Swift units are default:
        # L = 1.0 Mpc
        # M = 1e10 Msun
        # V = 1e5 km/s
        self._set_gizmo_units()
        self._set_swift_units()

        # This is the standard format for Swift files.
        swift_dict = dict(PartType0 = {}, PartType1 = {}, PartType4 = {})

        gas_dict = swift_dict['PartType0']
        dark_dict = swift_dict['PartType1']
        star_dict = swift_dict['PartType4']

        header = {}

        # Add on Swift header data
        header['swift'] = {}
        header['swift']['time'] = pygr.readheader(snapshot_file, 'time')
        header['swift']['redshift'] = self.redshift
        header['swift']['npart'] = np.array([self.ngas, self.ndm, 0, 0, self.nstar, 0, 0])
        header['swift']['box_size'] = self.box_size * self.gizmo_length / self.swift_length
        header['swift']['omega_matter'] = pygr.readheader(snapshot_file, 'O0')
        header['swift']['omega_baryon'] = omega_baryon
        header['swift']['omega_lambda'] = pygr.readheader(snapshot_file, 'Ol')
        header['swift']['hubble_constant'] = self.hubble_constant
        header['swift']['flag_entropy_ics'] = 0
        header['swift']['flag_metals'] = 9

        header['swift']['unit_current'] = 1.0
        header['swift']['unit_temperature'] = 1.0
        header['swift']['unit_length'] = self.swift_length
        header['swift']['unit_mass'] = self.swift_mass
        header['swift']['unit_time'] = self.swift_time

        util.io.info('Reading %s' % snapshot_file)
        util.io.info('Swift Header:')

        print(header)

        mass_factor = self.gizmo_mass / self.swift_mass
        length_factor = self.gizmo_length / self.swift_length
        time_factor = self.gizmo_time / self.swift_time
        density_factor = self.gizmo_density / self.swift_density
        specific_energy_factor =  self.gizmo_specific_energy / self.swift_specific_energy
        velocity_factor = self.gizmo_velocity / self.swift_velocity

        # Swift has a scale factor in the internal energy
        specific_energy_factor *= header['swift']['time']**2.0
        # Swift has a scale factor in velocity
        velocity_factor *= header['swift']['time']

        # Read in all of the snapshot file data and check if it matches
        # what we read in the header. If it doesn't match what is in the
        # header, we have a big problem! In that case, stop execution
        # immediately and tell the user.

        # Gas properties
        if self.ngas > 0:
            util.io.info('Building Swift gas dictionary.')

            gas_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'gas', suppress=1)
            gas_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'gas', suppress=1) * mass_factor
            gas_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'gas', suppress=1) * length_factor
            gas_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'gas', suppress=1) * velocity_factor
            # estimate smoothing length for 64 neighbors in a uniform grid; this will be recomputed in Swift
            hsm = header['swift']['box_size'] * np.cbrt(64. / self.ngas) 
            gas_dict['SmoothingLength'] = hsm * np.ones(self.ngas) * length_factor
            gas_dict['Density'] = gas_dict['Masses'] / (4.*np.pi/3 * gas_dict['SmoothingLength']**3)

            # Assume neutral primordial gas at CMB temperature
            int_energies = 2.73 * (1. + header['swift']['redshift'])  # initial temperature
            XH = 0.75  # H mass fraction
            mu = 4. / (1 + 3 * XH)  
            temp_to_u = util.const.KBOLTZ / (mu * util.const.PROTON_MASS * (util.const.GAMMA - 1.0))
            gas_dict['InternalEnergy'] = int_energies * temp_to_u * np.ones(self.ngas) * specific_energy_factor 

        # Dark properties
        if self.ndm > 0:
            util.io.info('Building Swift dark matter dictionary.')

            dark_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'dm', suppress=1)
            dark_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'dm', suppress=1) * mass_factor
            dark_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'dm', suppress=1) * length_factor
            dark_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'dm', suppress=1) * velocity_factor

        if self.nstar > 0:
            # Star properties
            util.io.info('Building Swift star dictionary.')

            star_dict['ParticleIDs'] = pygr.readsnap(snapshot_file, 'pid', 'star', suppress=1)
            star_dict['Masses'] = pygr.readsnap(snapshot_file, 'mass', 'star', suppress=1) * mass_factor
            star_dict['Coordinates'] = pygr.readsnap(snapshot_file, 'pos', 'star', suppress=1) * length_factor
            star_dict['Velocities'] = pygr.readsnap(snapshot_file, 'vel', 'star', suppress=1) * velocity_factor

        # Set the Hydrogen & Helium mass fraction.
        #if header['ngas'] > 0:
        #    gas_dict['ElementMassFractions'][:, 1] = 0.236 + (2.1 * gas_dict['MetalMassFractions'])
        #    gas_dict['ElementMassFractions'][:, 0] = 1.0 - gas_dict['ElementMassFractions'][:, 1] - gas_dict['MetalMassFractions']
        #if header['nstar'] > 0:
        #    star_dict['ElementMassFractions'][:, 1] = 0.236 + (2.1 * star_dict['MetalMassFractions'])
        #    star_dict['ElementMassFractions'][:, 0] = 1.0 - star_dict['ElementMassFractions'][:, 1] - star_dict['MetalMassFractions']

        snapshot_hdf5 = snapshot_file + '.hdf5'

        util.io.info('Writing Swift file to disk %s' % snapshot_hdf5)

        # Finally! Write the Swift file out :)
        util.io.write_swift(snapshot_hdf5, header['swift'], swift_dict)


    def _set_gizmo_units(self):
        self.gizmo_length = 3.085678e21 / self.hubble_constant  # 1 kpc/h
        self.gizmo_mass = 1.989e43 / self.hubble_constant       # 1e10 Msun/h
        self.gizmo_velocity = 1.0e5 * np.sqrt(1.+self.redshift) # 1 km/s, plus Gadget's weird 1/sqrt(a) factor

        self.gizmo_specific_energy = self.gizmo_velocity**2
        self.gizmo_density = self.gizmo_mass / self.gizmo_length**3
        self.gizmo_time = self.gizmo_length / self.gizmo_velocity

    def _set_swift_units(self):
        self.swift_length = 3.08567758e24  # 1 Mpc
        self.swift_mass = 1.98841e43       # 1e10 Msun
        self.swift_velocity = 1.0e5        # 1 km/s

        self.swift_specific_energy = self.swift_velocity**2
        self.swift_density = self.swift_mass / self.swift_length**3
        self.swift_time = self.swift_length / self.swift_velocity

