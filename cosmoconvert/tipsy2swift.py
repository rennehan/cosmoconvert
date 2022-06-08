import numpy as np
from . import util


class tipsy2swift(object):
    # TODO: Make these not hard-coded numbers :(
    tipsy_num_metals = 4
    swift_num_metals = 11

    # Common indices for structure
    mass_idx = 0
    pos_x_idx = 1
    pos_y_idx = 2
    pos_z_idx = 3
    vel_x_idx = 4
    vel_y_idx = 5
    vel_z_idx = 6

    # Additional gas indices
    rho_idx = 7
    temp_idx = 8
    hsmooth_idx = 9
    metals_gas_idx = 10
    phi_gas_idx = 11

    # Additional dark indices
    eps_dark_idx = 7
    phi_dark_idx = 8

    # Additional star indices
    metals_star_idx = 7
    tform_idx = 8
    eps_star_idx = 9
    phi_star_idx = 10

    # Aux. ordering for metals
    carbon_idx = 0
    oxygen_idx = 1
    silicon_idx = 2
    iron_idx = 3

    # Additional aux. indices for gas
    sfr_idx = 4
    tmax_idx = 5
    delay_idx = 6
    ne_idx = 7
    nh_idx = 8

    # Liang+'16 aux: fffffffffi
    # Liang+'16 bin: ffffffi
    def __init__(self, snapshot_file, box_size, cosmology, aux_file = None,
                 aux_gas_str = 'fffffffffih',
                 aux_star_str = 'ffffffih',
                 idnum_file = None):
        self.box_size = box_size
        self.hubble_constant = cosmology['hubble_constant']
        # cosmology['omega0'], cosmology['omega_lambda'] are required

        util.io.verbose = True

        # Calculate the necessary units for conversion.
        # Everything is based on the box size and hubble constant
        # for TIPSY units.
        #
        # The Swift units are default:
        # L = 1.0 Mpc
        # M = 1e10 Msun
        # V = 1e5 km/s
        self._set_tipsy_units()
        self._set_swift_units()

        # This is the standard format for Swift files.
        swift_dict = dict(PartType0 = {}, PartType1 = {}, PartType4 = {})

        gas_dict = swift_dict['PartType0']
        dark_dict = swift_dict['PartType1']
        star_dict = swift_dict['PartType4']

        header = util.io.read_tipsy(snapshot_file, header_only = True)

        # Add on Swift header data
        header['swift'] = {}
        if header['time'] > 1.0:
            header['swift']['time'] = 1.0
            header['swift']['redshift'] = 0.0
        else:
            header['swift']['time'] = header['time']
            header['swift']['redshift'] = (1.0 / header['time']) - 1.0
        header['swift']['npart'] = np.array([header['ngas'], header['ndark'], 0, 0, header['nstar'], 0])
        header['swift']['box_size'] = self.box_size / self.hubble_constant # Swift expects Mpc?
        header['swift']['omega_matter'] = cosmology['omega_matter']
        header['swift']['omega_lambda'] = cosmology['omega_lambda']
        header['swift']['hubble_constant'] = self.hubble_constant
        header['swift']['flag_entropy_ics'] = 0

        util.io.info('Reading %s' % snapshot_file)
        util.io.info('Box size %s Mpc' % box_size)
        util.io.info('TIPSY+Swift Header:')

        print(header)

        mass_factor = self.unit_mass / self.swift_mass
        length_factor = self.unit_length / self.swift_length
        time_factor = self.unit_time / self.swift_time
        density_factor = self.unit_density / self.swift_density
        specific_energy_factor = 1.0 / self.swift_specific_energy
        velocity_factor = self.unit_velocity / self.swift_velocity

        # Read in all of the snapshot file data and check if it matches
        # what we read in the header. If it doesn't match what is in the
        # header, we have a big problem! In that case, stop execution
        # immediately and tell the user.
        gas_data, dark_data, star_data, byte_swap = util.io.read_tipsy(snapshot_file)

        if header['ngas'] > 0:
            gas_data = gas_data.view(gas_data.dtype[0]).reshape(gas_data.shape + (-1,))
            if len(gas_data) != header['ngas']:
                raise ValueError('Gas data array length does not match header (%d != %d)'
                                 % (len(gas_data), header['ngas']))

        dark_data = dark_data.view(dark_data.dtype[0]).reshape(dark_data.shape + (-1,))
        if len(dark_data) != header['ndark']:
            raise ValueError('Dark data array length does not match header (%d != %d)'
                             % (len(dark_data), header['ndark']))

        if header['nstar'] > 0:
            star_data = star_data.view(star_data.dtype[0]).reshape(star_data.shape + (-1,))
            if len(star_data) != header['nstar']:
                raise ValueError('Star data array length does not match header (%d != %d)'
                                 % (len(star_data), header['nstar']))

        particle_ids = None
        unique = True
        if idnum_file is not None:
            particle_ids = np.fromfile(idnum_file, dtype = np.uint32)
            if len(particle_ids) != (header['ngas'] + header['ndark'] + header['nstar']):
                raise ValueError('The number of particle IDs does not match the number of particles (%d != %d)'
                                 % (len(particle_ids), int(header['ngas'] + header['ndark'] + header['nstar'])))

            if len(np.unique(particle_ids)) != len(particle_ids):
                unique = False
                util.io.info('There are non-unique particle IDs (u: %d, t: %d)'
                             % (len(np.unique(particle_ids)), len(particle_ids)))

        util.io.info('Successfully read TIPSY data, shifting particle positions now.')

        # Make sure all of the particles are within the range [0, 1.0]
        self._shift_particle_positions(gas_data, dark_data, star_data, header)

        pos_indices = [self.pos_x_idx, self.pos_y_idx, self.pos_z_idx]
        vel_indices = [self.vel_x_idx, self.vel_y_idx, self.vel_z_idx]

        final_idx = 1
        if header['ngas'] > 0:
            # Gas properties
            util.io.info('Building Swift gas dictionary.')
            start_idx = 1
            final_idx = header['ngas'] + 1

            if particle_ids is not None:
                gas_dict['ParticleIDs'] = particle_ids[start_idx - 1:final_idx - 1]
                assert len(np.unique(gas_dict['ParticleIDs'])) == len(gas_dict['ParticleIDs'])
            else:
                gas_dict['ParticleIDs'] = np.arange(start_idx, final_idx)

            gas_dict['Masses'] = gas_data[:, self.mass_idx] * mass_factor
            gas_dict['Density'] = gas_data[:, self.rho_idx] * density_factor

            gas_dict['Coordinates'] = np.zeros((header['ngas'], 3))
            for i, j in enumerate(pos_indices):
                gas_dict['Coordinates'][:, i] = gas_data[:, j] * length_factor

            gas_dict['Velocities'] = np.zeros((header['ngas'], 3))
            for i, j in enumerate(vel_indices):
                gas_dict['Velocities'][:, i] = gas_data[:, j] * velocity_factor

            gas_dict['SmoothingLength'] = gas_data[:, self.hsmooth_idx] * length_factor

            gas_dict['Metallicity'] = np.zeros((header['ngas'], header['swift']['flag_metals']))
            gas_dict['Metallicity'][:, 0] = gas_data[:, self.metals_gas_idx]

            # Save this for later when we have electron abundances
            int_energies = gas_data[:, self.temp_idx]

            # We don't need this anymore, free the memory.
            del gas_data

        # Dark properties
        util.io.info('Building Swift dark matter dictionary.')
        start_idx = final_idx
        final_idx += header['ndark']

        if particle_ids is not None:
            dark_dict['ParticleIDs'] = particle_ids[start_idx - 1:final_idx - 1]
            assert len(np.unique(dark_dict['ParticleIDs'])) == len(dark_dict['ParticleIDs'])
        else:
            dark_dict['ParticleIDs'] = np.arange(start_idx, final_idx)

        dark_dict['Masses'] = dark_data[:, self.mass_idx] * mass_factor

        dark_dict['Coordinates'] = np.zeros((header['ndark'], 3))
        for i, j in enumerate(pos_indices):
            dark_dict['Coordinates'][:, i] = dark_data[:, j] * length_factor

        dark_dict['Velocities'] = np.zeros((header['ndark'], 3))
        for i, j in enumerate(vel_indices):
            dark_dict['Velocities'][:, i] = dark_data[:, j] * velocity_factor

        del dark_data

        if header['nstar'] > 0:
            # Star properties
            util.io.info('Building Swift star dictionary.')
            start_idx = final_idx
            final_idx += header['nstar']

            if particle_ids is not None and unique:
                star_dict['ParticleIDs'] = particle_ids[start_idx - 1:final_idx - 1]
            else:
                max_gas_id = np.amax(gas_dict['ParticleIDs'])
                max_dark_id = np.amax(dark_dict['ParticleIDs'])

                if max_gas_id > max_dark_id:
                    start_value = max_gas_id
                else:
                    start_value = max_dark_id

                util.io.info('Maximum gas/dark ParticleID: %d' % start_value)
                star_dict['Old_ParticleIDs'] = particle_ids[start_idx - 1:final_idx - 1]
                star_dict['ParticleIDs'] = np.arange(start_value + 1, start_value + 1 + header['nstar'])
                util.io.info('New maximum star ParticleID: %d' % np.amax(star_dict['ParticleIDs']))

                assert len(np.unique(star_dict['ParticleIDs'])) == len(star_dict['ParticleIDs'])

            star_dict['Masses'] = star_data[:, self.mass_idx] * mass_factor

            star_dict['Coordinates'] = np.zeros((header['nstar'], 3))
            for i, j in enumerate(pos_indices):
                star_dict['Coordinates'][:, i] = star_data[:, j] * length_factor

            star_dict['Velocities'] = np.zeros((header['nstar'], 3))
            for i, j in enumerate(vel_indices):
                star_dict['Velocities'][:, i] = star_data[:, j] * velocity_factor

            star_dict['StellarFormationTime'] = star_data[:, self.tform_idx] * time_factor

            star_dict['Metallicity'] = np.zeros((header['nstar'], header['swift']['flag_metals']))
            star_dict['Metallicity'][:, 0] = star_data[:, self.metals_star_idx]

            del star_data

        if aux_file is not None:
            util.io.info('Loading auxiliary file %s' % aux_file)
            gas_data, star_data = util.io.read_tipsy_aux(aux_file,
                                                         header['ngas'],
                                                         header['nstar'],
                                                         aux_gas_str,
                                                         aux_star_str,
                                                         byte_swap = byte_swap)

            # TODO: This will destroy the integer value at the end, but we don't need it
            gas_data = gas_data.view(gas_data.dtype[0]).reshape(gas_data.shape + (-1,))
            if len(gas_data) != header['ngas']:
                raise ValueError('Gas aux. data array length does not match header (%d != %d)'
                                 % (len(gas_data), header['ngas']))

            star_data = star_data.view(star_data.dtype[0]).reshape(star_data.shape + (-1,))
            if len(star_data) != header['nstar']:
                raise ValueError('Star aux. data array length does not match header (%d != %d)'
                                 % (len(star_data), header['nstar']))

            util.io.info('Successfullly loaded auxiliary file.')
            util.io.info('Preparing gas metal dictionary.')

            # TODO: This only works if tipsy_num_metals = 4, in C, O, Si, Fe order
            gas_dict['Metallicity'][:, 2] = gas_data[:, self.carbon_idx]
            gas_dict['Metallicity'][:, 4] = gas_data[:, self.oxygen_idx]
            gas_dict['Metallicity'][:, 7] = gas_data[:, self.silicon_idx]
            gas_dict['Metallicity'][:, 10] = gas_data[:, self.iron_idx]

            gas_dict['StarFormationRate'] = gas_data[:, self.sfr_idx]
            gas_dict['DelayTime'] = gas_data[:, self.delay_idx]
            gas_dict['ElectronAbundance'] = gas_data[:, self.ne_idx]
            gas_dict['NeutralHydrogenAbundance'] = gas_data[:, self.nh_idx]

            mu_mp = 4.0 / (3.0 * util.const.XH + 1.0 + 4.0 * util.const.XH * gas_data[:, self.ne_idx])
            mu_mp *= util.const.PROTON_MASS
            real_int_energies = int_energies * util.const.KBOLTZ / (mu_mp * (util.const.GAMMA - 1.0))
            gas_dict['InternalEnergy'] = real_int_energies * specific_energy_factor  # From cm/s to km/s

            del gas_data

            util.io.info('Preparing star metal dictionary.')

            # TODO: This only works if tipsy_num_metals = 4, in C, O, Si, Fe order
            star_dict['Metallicity'][:, 2] = star_data[:, self.carbon_idx]
            star_dict['Metallicity'][:, 4] = star_data[:, self.oxygen_idx]
            star_dict['Metallicity'][:, 7] = star_data[:, self.silicon_idx]
            star_dict['Metallicity'][:, 10] = star_data[:, self.iron_idx]

            del star_data

        # Set the Helium mass fraction.
        if header['ngas'] > 0:
            gas_dict['Metallicity'][:, 1] = 0.236 + (2.1 * gas_dict['Metallicity'][:, 0])
        if header['nstar'] > 0:
            star_dict['Metallicity'][:, 1] = 0.236 + (2.1 * star_dict['Metallicity'][:, 0])

        snapshot_hdf5 = snapshot_file + '.hdf5'

        util.io.info('Writing Swift file to disk %s' % snapshot_hdf5)

        # Finally! Write the Swift file out :)
        util.io.write_swift(snapshot_hdf5, header['swift'], swift_dict)

    def _shift_particle_positions(self, gas, dark, star, header):
        for i in range(self.pos_x_idx, self.pos_z_idx + 1):
            if header['ngas'] > 0:
                gas[:, i] += 0.5
                out_idx = np.where(gas[:, i] > 1.0)
                gas[out_idx, i] -= 1.0

            if header['nstar'] > 0:
                star[:, i] += 0.5
                out_idx = np.where(star[:, i] > 1.0)
                star[out_idx, i] -= 1.0

            dark[:, i] += 0.5
            out_idx = np.where(dark[:, i] > 1.0)
            dark[out_idx, i] -= 1.0


    def _set_swift_units(self):
        self.swift_length = 3.08567758e24 #3.085678e21 / self.hubble_constant  # 1 kpc/h
        self.swift_mass = 1.98841e43 #1.989e43 / self.hubble_constant       # 1e10 Msun/h
        self.swift_velocity = 1.0e5                             # 1 km/s

        self.swift_specific_energy = self.swift_velocity**2
        self.swift_density = self.swift_mass / self.swift_length**3
        self.swift_time = self.swift_length / self.swift_velocity

    def _set_tipsy_units(self):
        self.unit_time = 8.9312007e17 / self.hubble_constant
        self.unit_density = 1.87870751e-29 * self.hubble_constant**2
        self.unit_length = self.box_size * util.const.CM_PER_MPC / self.hubble_constant

        self.unit_mass = self.unit_density * self.unit_length**3
        self.unit_velocity = self.unit_length / self.unit_time
        self.unit_energy = self.unit_mass * self.unit_length**2 / self.unit_time**2
