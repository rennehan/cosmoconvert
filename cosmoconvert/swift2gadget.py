import numpy as np
from . import util
import h5py


class swift2gadget(object):
    def __init__(self, snapshot_file, verbose=True, out=None):
        util.io.verbose = verbose

        with h5py.File(snapshot_file) as f:
            snap_header = f["Header"]
            self.redshift = snap_header.attrs["Redshift"]
            self.hubble_constant = f["Cosmology"].attrs["H0 [internal units]"] / 100

            # This is an array of 3. We'll only take the first one, since boxes are cubes.
            self.box_size = snap_header.attrs["BoxSize"]
            assert self.box_size[0] == self.box_size[1] == self.box_size[2]
            # Since all three dimensions are equal, we can just take the first one
            self.box_size = self.box_size[0]

            # particle counts are split into 2 32 bit integers.
            particle_counts = (
                snap_header.attrs["NumPart_Total_HighWord"] << 32
            ) + snap_header.attrs["NumPart_Total"]
            self.ngas = particle_counts[0]
            self.nstar = particle_counts[4]
            self.ndm = particle_counts[1]

            # Calculate the necessary units for conversion.
            # The Swift units are default:
            # L = 1.0 Mpc
            # M = 1e10 Msun
            # V = 1e5 km/s
            self._set_gizmo_units()
            self._set_swift_units()

            header = {"gadget": {}}
            header["gadget"]["time"] = snap_header.attrs["Time"]
            header["gadget"]["redshift"] = self.redshift
            header["gadget"]["npart"] = np.array(
                [self.ngas, self.ndm, 0, 0, self.nstar, 0, 0]
            )
            header["gadget"]["box_size"] = (
                self.box_size * self.swift_length / self.gizmo_length
            )
            header["gadget"]["omega0"] = (
                f["Cosmology"].attrs["Omega_b"] + f["Cosmology"].attrs["Omega_cdm"]
            )
            header["gadget"]["omega_baryon"] = f["Cosmology"].attrs["Omega_b"]
            header["gadget"]["omega_lambda"] = f["Cosmology"].attrs["Omega_lambda"]
            header["gadget"]["hubble_constant"] = self.hubble_constant

            header["gadget"]["unit_current"] = 1
            header["gadget"]["unit_temperature"] = 1
            header["gadget"]["unit_length"] = self.gizmo_length
            header["gadget"]["unit_mass"] = self.gizmo_mass
            header["gadget"]["unit_time"] = self.gizmo_time

            header["gadget"]["flag_sfr"] = 0
            header["gadget"]["flag_cooling"] = 0
            header["gadget"]["flag_stellarage"] = 0
            header["gadget"]["flag_metals"] = 11
            header["gadget"]["flag_feedback"] = 0
            header["gadget"]["flag_doubleprecision"] = 0
            header["gadget"]["flag_ic_info"] = 0

            util.io.info("Reading %s" % snapshot_file)
            util.io.info("Gadget Header:")

            print(header)

            mass_factor = self.swift_mass / self.gizmo_mass
            length_factor = self.swift_length / self.gizmo_length
            time_factor = self.swift_time / self.gizmo_time
            density_factor = self.swift_density / self.gizmo_density
            specific_energy_factor = (
                self.swift_specific_energy / self.gizmo_specific_energy
            )
            velocity_factor = self.swift_velocity / self.gizmo_velocity

            gadget_dict = {
                "PartType0": {},
                "PartType1": {},
                "PartType4": {},
            }
            gas_dict = gadget_dict["PartType0"]
            dark_dict = gadget_dict["PartType1"]
            star_dict = gadget_dict["PartType4"]

            # Gas properties
            if self.ngas > 0:
                util.io.info("Building Gadget gas dictionary.")
                factors = [
                    ["ParticleIDs", "ParticleIDs", 1],
                    ["Masses", "Masses", mass_factor],
                    ["Coordinates", "Coordinates", length_factor],
                    ["Velocities", "Velocities", velocity_factor],
                    ["SmoothingLength", "SmoothingLengths", length_factor],
                    ["Density", "Densities", density_factor],
                    ["InternalEnergy", "InternalEnergies", specific_energy_factor],
                ]
                for [gadget_id, swift_id, factor] in factors:
                    gas_dict[gadget_id] = f["PartType0"][swift_id][:] * factor

            # Dark Matter properties
            if self.ndm > 0:
                util.io.info("Building Gadget dark matter dictionary.")
                factors = [
                    ["ParticleIDs", "ParticleIDs", 1],
                    ["Masses", "Masses", mass_factor],
                    ["Coordinates", "Coordinates", length_factor],
                    ["Velocities", "Velocities", velocity_factor],
                ]
                for [gadget_id, swift_id, factor] in factors:
                    dark_dict[gadget_id] = f["PartType1"][swift_id][:] * factor

            # Star properties
            if self.nstar > 0:
                util.io.info("Building Gadget star dictionary.")
                factors = [
                    ["ParticleIDs", "ParticleIDs", 1],
                    ["Masses", "Masses", mass_factor],
                    ["Coordinates", "Coordinates", length_factor],
                    ["Velocities", "Velocities", velocity_factor],
                ]
                for [gadget_id, swift_id, factor] in factors:
                    star_dict[gadget_id] = f["PartType4"][swift_id][:] * factor

        snapshot_hdf5 = out or snapshot_file + ".hdf5"
        util.io.info("Writing Gadget file to disk %s" % snapshot_hdf5)
        util.io.write_gadget(snapshot_hdf5, header["gadget"], gadget_dict)

    def _set_gizmo_units(self):
        self.gizmo_length = 3.085678e21 / self.hubble_constant  # 1 kpc/h
        self.gizmo_mass = 1.989e43 / self.hubble_constant  # 1e10 Msun/h
        self.gizmo_velocity = 1.0e5 * np.sqrt(
            1.0 + self.redshift
        )  # 1 km/s, plus Gadget's weird 1/sqrt(a) factor

        self.gizmo_specific_energy = self.gizmo_velocity**2
        self.gizmo_density = self.gizmo_mass / self.gizmo_length**3
        self.gizmo_time = self.gizmo_length / self.gizmo_velocity

    def _set_swift_units(self):
        self.swift_length = 3.08567758e24  # 1 Mpc
        self.swift_mass = 1.98841e43  # 1e10 Msun
        self.swift_velocity = 1.0e5  # 1 km/s

        self.swift_specific_energy = self.swift_velocity**2
        self.swift_density = self.swift_mass / self.swift_length**3
        self.swift_time = self.swift_length / self.swift_velocity
