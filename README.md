# cosmoconvert

## How to setup!

```
python setup.py build && python setup.py install
```

## How to run!
Convert between cosmological simulation formats, why not!

Example for Swift IC conversion from MUSIC Tipsy format:

```
import cosmoconvert.tipsy2swift as converter

snapshot_file = "test_ics.dat"
box_size = 25.0  # Tipsy Mpc/h
cosmology = {"hubble_constant": 0.7,
             "omega_matter": 0.3,
             "omega_lambda": 0.7,
             "omega_baryon": 0.048}

converter.tipsy2swift(snapshot_file, box_size, cosmology, is_IC = True)
```

Converting MUSIC Gadget-2 format to Swift is even easier, but
you will need to install pygadgetreader: https://github.com/dnarayanan/pygadgetreader
```
import cosmoconvert.gadget2swift as converter

snapshot_file = "test_ics.dat"

converter.gadget2swift(snapshot_file, omega_baryon=0.048)
```

