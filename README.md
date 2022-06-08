# cosmoconvert
Convert between cosmological simulation formats, why not!

Ex:

```
import cosmoconvert.tipsy2swift as converter

snapshot_file = "test_ics.bin"
box_size = 25.0  # Tipsy Mpc/h
cosmology = {"hubble_constant": 0.7,
             "omega_matter": 0.3,
             "omega_lambda": 0.7}

converter.tipsy2swift(snapshot_file, box_size, cosmology)
```
