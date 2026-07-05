# Sequence Mode

## Enter the sequence mode while evaluating

```bash
python -m maploc.evaluation.rho_unstruc --experiment RHO_neuanspach/best.ckpt --sequential model.num_rotations=256
python -m maploc.evaluation.onet_unstructured --experiment Onet_neuanspach/best.ckpt --sequential model.num_rotations=256
```

If you want to evaluate model performance in other cities, use the corresponding model checkpoint and remember to change the ```scenes``` in ```rho_unstruc.py```, also the ```scenes``` and ```split``` in ```rho_unstruc.yaml```. The corresponding files of OrienterNet are ```onet_unstructured.py``` and ```mapillary_unstructured.yaml```.

For example, using the data of tirol:

```bash
python -m maploc.evaluation.rho_unstruc --experiment RHO_tirol/best.ckpt --sequential model.num_rotations=256
python -m maploc.evaluation.onet_unstructured --experiment Onet_tirol/best.ckpt --sequential model.num_rotations=256
```

Commonly used split json files are under the relevant directories. Naming rule is ```splits_MODEL_CITY.json```.

If you want to increase or decrease the number of frames in the sequence mode. You can change the ```max_length``` variable in rho_unstruc.py. Remember in RHO model, the ```max_length``` represents the number of pinhole views in a sequence. If you want to include 10 frames in a sequence in RHO, the ```max_length``` should be 30 (10 pano views), while in OrienterNet it's 10.
