# MsgaBpred
The data and standalone program of MsgaBpred
# Pre-requisite：
* conda env create -f environment.yml -n env_name
* You need to apply an account of [ESM-C](https://forge.evolutionaryscale.ai)
# Software
* To run the full & accurate version of MsgaBpred, you need to make sure the following software is in the mkdssp directory:<br> [DSSP](https://github.com/cmbi/dssp)
# Build dataset
*  1.``git clone https://github.com/Moon-kind-W/MsgaBpred.git && cd MsgaBpred``
*  2.``python esmc.py``
<br>Then enter the API key of ESM-C.
*  3.``python dataset.py --gpu 0``
# Run MsgaBpred for training
After building our dataset epitope3D, train the model with default hyper params:
* ``python train.py``
# Run MsgaBpred for prediction
Please execute the following command directly if you can provide the PDB file.
If you do not have a PDB file, you can use AlphaFold3 to predict the protein structure.
* ``python test.py``
