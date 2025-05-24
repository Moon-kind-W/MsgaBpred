from Bio.PDB import PDBParser
import esm.inverse_folding
import torch
import os

def esm_if_2_embedding(pdb_id, path):

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    parser = PDBParser()
    pdb_file_path = os.path.join(f"{path}", f"{pdb_id}.pdb")
    pdb_structure = parser.get_structure(f"{pdb_id}", pdb_file_path)
    pdb_model = pdb_structure[0]

    chain_list = [chain.get_id() for chain in pdb_model]
    fpath = os.path.join(f"{path}/" + f"{pdb_id}.pdb")

    chain_esm_if_list = []
    chain_coord_list = []

    for target_chain_id in chain_list:

        structure = esm.inverse_folding.util.load_structure(fpath, target_chain_id)
        coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

        esm_if_rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        chain_esm_if_list.append(torch.tensor(esm_if_rep).float())

        coord_list = [coords[idx // 3][1] for idx in range(len(structure)) if idx % 3 == 0]
        chain_coord_list.append(coord_list)
    chain_esm_if_list = torch.concat(chain_esm_if_list)
    return chain_esm_if_list, chain_coord_list