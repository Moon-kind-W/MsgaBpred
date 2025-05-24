import os
import numpy as np
import subprocess
DICT={
    'ALA': 'A', 'CYS': 'C', 'CCS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
def pdb_split(line):
    order=int(line[6:11].strip())
    atom=line[11:16].strip()
    amino=line[16:21].strip()
    chain=line[21]
    site=line[22:28].strip()
    x=line[28:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    return order,atom,amino,chain,site,x,y,z
def judge(line,filt_atom='CA'):
    kind=line[:6].strip()
    if kind not in ['HETATM','ATOM']:
        return None
    order,atom,amino,chain,site,x,y,z=pdb_split(line)
    if filt_atom is not None and atom!=filt_atom:
        return None
    prefix=''
    if len(amino)>3:
        prefix=amino[0]
        amino=amino[-3:]
    if amino=='MSE':
        amino='MET'
    elif amino=='CCS' or amino[:-1]=='CS':
        amino='CYS'
    elif amino not in DICT.keys():
        return None
    return prefix+amino,chain,site,float(x),float(y),float(z)
def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]
    with open(dssp_file, "r") as f:
        lines = f.readlines()
    seq = ""
    dssp_feature = []
    position = []
    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        POS = lines[i][5:11].strip()
        position.append(POS)
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature,position
def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]
    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)
    return dssp_feature


def prepare_pdb_for_dssp(input_path, output_path):

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:

        f_out.write("HEADER    GENERATED FOR DSSP ANALYSIS\n")


        for line in f_in:
            if line.startswith(('ATOM', 'HETATM')):
                f_out.write(line)

        f_out.write("TER\nEND\n")


def get_dssp(ID, root):

    os.makedirs(f"{root}/dssp/", exist_ok=True)

    orig_pdb = f"{root}/purePDB/{ID}.pdb"
    temp_pdb = f"{root}/purePDB/{ID}_dssp_temp.pdb"
    dssp_out = f"{root}/dssp/{ID}.dssp"

    prepare_pdb_for_dssp(orig_pdb, temp_pdb)

    try:
        cmd = f"/mkdssp/mkdssp.exe --output-format=dssp  {temp_pdb}  {dssp_out}"
        ret = subprocess.run(cmd, shell=True, check=True,
                             stderr=subprocess.PIPE, text=True)

        dssp_seq, dssp_matrix, position = process_dssp(dssp_out)
        np.save(f"{root}/dssp/{ID}", transform_dssp(dssp_matrix))
        np.save(f"{root}/dssp/{ID}_pos", position)
        return True
    except subprocess.CalledProcessError as e:
        return None
    finally:

        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)