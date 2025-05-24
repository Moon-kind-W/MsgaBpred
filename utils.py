import os
import torch
import numpy as np
import pickle as pk
from tqdm import tqdm
from preprocess import *
from graph_construction import calcPROgraph
from esm_embedding import esm_if_2_embedding

# Amino acid to ID mapping
amino2id = {
    '<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
    'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22,
    'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28,
    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32, '<cath>': 33, '<af2>': 34
}


class ProteinChain:
    def __init__(self):
        self.sequence = []
        self.amino = []
        self.coord = []
        self.site = {}
        self.date = ''
        self.length = 0
        self.adj = None
        self.edge = None
        self.feat = None
        self.dssp = None
        self.name = ''
        self.chain_name = ''
        self.protein_name = ''
        self.label = None

    def add(self, amino, pos, coord):
        """Add amino acid to the chain"""
        self.sequence.append(DICT[amino])
        self.amino.append(amino2id[DICT[amino]])
        self.coord.append(coord)
        self.site[self.length] = self.length
        self.length += 1

    def process(self):
        """Convert lists to tensors"""

        self.amino = torch.LongTensor(self.amino)
        self.coord = torch.FloatTensor(self.coord)
        self.sequence = ''.join(self.sequence)
        if self.label is None:  # Initialize labels if not set
            self.label = torch.zeros(self.length, dtype=torch.long)

    def extract(self, model, device, path):
        with open(f'{path}/esmc/{self.name}.fasta/origin_seq_emb_6b.pkl', 'rb') as f:
            feat = pk.load(f)
        torch.save(feat, f'{path}/feat/{self.name}_esmc.ts')


    def load_feat(self, path):
        self.feat = torch.load(f'{path}/feat/{self.name}_esmc.ts')
    def update_from_labels(self, label_str):
        """Update labels from label string (0s and 1s)"""
        labels = []
        for c in label_str:
            if c == '1':
                labels.append(1)
            else:
                labels.append(0)

        if len(labels) != self.length:
            raise ValueError(f"Label length ({len(labels)}) doesn't match sequence length ({self.length})")

        self.label = torch.LongTensor(labels)

    def load_esm_if(self, path):
        """Load ESM-IF embeddings"""
        self.esm_if = torch.load(f'{path}/esm_if/{self.name}_esm_if.pt')

    def get_adj(self, path, dseq=3, dr=10, dlong=5, k=10):
        """Calculate and save adjacency matrix"""
        graph = calcPROgraph(self.sequence, self.coord, dseq, dr, dlong, k)
        torch.save(graph, f'{path}/graph/{self.name}.graph')

    def load_dssp(self,path):
        dssp=torch.Tensor(np.load(f'{path}/dssp/{self.name}.npy'))
        #pos=np.load(f'{path}/dssp/{self.name}_pos.npy')
        self.dssp=torch.Tensor([
            -2.4492936e-16, -2.4492936e-16,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]).repeat(self.length,1)
        self.rsa=torch.zeros(self.length)

        for i in range(len(dssp)):

            self.dssp[i]=dssp[i]
            if dssp[i][4]>0.15:
                self.rsa[i]=1
        self.rsa=self.rsa.bool()

    def load_adj(self,path,self_cycle=False):
        graph=torch.load(f'{path}/graph/{self.name}.graph')
        self.adj=graph['adj'].to_dense()
        self.edge=graph['edge'].to_dense()
        if not self_cycle:
            self.adj[range(len(self)),range(len(self))]=0
            self.edge[range(len(self)),range(len(self))]=0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.amino[idx], self.coord[idx], self.label[idx]


def parse_fasta_file(fasta_path):
    """Parse a single FASTA file with sequence and labels"""
    with open(fasta_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None

    # Parse header
    if not lines[0].startswith('>'):
        raise ValueError("FASTA file must start with '>'")

    filename = lines[0][1:].split()[0]  # Get the first word after '>'

    # Separate sequence and label lines
    sequence_lines = []
    label_lines = []
    is_sequence = True

    for line in lines[1:]:
        if line.startswith('>'):
            break  # Stop at next sequence header

        if all(c in '01' for c in line):  # Detect label line
            is_sequence = False

        if is_sequence:
            sequence_lines.append(line)
        else:
            label_lines.append(line)

    # Combine multi-line sequences and labels
    sequence = ''.join(sequence_lines)
    labels = ''.join(label_lines)

    # Validate lengths
    if len(labels) != len(sequence):
        raise ValueError(f"Sequence length ({len(sequence)}) and label length ({len(labels)}) mismatch")

    return {
        'filename': filename,
        'sequence': sequence,
        'labels': labels
    }


def process_pdb_chain(data, root, pdb_id, model, device):
    get_dssp(pdb_id, root)
    """Process PDB file to extract structure information"""
    pdb_file = f'{root}/purePDB/{pdb_id}.pdb'

    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    same = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line[:6] == 'HEADER':
                data.date = line[50:59].strip()
                continue

            feats = judge(line, 'CA')
            if feats is None:
                continue

            amino, _, site, x, y, z = feats

            # Handle modified residues
            if len(amino) > 3:
                if same.get(site) is None:
                    same[site] = amino[0]
                if same[site] != amino[0]:
                    continue
                amino = amino[-3:]

            data.add(amino, site, [x, y, z])

    # Generate and save ESM-IF embeddings
    esm_if_rep, _ = esm_if_2_embedding(pdb_id, f"{root}/purePDB")
    os.makedirs(f'{root}/esm_if', exist_ok=True)
    torch.save(esm_if_rep, f'{root}/esm_if/{data.name}_esm_if.pt')
    data.load_esm_if(root)

    # Process and save graph structure
    data.process()
    os.makedirs(f'{root}/graph', exist_ok=True)
    data.get_adj(root)
    data.extract(None, device, root)
    return data
def collate_fn(batch):
    edges = [item['edge_index'] for item in batch]
    edge_attrs = [item['edge_attr'] for item in batch]
    feats = [item['feat'] for item in batch]
    labels = torch.cat([item['label'] for item in batch], 0)
    return feats, edges, edge_attrs, labels

def process_train_fasta_directory(fasta_dir, output_dir, model=None, device='cpu', use_pdb=True):
    """Process all FASTA files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(('.fasta', '.fa'))]
    samples = []

    with tqdm(fasta_files, desc="Processing FASTA files") as pbar:
        for fasta_file in pbar:
            pbar.set_postfix(file=fasta_file)
            fasta_path = os.path.join(fasta_dir, fasta_file)

            try:
                # Parse FASTA file
                parsed = parse_fasta_file(fasta_path)
                if not parsed:
                    continue

                # Extract PDB ID and chain ID from filename
                filename = parsed['filename']
                if '_' not in filename:
                    print(f"Skipping {filename}: expected format PDBID_CHAIN")
                    continue

                pdb_id, chain_id = filename.split('_')[:2]

                # Initialize protein chain
                chain_data = ProteinChain()
                chain_data.protein_name = pdb_id
                chain_data.chain_name = chain_id
                chain_data.name = filename

                # Process PDB structure if available
                if use_pdb:
                    try:
                        chain_data = process_pdb_chain(chain_data, output_dir, f"{pdb_id}_{chain_id}", model, device)
                    except Exception as e:
                        print(f"Error processing PDB for {filename}: {str(e)}")
                        continue
                else:
                    # If not using PDB, just store the sequence
                    chain_data.sequence = parsed['sequence']
                    chain_data.process()

                # Update labels from FASTA
                chain_data.update_from_labels(parsed['labels'])
                samples.append(chain_data)

            except Exception as e:
                print(f"Error processing {fasta_file}: {str(e)}")
                continue

    # Save processed data
    output_file = os.path.join(output_dir, 'train.pkl')
    with open(output_file, 'wb') as f:
        pk.dump(samples, f)

    print(f"\nSuccessfully processed {len(samples)} samples")
    print(f"Results saved to: {output_file}")

    return samples

def process_test_fasta_directory(fasta_dir, output_dir, model=None, device='cpu', use_pdb=True):
    """Process all FASTA files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(('.fasta', '.fa'))]
    samples = []

    with tqdm(fasta_files, desc="Processing FASTA files") as pbar:
        for fasta_file in pbar:
            pbar.set_postfix(file=fasta_file)
            fasta_path = os.path.join(fasta_dir, fasta_file)

            try:
                # Parse FASTA file
                parsed = parse_fasta_file(fasta_path)
                if not parsed:
                    continue

                # Extract PDB ID and chain ID from filename
                filename = parsed['filename']
                if '_' not in filename:
                    print(f"Skipping {filename}: expected format PDBID_CHAIN")
                    continue

                pdb_id, chain_id = filename.split('_')[:2]

                # Initialize protein chain
                chain_data = ProteinChain()
                chain_data.protein_name = pdb_id
                chain_data.chain_name = chain_id
                chain_data.name = filename

                # Process PDB structure if available
                if use_pdb:
                    try:
                        chain_data = process_pdb_chain(chain_data, output_dir, f"{pdb_id}_{chain_id}", model, device)
                    except Exception as e:
                        print(f"Error processing PDB for {filename}: {str(e)}")
                        continue
                else:
                    # If not using PDB, just store the sequence
                    chain_data.sequence = parsed['sequence']
                    chain_data.process()

                # Update labels from FASTA
                chain_data.update_from_labels(parsed['labels'])
                samples.append(chain_data)

            except Exception as e:
                print(f"Error processing {fasta_file}: {str(e)}")
                continue

    # Save processed data
    output_file = os.path.join(output_dir, 'test.pkl')
    with open(output_file, 'wb') as f:
        pk.dump(samples, f)

    print(f"\nSuccessfully processed {len(samples)} samples")
    print(f"Results saved to: {output_file}")

    return samples