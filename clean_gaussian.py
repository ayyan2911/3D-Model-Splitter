#!/usr/bin/env python3
"""
Clean Gaussian Splatting PLY files with coordinate transformation
Calcola la trasformazione al volo dai file binari COLMAP
"""

import numpy as np
import argparse
import json
import struct
from pathlib import Path

# ============================================================================
# FUNZIONI DI LETTURA COLMAP BINARY (per ricalcolare la PCA al volo)
# ============================================================================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    return struct.unpack(endian_character + format_char_sequence, fid.read(num_bytes))

def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            prop = read_next_bytes(fid, 43, "QdddBBBd")
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, 1) # Salta i track elems che non servono
            points3D[prop[0]] = np.array(prop[1:4]) # Salva solo XYZ
    return points3D

def get_rotation_transform_from_sparse(sparse_dir):
    """Ricalcola gli stessi identici parametri di rotazione della UI"""
    points3D_path = Path(sparse_dir) / "points3D.bin"
    if not points3D_path.exists():
        raise FileNotFoundError(f"Impossibile trovare {points3D_path}. Serve per ricalcolare la rotazione.")
        
    print("🔄 Ricalcolo matrice di rotazione dai dati sparse di COLMAP...")
    points3D = read_points3D_binary(points3D_path)
    xyz = np.array(list(points3D.values()))
    
    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid
    
    cov = np.cov(xyz_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    R = eigvecs[:, idx]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    
    xyz_rotated = xyz_centered @ R
    final_centroid = xyz_rotated.mean(axis=0)
    
    return {
        "centroid": centroid,
        "R": R,
        "final_centroid": final_centroid
    }

# ============================================================================
# FUNZIONI DI I/O GAUSSIAN PLY
# ============================================================================

def read_ply_gaussian(filepath):
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[2], parts[1]))
        
        dtype_list = []
        for prop_name, prop_type in properties:
            if prop_type in ['float', 'float32']: dtype_list.append((prop_name, 'f4'))
            elif prop_type in ['double', 'float64']: dtype_list.append((prop_name, 'f8'))
            elif prop_type in ['uchar', 'uint8']: dtype_list.append((prop_name, 'u1'))
            elif prop_type in ['int', 'int32']: dtype_list.append((prop_name, 'i4'))
        
        dtype = np.dtype(dtype_list)
        data = np.frombuffer(f.read(), dtype=dtype, count=num_vertices)
        return data, properties

def write_ply_gaussian(filepath, data, properties):
    with open(filepath, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {len(data)}\n'.encode('utf-8'))
        for prop_name, prop_type in properties:
            f.write(f'property {prop_type} {prop_name}\n'.encode('utf-8'))
        f.write(b'end_header\n')
        data.tofile(f)

# ============================================================================
# LOGICA PRINCIPALE DI PULIZIA
# ============================================================================

def clean_gaussian_with_transform(ply_file, bounds_file, colmap_sparse_dir, part_index, output_file, use_overlap=False):
    with open(bounds_file, 'r') as f:
        bounds_data = json.load(f)
    
    part_bounds = None
    for part in bounds_data['parts']:
        if part['part_index'] == part_index:
            part_bounds = part
            break
    
    if part_bounds is None:
        raise ValueError(f"Parte {part_index} non trovata nel file JSON.")
    
    if use_overlap:
        bounds_target = part_bounds['bounds']
        print(f"🔄 Utilizzo dei bounds ESPANSI (con overlap) per la parte {part_index}")
    else:
        bounds_target = part_bounds['original_bounds']
        print(f"🔄 Utilizzo dei bounds ORIGINALI (senza overlap) per la parte {part_index}")
    
    # Ricava i parametri geometrici estratti da COLMAP sparse
    transform_params = get_rotation_transform_from_sparse(colmap_sparse_dir)
    R = transform_params['R']
    centroid = transform_params['centroid']
    final_centroid = transform_params['final_centroid']
    
    print(f"📖 Lettura di {ply_file}...")
    data, properties = read_ply_gaussian(ply_file)
    original_count = len(data)
    print(f"   Punti totali originali: {original_count:,}")
    
    # Estrai e trasforma le coordinate dei Gaussian
    xyz_orig = np.stack([data['x'], data['y'], data['z']], axis=1)
    xyz_rot = (xyz_orig - centroid) @ R - final_centroid
    
    x_rot = xyz_rot[:, 0]
    y_rot = xyz_rot[:, 1]
    
    # Applica la maschera di ritaglio nello spazio allineato
    mask = (x_rot >= bounds_target['min_x']) & (x_rot <= bounds_target['max_x']) & \
           (y_rot >= bounds_target['min_y']) & (y_rot <= bounds_target['max_y'])
    
    filtered_data = data[mask]
    filtered_count = len(filtered_data)
    
    print(f"\n✨ Risultato del filtraggio:")
    print(f"   Punti mantenuti: {filtered_count:,} ({100*filtered_count/original_count:.2f}%)")
    
    print(f"💾 Scrittura del file in {output_file}...")
    write_ply_gaussian(output_file, filtered_data, properties)
    print(f"🏁 Completato!")

def main():
    parser = argparse.ArgumentParser(description="Pulisce file PLY Gaussian ricalcolando la PCA al volo")
    parser.add_argument("ply_file", help="Input Gaussian PLY file")
    parser.add_argument("bounds_file", help="File cluster_bounds.json")
    parser.add_argument("colmap_sparse_dir", help="Cartella contenente sparse/0 (dove c'è points3D.bin)")
    parser.add_argument("--part", type=int, required=True, help="Indice della parte")
    parser.add_argument("-o", "--output", required=True, help="Output PLY")
    parser.add_argument("--use_overlap", action="store_true", help="Usa i bounds con overlap")
    
    args = parser.parse_args()
    
    clean_gaussian_with_transform(
        args.ply_file,
        args.bounds_file,
        args.colmap_sparse_dir,
        args.part,
        args.output,
        use_overlap=args.use_overlap
    )

if __name__ == "__main__":
    main()