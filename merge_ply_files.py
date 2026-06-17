#!/usr/bin/env python3
"""
Merge Gaussian Splatting PLY parts into a single PLY file
"""

import numpy as np
import argparse
from pathlib import Path

def read_ply_gaussian(filepath):
    """Legge un file PLY Gaussian mantenendo l'header dinamico"""
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
    """Scrive il file PLY Gaussian unito"""
    with open(filepath, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {len(data)}\n'.encode('utf-8'))
        for prop_name, prop_type in properties:
            f.write(f'property {prop_type} {prop_name}\n'.encode('utf-8'))
        f.write(b'end_header\n')
        data.tofile(f)

def main():
    parser = argparse.ArgumentParser(description="Unisci più parti PLY di Gaussian Splatting in un unico modello")
    parser.add_argument("input_files", nargs='+', help="Elenco dei file PLY delle parti da unire (es. part_*.ply)")
    parser.add_argument("-o", "--output", required=True, help="File PLY di output unito (es. merged_model.ply)")
    
    args = parser.parse_args()
    
    all_data = []
    base_properties = None
    total_points = 0
    
    print(f"🚀 Inizio unione di {len(args.input_files)} parti...")
    
    for filepath in args.input_files:
        p = Path(filepath)
        if not p.exists():
            print(f"⚠️ Salto {filepath}: file non trovato.")
            continue
            
        print(f"📖 Lettura di {p.name}...")
        data, properties = read_ply_gaussian(p)
        
        # Salva le proprietà del primo file come riferimento per la struttura dell'header
        if base_properties is None:
            base_properties = properties
            
        all_data.append(data)
        total_points += len(data)
    
    if not all_data:
        print("❌ Nessun dato valido da unire. Esco.")
        return
        
    # Concatena tutti gli array strutturati di NumPy
    print(f"\n🔄 Concatenamento dei punti in corso...")
    merged_data = np.concatenate(all_data)
    
    print(f"💾 Scrittura del modello unito ({total_points:,} punti totali) in {args.output}...")
    write_ply_gaussian(args.output, merged_data, base_properties)
    print("🏁 Merge completato con successo!")

if __name__ == "__main__":
    main()