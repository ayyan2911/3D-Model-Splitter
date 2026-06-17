#!/usr/bin/env python3
"""
COLMAP Splitter - Tagli su nuvola dritta, salvataggio in geometria originale

Caratteristiche:
- La nuvola viene ruotata (PCA) per essere visualizzata "dritta" e allineata
- I tagli vengono fatti sulla nuvola dritta (facile e preciso)
- Il salvataggio avviene NELLA GEOMETRIA ORIGINALE (inclinata)
- I tagli vengono trasformati automaticamente tra i due spazi
- Overlap impostato da terminale (--overlap)
"""

import numpy as np
import struct
import argparse
import shutil
import json
import copy
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# ============================================================================
# CLASSI PER I DATI COLMAP
# ============================================================================

class Point3D:
    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs


class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


class Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


# ============================================================================
# FUNZIONI DI I/O PER FILE BINARI COLMAP
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
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            points3D[prop[0]] = Point3D(
                prop[0], np.array(prop[1:4]), np.array(prop[4:7]), prop[7],
                np.array(tuple(map(int, track_elems[0::2]))),
                np.array(tuple(map(int, track_elems[1::2])))
            )
    return points3D


def write_points3D_binary(points3D, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("Q", len(points3D)))
        for p in points3D.values():
            fid.write(struct.pack("Q", p.id))
            fid.write(struct.pack("ddd", *p.xyz))
            fid.write(struct.pack("BBB", *p.rgb))
            fid.write(struct.pack("d", p.error))
            fid.write(struct.pack("Q", len(p.image_ids)))
            for img_id, p2d_idx in zip(p.image_ids, p.point2D_idxs):
                fid.write(struct.pack("ii", img_id, p2d_idx))


def read_cameras_binary(path):
    cameras = {}
    num_params_map = {0: 3, 1: 4, 2: 5, 3: 5, 4: 4, 5: 5, 6: 8, 7: 10, 8: 10, 9: 10}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            prop = read_next_bytes(fid, 24, "iiQQ")
            num_params = num_params_map[prop[1]]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[prop[0]] = Camera(prop[0], prop[1], prop[2], prop[3], np.array(params))
    return cameras


def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("Q", len(cameras)))
        for c in cameras.values():
            fid.write(struct.pack("iiQQ", c.id, c.model, c.width, c.height))
            fid.write(struct.pack("d" * len(c.params), *c.params))


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            prop = read_next_bytes(fid, 64, "idddddddi")
            image_name = ""
            char = read_next_bytes(fid, 1, "c")[0]
            while char != b"\x00":
                image_name += char.decode("utf-8")
                char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3]))
            ])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[prop[0]] = Image(
                prop[0], np.array(prop[1:5]), np.array(prop[5:8]),
                prop[8], image_name, xys, point3D_ids
            )
    return images


def write_images_binary(images, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("Q", len(images)))
        for img in images.values():
            fid.write(struct.pack("i", img.id))
            fid.write(struct.pack("dddd", *img.qvec))
            fid.write(struct.pack("ddd", *img.tvec))
            fid.write(struct.pack("i", img.camera_id))
            fid.write((img.name + "\x00").encode("utf-8"))
            fid.write(struct.pack("Q", len(img.point3D_ids)))
            for xy, p_id in zip(img.xys, img.point3D_ids):
                fid.write(struct.pack("ddq", xy[0], xy[1], p_id))


# ============================================================================
# FUNZIONI PER TRASFORMAZIONI GEOMETRICHE
# ============================================================================

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2*qvec[2]**2 - 2*qvec[3]**2,
         2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
         2*qvec[1]*qvec[3] + 2*qvec[0]*qvec[2]],
        [2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
         1 - 2*qvec[1]**2 - 2*qvec[3]**2,
         2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1]],
        [2*qvec[1]*qvec[3] - 2*qvec[0]*qvec[2],
         2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
         1 - 2*qvec[1]**2 - 2*qvec[2]**2]
    ])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def get_rotation_transform(points3D, images):
    """
    Calcola una trasformazione di rotazione per centrare e allineare il modello.
    La nuvola viene ruotata per essere visualizzata "dritta".
    """
    point_ids = list(points3D.keys())
    xyz = np.array([points3D[pid].xyz for pid in point_ids])
    
    # Centratura iniziale
    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid
    
    # Calcolo rotazione principale (PCA)
    cov = np.cov(xyz_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    R = eigvecs[:, idx]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    
    # Applicazione rotazione e nuova centratura
    xyz_rotated = xyz_centered @ R
    final_centroid = xyz_rotated.mean(axis=0)
    xyz_absolute_zero = xyz_rotated - final_centroid
    
    # Creazione copia ruotata dei punti (per la visualizzazione)
    rotated_points = copy.deepcopy(points3D)
    for pid, new_xyz in zip(point_ids, xyz_absolute_zero):
        rotated_points[pid].xyz = new_xyz
    
    # Rotazione delle camere (per la visualizzazione)
    rotated_images = copy.deepcopy(images)
    for img in rotated_images.values():
        R_world = qvec2rotmat(img.qvec)
        cam_center = -R_world.T @ img.tvec
        cam_centered = cam_center - centroid
        cam_rotated = cam_centered @ R
        cam_final = cam_rotated - final_centroid
        new_R = R_world @ R.T
        new_qvec = rotmat2qvec(new_R)
        new_tvec = -new_R @ cam_final
        img.qvec = new_qvec
        img.tvec = new_tvec
    
    transform_params = {
        "centroid": centroid.tolist(),
        "R": R.tolist(),
        "final_centroid": final_centroid.tolist()
    }
    
    return rotated_points, rotated_images, transform_params


# ============================================================================
# LOGICA DI TAGLIO CON OVERLAP CORRETTA
# ============================================================================

def get_primary_cell(x, y, x_cuts, y_cuts, x_min, x_max, y_min, y_max):
    """Trova la cella principale (senza overlap)"""
    # Trova indice X
    x_boundaries = [x_min] + sorted(x_cuts) + [x_max]
    x_idx = 0
    for i in range(len(x_boundaries) - 1):
        if x_boundaries[i] <= x <= x_boundaries[i + 1]:
            x_idx = i
            break
    
    # Trova indice Y
    y_boundaries = [y_min] + sorted(y_cuts) + [y_max]
    y_idx = 0
    for i in range(len(y_boundaries) - 1):
        if y_boundaries[i] <= y <= y_boundaries[i + 1]:
            y_idx = i
            break
    
    return y_idx * (len(x_cuts) + 1) + x_idx


def get_cells_with_overlap(x, y, x_cuts, y_cuts, x_min, x_max, y_min, y_max, overlap_frac=0.15):
    """
    Restituisce la cella principale e le celle adiacenti se il punto è 
    nell'area di overlap.
    """
    primary = get_primary_cell(x, y, x_cuts, y_cuts, x_min, x_max, y_min, y_max)
    cells = [primary]
    
    if overlap_frac == 0:
        return cells
    
    num_x_cells = len(x_cuts) + 1
    num_y_cells = len(y_cuts) + 1
    
    # Calcola le soglie di overlap per ogni bordo
    x_boundaries = [x_min] + sorted(x_cuts) + [x_max]
    y_boundaries = [y_min] + sorted(y_cuts) + [y_max]
    
    # Controlla se il punto è vicino a un taglio X (entro overlap)
    for i in range(1, len(x_boundaries) - 1):
        cut_pos = x_boundaries[i]
        # Calcola larghezza della cella a sinistra e destra
        left_width = x_boundaries[i] - x_boundaries[i-1]
        right_width = x_boundaries[i+1] - x_boundaries[i]
        overlap_dist = max(left_width, right_width) * overlap_frac
        
        if abs(x - cut_pos) <= overlap_dist:
            # Il punto è nell'overlap, aggiungi la cella adiacente
            if x < cut_pos and i < num_x_cells:
                # A sinistra del taglio, aggiungi cella a destra
                adj_cell = primary + 1
                if adj_cell % num_x_cells != 0:  # Non oltre bordo destro
                    cells.append(adj_cell)
            elif x > cut_pos and i > 0:
                # A destra del taglio, aggiungi cella a sinistra
                adj_cell = primary - 1
                if adj_cell % num_x_cells != num_x_cells - 1:  # Non oltre bordo sinistro
                    cells.append(adj_cell)
    
    # Controlla se il punto è vicino a un taglio Y (entro overlap)
    for i in range(1, len(y_boundaries) - 1):
        cut_pos = y_boundaries[i]
        # Calcola altezza della cella sopra e sotto
        below_height = y_boundaries[i] - y_boundaries[i-1]
        above_height = y_boundaries[i+1] - y_boundaries[i]
        overlap_dist = max(below_height, above_height) * overlap_frac
        
        if abs(y - cut_pos) <= overlap_dist:
            # Il punto è nell'overlap, aggiungi la cella adiacente
            if y < cut_pos and i < num_y_cells:
                # Sotto il taglio, aggiungi cella sopra
                adj_cell = primary + num_x_cells
                if adj_cell < num_x_cells * num_y_cells:
                    cells.append(adj_cell)
            elif y > cut_pos and i > 0:
                # Sopra il taglio, aggiungi cella sotto
                adj_cell = primary - num_x_cells
                if adj_cell >= 0:
                    cells.append(adj_cell)
    
    return list(set(cells))  # Rimuovi duplicati


def get_cell_label(x, y, x_cuts, y_cuts):
    """Versione semplice senza overlap per la visualizzazione"""
    x_idx = sum(1 for c in sorted(x_cuts) if x > c)
    y_idx = sum(1 for c in sorted(y_cuts) if y > c)
    return y_idx * (len(x_cuts) + 1) + x_idx


def balance_cuts(cuts_raw, all_coords):
    """
    Dato un click approssimativo, ribilancia TUTTI i tagli sull'asse in modo
    che ogni cella contenga circa lo stesso numero di punti.
    """
    n_cuts = len(cuts_raw)
    n_cells = n_cuts + 1
    percentiles = [100.0 * i / n_cells for i in range(1, n_cells)]
    balanced = [float(np.percentile(all_coords, p)) for p in percentiles]
    return sorted(balanced)


# ============================================================================
# FUNZIONE PER SALVARE I CLUSTER BOUNDS
# ============================================================================

def save_cluster_bounds(output_path, x_cuts, y_cuts, x_min, x_max, y_min, y_max, z_min, z_max, overlap_pct):
    """
    Salva i bounds delle partizioni in formato JSON per clean_gaussian.py
    """
    overlap_frac = overlap_pct / 100.0
    
    # Costruisci i bounds per ogni parte
    parts = []
    part_index = 0
    
    x_boundaries = [x_min] + sorted(x_cuts) + [x_max]
    y_boundaries = [y_min] + sorted(y_cuts) + [y_max]
    
    for yi in range(len(y_boundaries) - 1):
        for xi in range(len(x_boundaries) - 1):
            # Bounds originali (senza overlap)
            original_bounds = {
                "min_x": x_boundaries[xi],
                "max_x": x_boundaries[xi + 1],
                "min_y": y_boundaries[yi],
                "max_y": y_boundaries[yi + 1],
                "min_z": z_min,
                "max_z": z_max
            }
            
            # Bounds con overlap
            width_x = original_bounds["max_x"] - original_bounds["min_x"]
            width_y = original_bounds["max_y"] - original_bounds["min_y"]
            
            bounds_with_overlap = {
                "min_x": original_bounds["min_x"] - (width_x * overlap_frac if xi > 0 else 0),
                "max_x": original_bounds["max_x"] + (width_x * overlap_frac if xi < len(x_boundaries) - 2 else 0),
                "min_y": original_bounds["min_y"] - (width_y * overlap_frac if yi > 0 else 0),
                "max_y": original_bounds["max_y"] + (width_y * overlap_frac if yi < len(y_boundaries) - 2 else 0),
                "min_z": z_min,
                "max_z": z_max
            }
            
            part = {
                "part_index": part_index,
                "original_bounds": original_bounds,
                "bounds": bounds_with_overlap
            }
            parts.append(part)
            part_index += 1
    
    bounds_data = {
        "parts": parts,
        "overlap": overlap_frac,
        "x_cuts": x_cuts,
        "y_cuts": y_cuts,
        "global_bounds": {
            "min_x": x_min, "max_x": x_max,
            "min_y": y_min, "max_y": y_max,
            "min_z": z_min, "max_z": z_max
        }
    }
    
    # Salva il file
    bounds_path = Path(output_path) / "cluster_bounds.json"
    with open(bounds_path, "w") as f:
        json.dump(bounds_data, f, indent=2)
    
    print(f"✅ Salvato cluster_bounds.json in {bounds_path}")
    return bounds_path


# ============================================================================
# APPLICAZIONE PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dividi un modello COLMAP - Tagli su nuvola dritta, salvataggio originale"
    )
    parser.add_argument("colmap_dir", help="Cartella contenente l'output COLMAP")
    parser.add_argument("output_dir", help="Cartella di output per le parti salvate")
    parser.add_argument("--overlap", type=float, default=15.0,
                        help="Percentuale di overlap tra le parti (default: 15%%)")
    args = parser.parse_args()

    sparse_base = Path(args.colmap_dir) / "sparse" / "0"

    print("📖 Lettura dei file COLMAP...")
    cameras = read_cameras_binary(sparse_base / "cameras.bin")
    images_original = read_images_binary(sparse_base / "images.bin")
    points3D_original = read_points3D_binary(sparse_base / "points3D.bin")

    print("🔄 Applicazione trasformazione di rotazione (nuvola dritta per UI)...")
    rotated_points, rotated_images, transform_params = get_rotation_transform(
        points3D_original, images_original
    )
    
    # Dati per la visualizzazione (nello spazio ruotato "dritto")
    p_xyz_rotated = np.array([p.xyz for p in rotated_points.values()])
    p_rgb = np.array([p.rgb for p in rotated_points.values()])

    # Calcolo bounds nello spazio ruotato (per UI)
    global_x_min_rot, global_x_max_rot = float(p_xyz_rotated[:, 0].min()), float(p_xyz_rotated[:, 0].max())
    global_y_min_rot, global_y_max_rot = float(p_xyz_rotated[:, 1].min()), float(p_xyz_rotated[:, 1].max())
    
    # Array completo per bilanciamento (nello spazio ruotato)
    all_xyz_rotated = p_xyz_rotated

    # Posizioni camere nello spazio ruotato (per UI)
    cam_centers_rotated = []
    for img in rotated_images.values():
        R_world = qvec2rotmat(img.qvec)
        cam_centers_rotated.append(-R_world.T @ img.tvec)
    cam_centers_rotated = np.array(cam_centers_rotated)

    # Sottocampionamento visualizzazione
    step = max(1, len(p_xyz_rotated) // 50000)
    display_xyz = p_xyz_rotated[::step]
    display_rgb = p_rgb[::step]
    original_colors = [f"rgb({r},{g},{b})" for r, g, b in display_rgb]

    # Overlap preso dal terminale
    overlap_pct = args.overlap
    overlap_frac = overlap_pct / 100.0

    # ========================================================================
    # DASH APPLICATION
    # ========================================================================

    app = dash.Dash(__name__, title="COLMAP Splitter - Tagli su nuvola dritta")

    app.index_string = '''
    <!DOCTYPE html>
    <html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background-color: #111418;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
            }
            button { transition: all 0.2s ease-in-out; cursor: pointer; }
            button:hover { filter: brightness(1.15); transform: translateY(-1px); }
            button:active { transform: translateY(1px); }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
    </html>
    '''

    app.layout = html.Div([
        html.Div([
            html.H2("⚡ 3D MODEL SPLITTER",
                    style={'color': '#FFFFFF', 'fontSize': '20px', 'fontWeight': '700',
                           'margin': '0 0 5px 0'}),
            html.P("Tagli su nuvola DRITTA · Salvataggio in geometria ORIGINALE",
                   style={'color': '#00F59B', 'fontSize': '12px', 'margin': '0 0 20px 0'}),

            html.Hr(style={'border': '0', 'borderTop': '1px solid #2A313D', 'margin': '0 0 20px 0'}),

            html.Label("✂️ MODALITÀ DI TAGLIO:",
                       style={'color': '#707E94', 'fontSize': '11px', 'fontWeight': '700',
                              'letterSpacing': '1px', 'display': 'block', 'marginBottom': '10px'}),
            dcc.RadioItems(
                id='line-type',
                options=[
                    {'label': ' ➕ Croce (X + Y)', 'value': 'both'},
                    {'label': ' 🟦 Solo Asse X',   'value': 'x_only'},
                    {'label': ' 🟩 Solo Asse Y',   'value': 'y_only'}
                ],
                value='both',
                labelStyle={'display': 'inline-block', 'marginRight': '15px',
                            'color': '#E2E8F0', 'fontSize': '13px'},
                style={'marginBottom': '20px'}
            ),

            html.Div(id='balance-info',
                     style={'color': '#10B981', 'fontSize': '11px', 'fontFamily': 'monospace',
                            'backgroundColor': '#0A1F18', 'padding': '8px', 'borderRadius': '6px',
                            'border': '1px solid #0D3025', 'marginBottom': '20px',
                            'whiteSpace': 'pre-wrap'}),

            # Info overlap (mostra il valore dal terminale)
            html.Div([
                html.Div(
                    f"🔀 OVERLAP: {overlap_pct}% (impostato da terminale)",
                    style={'color': '#00F59B', 'fontSize': '12px', 'fontFamily': 'monospace',
                           'backgroundColor': '#0A1F18', 'padding': '10px', 'borderRadius': '6px',
                           'border': '1px solid #00F59B33', 'marginBottom': '20px',
                           'textAlign': 'center'}
                )
            ]),

            html.Div([
                html.Span("📊 PARTI TOTALI:",
                          style={'color': '#A0AEC0', 'fontSize': '13px', 'fontWeight': '500'}),
                html.Span(id="total-parts", children="1",
                          style={'color': '#00F59B', 'fontSize': '28px', 'fontWeight': '700',
                                 'float': 'right', 'lineHeight': '1'})
            ], style={'backgroundColor': '#1A212D', 'padding': '15px', 'borderRadius': '10px',
                      'border': '1px solid #2D3748', 'marginBottom': '20px'}),

            html.Div(id='overlap-info',
                     style={'color': '#F59E0B', 'fontSize': '12px', 'fontFamily': 'monospace',
                            'backgroundColor': '#1A1600', 'padding': '10px', 'borderRadius': '6px',
                            'border': '1px solid #3D3000', 'marginBottom': '15px',
                            'whiteSpace': 'pre-wrap'}),

            html.Div([
                html.Label("📍 TAGLI ATTUALI (su nuvola dritta)",
                           style={'color': '#707E94', 'fontSize': '11px', 'fontWeight': '700',
                                  'letterSpacing': '1px', 'display': 'block', 'marginBottom': '8px'}),
                html.Div(id="debug",
                         style={'color': '#E2E8F0', 'fontSize': '12px', 'fontFamily': 'monospace',
                                'backgroundColor': '#111418', 'padding': '10px', 'borderRadius': '6px',
                                'border': '1px solid #232934', 'whiteSpace': 'pre-wrap'})
            ], style={'marginBottom': '30px'}),

            html.Div([
                html.Button("🗑️ CANCELLA TUTTI I TAGLI", id="btn-reset", n_clicks=0,
                            style={'width': '100%', 'padding': '12px', 'backgroundColor': '#2A313D',
                                   'color': '#E2E8F0', 'border': '1px solid #3E4756',
                                   'borderRadius': '8px', 'fontSize': '13px', 'fontWeight': '600',
                                   'marginBottom': '10px'}),
                html.Button("↩️ ANNULLA ULTIMO TAGLIO", id="btn-undo", n_clicks=0,
                            style={'width': '100%', 'padding': '12px', 'backgroundColor': '#3A2A2A',
                                   'color': '#FFAAAA', 'border': '1px solid #5E3E3E',
                                   'borderRadius': '8px', 'fontSize': '13px', 'fontWeight': '600',
                                   'marginBottom': '10px'}),
                html.Button("💾 SALVA (geometria originale)", id="btn-save", n_clicks=0,
                            style={'width': '100%', 'padding': '15px', 'backgroundColor': '#00F59B',
                                   'color': '#0A0E14', 'border': 'none', 'borderRadius': '8px',
                                   'fontWeight': '700', 'fontSize': '14px',
                                   'boxShadow': '0 4px 15px rgba(0, 245, 155, 0.2)'})
            ], style={'position': 'absolute', 'bottom': '25px', 'width': 'calc(100% - 50px)'}),
            
            html.Div(id="save-msg", style={'marginTop': '12px', 'padding': '12px',
                                           'borderRadius': '8px', 'fontWeight': '600',
                                           'fontSize': '13px', 'textAlign': 'center'})

        ], style={'width': '26%', 'position': 'fixed', 'top': 0, 'left': 0,
                  'backgroundColor': '#161B25', 'padding': '25px', 'height': '100vh',
                  'boxSizing': 'border-box', 'zIndex': '10', 'borderRight': '1px solid #222936',
                  'overflowY': 'auto'}),

        html.Div([
            dcc.Graph(id='graph', style={'height': '100vh', 'backgroundColor': '#111418'},
                      config={'scrollZoom': True, 'displaymodeBar': True, 'responsive': True})
        ], style={'width': '74%', 'float': 'right', 'backgroundColor': '#111418'}),

        dcc.Store(id='x-cuts',    data=[]),
        dcc.Store(id='y-cuts',    data=[]),
        dcc.Store(id='history-x', data=[]),
        dcc.Store(id='history-y', data=[]),
    ], style={'backgroundColor': '#111418', 'margin': '0'})

    # ========================================================================
    # CALLBACK PRINCIPALE
    # ========================================================================

    @app.callback(
        [Output('graph',        'figure'),
         Output('total-parts',  'children'),
         Output('debug',        'children'),
         Output('overlap-info', 'children'),
         Output('balance-info', 'children'),
         Output('x-cuts',       'data'),
         Output('y-cuts',       'data'),
         Output('history-x',    'data'),
         Output('history-y',    'data')],
        [Input('graph',          'clickData'),
         Input('btn-reset',      'n_clicks'),
         Input('btn-undo',       'n_clicks')],
        [State('x-cuts',      'data'),
         State('y-cuts',      'data'),
         State('history-x',   'data'),
         State('history-y',   'data'),
         State('line-type',   'value'),
         State('graph',       'relayoutData')]
    )
    def update(clickData, reset_clicks, undo_clicks,
               x_cuts, y_cuts, history_x, history_y, line_type, relayoutData):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        current_x = x_cuts.copy() if x_cuts else []
        current_y = y_cuts.copy() if y_cuts else []

        if trigger_id == 'btn-reset':
            x_cuts, y_cuts = [], []
            history_x, history_y = [], []

        elif trigger_id == 'btn-undo':
            if history_x or history_y:
                x_cuts   = history_x[-1] if history_x else []
                y_cuts   = history_y[-1] if history_y else []
                history_x = history_x[:-1] if history_x else []
                history_y = history_y[:-1] if history_y else []
            else:
                x_cuts, y_cuts = [], []

        elif trigger_id == 'graph' and clickData and 'points' in clickData:
            pt = clickData['points'][0]
            history_x.append(current_x)
            history_y.append(current_y)

            # Modalità solo bilanciata (rimossa quella manuale)
            if line_type in ['both', 'x_only']:
                new_x_cuts = sorted(set(x_cuts + [float(pt['x'])]))
                x_cuts = balance_cuts(new_x_cuts, all_xyz_rotated[:, 0])
            if line_type in ['both', 'y_only']:
                new_y_cuts = sorted(set(y_cuts + [float(pt['y'])]))
                y_cuts = balance_cuts(new_y_cuts, all_xyz_rotated[:, 1])

        num_x_cells = len(x_cuts) + 1
        num_y_cells = len(y_cuts) + 1
        total_parts = num_x_cells * num_y_cells

        palette = ['#FF5A5A', '#3B82F6', '#10B981', '#F59E0B',
                   '#8B5CF6', '#EC4899', '#06B6D4', '#F97316',
                   '#14B8A6', '#D946EF', '#6366F1', '#EAB308']

        if not x_cuts and not y_cuts:
            colors = original_colors
        else:
            colors = []
            for x, y in zip(display_xyz[:, 0], display_xyz[:, 1]):
                label = get_cell_label(x, y, x_cuts, y_cuts)
                colors.append(palette[label % len(palette)])

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=display_xyz[:, 0], y=display_xyz[:, 1], z=display_xyz[:, 2],
            mode='markers', marker=dict(size=1.0, color=colors, opacity=0.85),
            name='Punti', hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=cam_centers_rotated[:, 0], y=cam_centers_rotated[:, 1], z=cam_centers_rotated[:, 2],
            mode='markers', marker=dict(size=2, color='#FF3333', symbol='diamond', opacity=0.9),
            name='Camere', hoverinfo='none'
        ))

        if x_cuts or y_cuts:
            x_min_d = display_xyz[:, 0].min()
            x_max_d = display_xyz[:, 0].max()
            y_min_d = display_xyz[:, 1].min()
            y_max_d = display_xyz[:, 1].max()
            z_min_d = display_xyz[:, 2].min()
            z_max_d = display_xyz[:, 2].max()

            for xc in x_cuts:
                fig.add_trace(go.Mesh3d(
                    x=[xc, xc, xc, xc],
                    y=[y_min_d, y_max_d, y_min_d, y_max_d],
                    z=[z_min_d, z_min_d, z_max_d, z_max_d],
                    opacity=0.25, color='#00F59B', name=f"X={xc:.2f}", hoverinfo='skip'
                ))

            for yc in y_cuts:
                fig.add_trace(go.Mesh3d(
                    x=[x_min_d, x_max_d, x_min_d, x_max_d],
                    y=[yc, yc, yc, yc],
                    z=[z_min_d, z_min_d, z_max_d, z_max_d],
                    opacity=0.25, color='#3B82F6', name=f"Y={yc:.2f}", hoverinfo='skip'
                ))

        scene_dict = dict(
            aspectmode='data',
            xaxis=dict(backgroundcolor="#111418", gridcolor="#222936",
                       showbackground=False, zerolinecolor="#222936", color="#707E94"),
            yaxis=dict(backgroundcolor="#111418", gridcolor="#222936",
                       showbackground=False, zerolinecolor="#222936", color="#707E94"),
            zaxis=dict(backgroundcolor="#111418", gridcolor="#222936",
                       showbackground=False, zerolinecolor="#222936", color="#707E94")
        )
        if relayoutData and 'scene.camera' in relayoutData:
            scene_dict['camera'] = relayoutData['scene.camera']

        fig.update_layout(
            scene=scene_dict,
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='#111418', plot_bgcolor='#111418',
            showlegend=False, uirevision=True
        )

        debug_txt = (
            f"🏷️ Tagli X su nuvola dritta ({len(x_cuts)}): {x_cuts if x_cuts else 'nessuno'}\n"
            f"🏷️ Tagli Y su nuvola dritta ({len(y_cuts)}): {y_cuts if y_cuts else 'nessuno'}\n"
            f"📐 Griglia: {num_x_cells} × {num_y_cells} = {total_parts} parti\n"
            f"🔀 Overlap: {overlap_pct}% (da terminale)"
        )

        # Info overlap
        overlap_info = f"🔀 Overlap attivo: {overlap_pct}% (i punti vicino ai tagli vengono duplicati)"

        # Info bilanciamento sempre mostrata
        if x_cuts or y_cuts:
            total_pts = len(all_xyz_rotated)
            lines = ["⚖️ Distribuzione punti per cella (bilanciata):"]
            all_x_b = [global_x_min_rot] + sorted(x_cuts) + [global_x_max_rot]
            all_y_b = [global_y_min_rot] + sorted(y_cuts) + [global_y_max_rot]
            for yi in range(num_y_cells):
                for xi in range(num_x_cells):
                    mask = (
                        (all_xyz_rotated[:, 0] >= all_x_b[xi]) & (all_xyz_rotated[:, 0] < all_x_b[xi+1]) &
                        (all_xyz_rotated[:, 1] >= all_y_b[yi]) & (all_xyz_rotated[:, 1] < all_y_b[yi+1])
                    )
                    cnt = int(mask.sum())
                    pct = 100.0 * cnt / total_pts
                    lines.append(f"  cella ({xi},{yi}): {cnt:,} punti ({pct:.1f}%)")
            balance_info = "\n".join(lines)
        else:
            balance_info = "🎯 Clicca sulla nuvola per aggiungere tagli bilanciati (distribuzione uniforme dei punti)."

        return (fig, str(total_parts), debug_txt, overlap_info, balance_info,
                x_cuts, y_cuts, history_x, history_y)

    # ========================================================================
    # CALLBACK SALVATAGGIO CON CLUSTER BOUNDS
    # ========================================================================

    @app.callback(
        [Output("save-msg", "children"),
         Output("save-msg", "style")],
        [Input("btn-save", "n_clicks")],
        [State("x-cuts", "data"),
         State("y-cuts", "data")]
    )
    def save(n_clicks, x_cuts_rotated, y_cuts_rotated):
        if not n_clicks:
            return "", {'display': 'none'}

        if not x_cuts_rotated and not y_cuts_rotated:
            return "⚠️ Aggiungi almeno un taglio!", {
                'color': '#FF5A5A',
                'backgroundColor': 'rgba(255,90,90,0.1)',
                'border': '1px solid rgba(255,90,90,0.2)',
                'display': 'block'
            }

        # Overlap preso dal terminale (args.overlap)
        overlap_pct = args.overlap
        ov_frac = overlap_pct / 100.0

        output_path = Path(args.output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

        # ordine stabile punti
        point_ids = np.array(list(points3D_original.keys()))

        points_xyz_rot = np.array([
            rotated_points[pid].xyz for pid in point_ids
        ])

        global_x_min = float(points_xyz_rot[:, 0].min())
        global_x_max = float(points_xyz_rot[:, 0].max())
        global_y_min = float(points_xyz_rot[:, 1].min())
        global_y_max = float(points_xyz_rot[:, 1].max())
        global_z_min = float(points_xyz_rot[:, 2].min())
        global_z_max = float(points_xyz_rot[:, 2].max())

        print(f"\n📊 STATISTICHE OVERLAP:")
        print(f"   X range: [{global_x_min:.3f}, {global_x_max:.3f}]")
        print(f"   Y range: [{global_y_min:.3f}, {global_y_max:.3f}]")
        print(f"   Z range: [{global_z_min:.3f}, {global_z_max:.3f}]")
        print(f"   X cuts: {x_cuts_rotated}")
        print(f"   Y cuts: {y_cuts_rotated}")
        print(f"   Overlap: {overlap_pct}% (da terminale)")

        # Mappa dei punti per cella
        label_map: dict[int, list] = {}

        for pid, (x, y) in zip(point_ids, points_xyz_rot[:, :2]):
            if ov_frac > 0:
                # Con overlap: il punto può andare in più celle
                cell_labels = get_cells_with_overlap(
                    x, y, x_cuts_rotated, y_cuts_rotated,
                    global_x_min, global_x_max, global_y_min, global_y_max,
                    overlap_frac=ov_frac
                )
            else:
                # Senza overlap: il punto va in una sola cella
                cell_labels = [get_primary_cell(
                    x, y, x_cuts_rotated, y_cuts_rotated,
                    global_x_min, global_x_max, global_y_min, global_y_max
                )]
            
            for lbl in cell_labels:
                if lbl not in label_map:
                    label_map[lbl] = []
                label_map[lbl].append(pid)

        # Verifica che tutte le celle abbiano punti
        expected_cells = (len(x_cuts_rotated) + 1) * (len(y_cuts_rotated) + 1)
        print(f"\n📊 VERIFICA CELLULE:")
        print(f"   Celle attese: {expected_cells}")
        print(f"   Celle trovate: {len(label_map)}")
        
        # Crea celle vuote se necessario (per mantenere la struttura)
        for cell_idx in range(expected_cells):
            if cell_idx not in label_map:
                print(f"   ⚠️ Cella {cell_idx} vuota (creata vuota)")
                label_map[cell_idx] = []
        
        # Stampa distribuzione per debug
        print(f"\n📊 DISTRIBUZIONE PUNTI (con overlap {overlap_pct}%):")
        total_assignments = sum(len(pids) for pids in label_map.values())
        for cell_idx in sorted(label_map.keys()):
            count = len(label_map[cell_idx])
            pct = 100.0 * count / total_assignments if total_assignments > 0 else 0
            print(f"   Part {cell_idx}: {count:,} assignments ({pct:.1f}%)")

        images_dict = {img.id: img for img in images_original.values()}
        cameras_dict = {cam.id: cam for cam in cameras.values()}

        print(f"\n📌 SALVATAGGIO PARTI:")
        print(f"   Partizioni trovate: {len(label_map)}")

        for label, cell_pids in label_map.items():
            if not cell_pids:
                # Crea comunque la directory per celle vuote
                part_dir = output_path / f"part_{label}"
                sparse_dir = part_dir / "sparse" / "0"
                images_dir = part_dir / "images"
                sparse_dir.mkdir(parents=True, exist_ok=True)
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # Scrivi file vuoti
                write_cameras_binary({}, sparse_dir / "cameras.bin")
                write_images_binary({}, sparse_dir / "images.bin")
                write_points3D_binary({}, sparse_dir / "points3D.bin")
                print(f" ⚠️ Part {label}: 0 punti (vuota)")
                continue

            part_dir = output_path / f"part_{label}"
            sparse_dir = part_dir / "sparse" / "0"
            images_dir = part_dir / "images"

            sparse_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            part_points = {pid: points3D_original[pid] for pid in cell_pids}

            img_ids = set()
            for pt in part_points.values():
                img_ids.update(pt.image_ids)

            part_images = {}
            for iid in img_ids:
                if iid not in images_dict:
                    continue

                img = images_dict[iid]
                valid = np.array([pid in part_points for pid in img.point3D_ids])

                if valid.any():
                    part_images[iid] = Image(
                        img.id,
                        img.qvec,
                        img.tvec,
                        img.camera_id,
                        img.name,
                        img.xys[valid],
                        img.point3D_ids[valid]
                    )

            cam_ids = {img.camera_id for img in part_images.values()}
            part_cameras = {cid: cameras_dict[cid] for cid in cam_ids if cid in cameras_dict}

            write_cameras_binary(part_cameras, sparse_dir / "cameras.bin")
            write_images_binary(part_images, sparse_dir / "images.bin")
            write_points3D_binary(part_points, sparse_dir / "points3D.bin")

            # copia immagini
            src_imgs = Path(args.colmap_dir) / "images"
            for img in part_images.values():
                src = src_imgs / img.name
                dst = images_dir / img.name
                if src.exists():
                    shutil.copy2(src, dst)

            print(f" ✅ Part {label}: {len(cell_pids):,} punti")

        # ====================================================================
        # SALVA IL FILE cluster_bounds.json
        # ====================================================================
        save_cluster_bounds(
            output_path,
            x_cuts_rotated,
            y_cuts_rotated,
            global_x_min, global_x_max,
            global_y_min, global_y_max,
            global_z_min, global_z_max,
            overlap_pct
        )

        # Messaggio di riepilogo per l'UI
        non_empty = len([p for p in label_map.values() if p])
        summary = f"🎉 Salvate {non_empty} parti non vuote su {expected_cells} totali!"
        summary += f"\n📌 Overlap {overlap_pct}% (impostato da terminale)"
        summary += f"\n📁 cluster_bounds.json salvato nella cartella di output"

        return (
            summary,
            {
                'color': '#00F59B',
                'backgroundColor': 'rgba(0,245,155,0.1)',
                'border': '1px solid rgba(0,245,155,0.2)',
                'display': 'block',
                'whiteSpace': 'pre-wrap'
            }
        )

    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()