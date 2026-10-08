"""Export and validate the immutable piece data and known 466 board."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from validator import OFFICIAL_CLUES, load_bundle, piece_taxonomy, read_pieces, sha256, validate_board


def export_bundle(pieces_path: str | Path, record_path: str | Path, destination: str | Path) -> dict:
    pieces_path, record_path, destination = Path(pieces_path), Path(record_path), Path(destination)
    if destination.resolve() in (pieces_path.resolve().parent, record_path.resolve().parent):
        raise ValueError('Export must go to a separate staging directory')
    taxonomy = piece_taxonomy(read_pieces(pieces_path))
    metadata = {
        'name': 'Official Eternity II, five clues', 'size': 16,
        'state_encoding': '4*(one-based piece ID-1)+clockwise rotation',
        'cell_indexing': 'zero-based row-major', 'piece_file_sides': ['U','D','L','R'],
        'oriented_sides': ['U','R','D','L'], 'gray': 0,
        'fixed_clues': [{'cell': cell, 'state': state, 'piece_id': state//4+1, 'rotation_cw': state%4}
                        for cell, state in OFFICIAL_CLUES.items()],
        'input_sha256': {'pieces.txt': sha256(pieces_path), 'record466.json': sha256(record_path)},
        'taxonomy': taxonomy,
        'reference_source': 'https://stats.eternityathome.org/',
    }
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in ((pieces_path, 'pieces.txt'), (record_path, 'record466.json')):
        target = destination / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Refusing to replace different existing data: {target}')
        if not target.exists():
            shutil.copyfile(source, target)
    (destination / 'puzzle.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    bundle = load_bundle(destination)
    report = validate_board(bundle.record_board, bundle)
    if not report['valid'] or report['score'] != 466:
        raise ValueError('Exported reference failed independent validation')
    return {'destination': str(destination), 'reference_validation': report, 'input_sha256': metadata['input_sha256']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pieces', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent/'data')
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.pieces, args.record, args.destination), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
