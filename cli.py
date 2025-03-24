import argparse
import os
from datetime import datetime
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from core import (
    process_csv,
    generate_polygons,
    test_all_region_containments,
    remove_duplicate_containments,
    generate_clouds_for_all_regions,
    GraphCloud
)

OUTPUT_FOLDER = os.path.join('tmp', 'results')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_batch_mode(input_file, num, rand, mod, gen):
    if not os.path.exists(input_file):
        print(f"Archivo no encontrado: {input_file}")
        return

    try:
        regions = process_csv(input_file)
        polygons = generate_polygons(regions)
        containment = test_all_region_containments(polygons)
        depurated = remove_duplicate_containments(containment)

        clouds = generate_clouds_for_all_regions(polygons, depurated, num, rand, mod, gen)

        if clouds.size == 0:
            print("No se generaron puntos.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f'plot_{timestamp}.png'
        eps_name = f'plot_{timestamp}.eps'
        csv_name = f'cloud_{timestamp}.csv'

        GraphCloud(clouds, OUTPUT_FOLDER, image_name, eps_name)

        pd.DataFrame(clouds, columns=["x", "y", "region", "boundary_flag"]).to_csv(
            os.path.join(OUTPUT_FOLDER, csv_name), index=False
        )

        print("Nube generada exitosamente:")
        print(f"- CSV:  {csv_name}")
        print(f"- PNG:  {image_name}")
        print(f"- EPS:  {eps_name}")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generador batch de nubes de puntos desde CSV.")
    parser.add_argument('--input', type=str, required=True, help="Archivo CSV con puntos de entrada.")
    parser.add_argument('--num', type=int, default=3, help="Densidad de puntos (mayor = más puntos).")
    parser.add_argument('--rand', type=int, default=1, help="Perturbación aleatoria (0 = no, 1 = sí).")
    parser.add_argument('--mod', type=int, default=1, help="Método: 0 = dmsh, 1 = grid.")
    parser.add_argument('--gen', type=int, default=0, help="0 = solo exterior, 1 = todas las regiones.")

    args = parser.parse_args()
    run_batch_mode(args.input, args.num, args.rand, args.mod, args.gen)