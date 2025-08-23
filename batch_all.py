import os
import pandas as pd
from datetime import datetime
from core import (
    process_csv, generate_polygons, test_all_region_containments,
    remove_duplicate_containments, generate_clouds_for_all_regions, GraphCloud
)

# Configuración fija para todos los casos
NUM = 1
RAND = 0
MOD = 0
GEN = 0

# Ruta raíz donde están los archivos de entrada
ROOT_DIR = os.path.join('Examples', 'Lakes')

# Recorre todos los subdirectorios buscando archivos *_Contour.csv
for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    for file in filenames:
        if file.endswith('_Contour.csv'):
            file_path = os.path.join(dirpath, file)
            print(f"Procesando: {file_path}")

            try:
                regions = process_csv(file_path)
                polygons = generate_polygons(regions)
                containment = test_all_region_containments(polygons)
                depurated = remove_duplicate_containments(containment)

                clouds = generate_clouds_for_all_regions(polygons, depurated, NUM, RAND, MOD, GEN)

                if clouds.size == 0:
                    print(f"\tNo se generaron puntos para {file}")
                    continue

                base_name = file.replace('_Contour.csv', '')
                image_name = f'{base_name}_Cloud_Adaptive.png'
                eps_name = f'{base_name}_Cloud_Adaptive.eps'
                csv_name  = f'{base_name}_Cloud_Adaptive.csv'

                GraphCloud(clouds, folder=dirpath, image_name=image_name, eps_name=eps_name)

                df = pd.DataFrame(clouds, columns=["x", "y", "region", "boundary_flag"])
                df.to_csv(os.path.join(dirpath, csv_name), index=False)

                print(f"\tListo → CSV: {csv_name}, PNG: {image_name}")

            except Exception as e:
                print(f"\tError procesando {file}: {e}")