import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None):
    """
    Configura y devuelve un logger personalizado con un formato específico.

    Parameters:
        name (str):          Nombre del logger
        log_file (str):      Ruta del archivo de log (opcional)

    Returns:
        logging.Logger:       Logger configurado
    """
    # Crear el logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formato del log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configurar handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Configurar handler para archivo si se especifica
    if log_file:
        # Asegurar que el directorio de logs existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Crear logger principal de la aplicación
app_logger = setup_logger(
    'mGFD_CloudGenerator',
    os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d")}.log')
)

def log_error(error, context=""):
    """
    Registra un error con contexto adicional.

    Parameters:
        error (Exception):    La excepción a registrar
        context (str):        Contexto adicional del error
    """
    error_message = f"{context} - {str(error)}"
    app_logger.error(error_message)
    return error_message