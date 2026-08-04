from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app, send_file, url_for
import os
import cv2
import numpy as np
import csv
import time
import tempfile
import threading
from werkzeug.utils import secure_filename
from datetime import datetime


from . import main_bp

@main_bp.route('/')
def home():
    """
    Render the main homepage of the application.
    
    Returns:
        str: Rendered HTML template for the home page
    """
    return render_template('home.html')
@main_bp.route('/set_language/<lang>')
def set_language(lang):
    from flask import redirect, request, make_response
    if lang not in ['en', 'es']:
        lang = 'en'
    
    # Get the URL the user came from
    referer = request.referrer or url_for('main.home')
    
    # Create response and set cookie
    resp = make_response(redirect(referer))
    resp.set_cookie('lang', lang, max_age=60*60*24*365) # Valid for 1 year
    return resp


@main_bp.route('/examples')
def examples():
    """Render the Examples page with case studies."""
    from flask_babel import gettext as _
    
    examples_list = [
        {
            'name': 'Balkhash',
            'title': _('Lake Balkhash'),
            'description': _('One of the largest lakes in Asia, located in southeastern Kazakhstan. Uniquely, its western part is fresh water while the eastern part is saline.'),
            'image': 'Balkhash.png',
            'cloud_image': 'Balkhash_cloud.png',
            'cloud_svg': 'Balkhash_cloud.svg',
            'contours_csv': 'Balkhash_contours.csv',
            'cloud_csv': 'Balkhash_cloud.csv'
        },
        {
            'name': 'Caspio',
            'title': _('Caspian Sea'),
            'description': _('The world\'s largest inland body of water, often described as the world\'s largest lake or a full-fledged sea. It lies between Europe and Asia.'),
            'image': 'Caspio.png',
            'cloud_image': 'Caspio_cloud.png',
            'cloud_svg': 'Caspio_cloud.svg',
            'contours_csv': 'Caspio_contours.csv',
            'cloud_csv': 'Caspio_cloud.csv'
        },
        {
            'name': 'Catemaco',
            'title': _('Lake Catemaco'),
            'description': _('A freshwater lake located in south-central Veracruz, Mexico, formed by natural damming of volcanic origin.'),
            'image': 'Catemaco.png',
            'cloud_image': 'Catemaco_cloud.png',
            'cloud_svg': 'Catemaco_cloud.svg',
            'contours_csv': 'Catemaco_contours.csv',
            'cloud_csv': 'Catemaco_cloud.csv'
        },
        {
            'name': 'Huron',
            'title': _('Lake Huron'),
            'description': _('One of the five Great Lakes of North America, connecting to Lake Michigan by the Straits of Mackinac.'),
            'image': 'Huron.png',
            'cloud_image': 'Huron_cloud.png',
            'cloud_svg': 'Huron_cloud.svg',
            'contours_csv': 'Huron_contours.csv',
            'cloud_csv': 'Huron_cloud.csv'
        },
        {
            'name': 'Malawi',
            'title': _('Lake Malawi'),
            'description': _('An African Great Lake and the southernmost lake in the East African Rift system, located between Malawi, Mozambique and Tanzania.'),
            'image': 'Malawi.png',
            'cloud_image': 'Malawi_cloud.png',
            'cloud_svg': 'Malawi_cloud.svg',
            'contours_csv': 'Malawi_contours.csv',
            'cloud_csv': 'Malawi_cloud.csv'
        },
        {
            'name': 'Patzcuaro',
            'title': _('Lake Pátzcuaro'),
            'description': _('A lake in Michoacán, Mexico, famous for its cultural significance, islands, and traditional fishing.'),
            'image': 'Patzcuaro.png',
            'cloud_image': 'Patzcuaro_cloud.png',
            'cloud_svg': 'Patzcuaro_cloud.svg',
            'contours_csv': 'Patzcuaro_contours.csv',
            'cloud_csv': 'Patzcuaro_cloud.csv'
        },
        {
            'name': 'Poopo',
            'title': _('Lake Poopó'),
            'description': _('A large saline lake in a shallow depression in the Altiplano Mountains in Bolivia, known for its fluctuating water levels.'),
            'image': 'Poopo.png',
            'cloud_image': 'Poopo_cloud.png',
            'cloud_svg': 'Poopo_cloud.svg',
            'contours_csv': 'Poopo_contours.csv',
            'cloud_csv': 'Poopo_cloud.csv'
        },
        {
            'name': 'Santa_Maria_del_Oro',
            'title': _('Santa María del Oro'),
            'description': _('A crater lake located in the crater of a volcano in the state of Nayarit, Mexico.'),
            'image': 'Santa_Maria_del_Oro.png',
            'cloud_image': 'Santa_Maria_del_Oro_cloud.png',
            'cloud_svg': 'Santa_Maria_del_Oro_cloud.svg',
            'contours_csv': 'Santa_Maria_del_Oro_contours.csv',
            'cloud_csv': 'Santa_Maria_del_Oro_cloud.csv'
        },
        {
            'name': 'Titicaca',
            'title': _('Lake Titicaca'),
            'description': _('A large, deep, freshwater lake in the Andes on the border of Bolivia and Peru, often called the highest navigable lake in the world.'),
            'image': 'Titicaca.png',
            'cloud_image': 'Titicaca_cloud.png',
            'cloud_svg': 'Titicaca_cloud.svg',
            'contours_csv': 'Titicaca_contours.csv',
            'cloud_csv': 'Titicaca_cloud.csv'
        },
        {
            'name': 'Yuriria',
            'title': _('Lake Yuriria'),
            'description': _('A man-made lake in Guanajuato, Mexico, constructed in the 16th century, representing the first hydraulic work of the colonial period in America.'),
            'image': 'Yuriria.png',
            'cloud_image': 'Yuriria_cloud.png',
            'cloud_svg': 'Yuriria_cloud.svg',
            'contours_csv': 'Yuriria_contours.csv',
            'cloud_csv': 'Yuriria_cloud.csv'
        },
        {
            'name': 'Zirahuen',
            'title': _('Lake Zirahuén'),
            'description': _('A deep, endorheic basin lake in Michoacán, Mexico, known for its clear blue waters.'),
            'image': 'Zirahuen.png',
            'cloud_image': 'Zirahuen_cloud.png',
            'cloud_svg': 'Zirahuen_cloud.svg',
            'contours_csv': 'Zirahuen_contours.csv',
            'cloud_csv': 'Zirahuen_cloud.csv'
        }
    ]
    return render_template('examples.html', examples=examples_list)


@main_bp.route('/about')
def about():
    """
    Render the about page with application information and documentation.
    
    Returns:
        str: Rendered HTML template for the about page
    """
    return render_template('about.html')


@main_bp.route('/privacy_notice')
def privacy_notice():
    """
    Render the privacy notice page (Aviso de Privacidad) compliant with Mexican law.
    
    Returns:
        str: Rendered HTML template for the privacy notice page
    """
    return render_template('privacy_notice.html')


