# -*- coding: utf-8 -*-
from pkg_resources import resource_filename
import api

img_path = {
    1:"1小玉/1 (5).jpg",
    2:"2小惠/2 (6).jpg",
    3:"3小晨/3 (4).jpg",
    4:"4小锐/4 (7).jpg",
    5:"5小木/5 (6).jpg",
    6:"6小满/6 (7).jpg",
    7:"7小健/7 (8).jpg"
}

def encodings(seq):
    path = img_path[seq]
    return api.face_encodings(api.load_image_file(resource_filename(__name__, path)))[0]