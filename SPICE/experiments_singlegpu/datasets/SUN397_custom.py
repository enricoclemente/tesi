import os
import collections

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np
import csv


SUN397_total_images = 108754
"""
    Scenes UNderstanding of 397 Scenes (SUN397) Dataset implementation for pytorch
    site: https://vision.princeton.edu/projects/2010/SUN/
    paper: https://vision.princeton.edu/projects/2010/SUN/paperIJCV.pdf
    The dataset has 397 scenes. Images are not equally distributed in scenes since
    some scenes are more frequent in every-day life then others

    Annotations are made for each scene, but there is also available a three-level hierarchy 
    annotation: https://vision.princeton.edu/projects/2010/SUN/hierarchy/

    Statistics:
        108754 images
        The two most smaller images are: 
            with the smallest H: sun397/SUN397/i/inn/outdoor/sun_auxzvqvzyapxfdmj.jpg W= 87 H= 65 
            with the smallest W: sun397/SUN397/i/inn/outdoor/sun_auxzvqvzyapxfdmj.jpg W= 87 H= 65
        Images distribution: 
            {'/a/abbey': 484, '/a/airplane_cabin': 115, '/a/airport_terminal': 1091, '/a/alley': 324, '/a/amphitheater': 316, '/a/amusement_arcade': 219, '/a/amusement_park': 750, '/a/anechoic_chamber': 173, '/a/apartment_building/outdoor': 527, '/a/apse/indoor': 164, '/a/aquarium': 169, '/a/aqueduct': 354, '/a/arch': 218, '/a/archive': 146, '/a/arrival_gate/outdoor': 164, '/a/art_gallery': 283, '/a/art_school': 139, '/a/art_studio': 310, '/a/assembly_line': 159, '/a/athletic_field/outdoor': 114, '/a/atrium/public': 127, '/a/attic': 332, '/a/auditorium': 340, '/a/auto_factory': 242, '/b/badlands': 463, '/b/badminton_court/indoor': 209, '/b/baggage_claim': 221, '/b/bakery/shop': 482, '/b/balcony/exterior': 125, '/b/balcony/interior': 177, '/b/ball_pit': 202, '/b/ballroom': 112, '/b/bamboo_forest': 141, '/b/banquet_hall': 296, '/b/bar': 738, '/b/barn': 279, '/b/barndoor': 140, '/b/baseball_field': 287, '/b/basement': 152, '/b/basilica': 295, '/b/basketball_court/outdoor': 111, '/b/bathroom': 951, '/b/batters_box': 117, '/b/bayou': 166, '/b/bazaar/indoor': 156, '/b/bazaar/outdoor': 111, '/b/beach': 1194, '/b/beauty_salon': 549, '/b/bedroom': 2084, '/b/berth': 120, '/b/biology_laboratory': 100, '/b/bistro/indoor': 100, '/b/boardwalk': 144, '/b/boat_deck': 123, '/b/boathouse': 111, '/b/bookstore': 210, '/b/booth/indoor': 483, '/b/botanical_garden': 263, '/b/bow_window/indoor': 211, '/b/bow_window/outdoor': 131, '/b/bowling_alley': 410, '/b/boxing_ring': 124, '/b/brewery/indoor': 124, '/b/bridge': 847, '/b/building_facade': 325, '/b/bullring': 187, '/b/burial_chamber': 112, '/b/bus_interior': 182, '/b/butchers_shop': 227, '/b/butte': 139, '/c/cabin/outdoor': 318, '/c/cafeteria': 104, '/c/campsite': 476, '/c/campus': 174, '/c/canal/natural': 193, '/c/canal/urban': 490, '/c/candy_store': 259, '/c/canyon': 255, '/c/car_interior/backseat': 222, '/c/car_interior/frontseat': 125, '/c/carrousel': 392, '/c/casino/indoor': 590, '/c/castle': 1100, '/c/catacomb': 135, '/c/cathedral/indoor': 238, '/c/cathedral/outdoor': 635, '/c/cavern/indoor': 430, '/c/cemetery': 569, '/c/chalet': 314, '/c/cheese_factory': 101, '/c/chemistry_lab': 174, '/c/chicken_coop/indoor': 102, '/c/chicken_coop/outdoor': 287, '/c/childs_room': 159, '/c/church/indoor': 236, '/c/church/outdoor': 921, '/c/classroom': 225, '/c/clean_room': 130, '/c/cliff': 234, '/c/cloister/indoor': 199, '/c/closet': 300, '/c/clothing_store': 336, '/c/coast': 458, '/c/cockpit': 695, '/c/coffee_shop': 161, '/c/computer_room': 186, '/c/conference_center': 105, '/c/conference_room': 833, '/c/construction_site': 527, '/c/control_room': 100, '/c/control_tower/outdoor': 445, '/c/corn_field': 339, '/c/corral': 134, '/c/corridor': 332, '/c/cottage_garden': 229, '/c/courthouse': 376, '/c/courtroom': 147, '/c/courtyard': 100, '/c/covered_bridge/exterior': 565, '/c/creek': 547, '/c/crevasse': 172, '/c/crosswalk': 186, '/c/cubicle/office': 142, '/d/dam': 154, '/d/delicatessen': 242, '/d/dentists_office': 166, '/d/desert/sand': 313, '/d/desert/vegetation': 195, '/d/diner/indoor': 164, '/d/diner/outdoor': 118, '/d/dinette/home': 117, '/d/dinette/vehicle': 197, '/d/dining_car': 243, '/d/dining_room': 1180, '/d/discotheque': 137, '/d/dock': 130, '/d/doorway/outdoor': 550, '/d/dorm_room': 165, '/d/driveway': 281, '/d/driving_range/outdoor': 125, '/d/drugstore': 126, '/e/electrical_substation': 101, '/e/elevator/door': 139, '/e/elevator/interior': 127, '/e/elevator_shaft': 121, '/e/engine_room': 125, '/e/escalator/indoor': 394, '/e/excavation': 334, '/f/factory/indoor': 101, '/f/fairway': 259, '/f/fastfood_restaurant': 169, '/f/field/cultivated': 343, '/f/field/wild': 248, '/f/fire_escape': 294, '/f/fire_station': 283, '/f/firing_range/indoor': 111, '/f/fishpond': 167, '/f/florist_shop/indoor': 154, '/f/food_court': 103, '/f/forest/broadleaf': 265, '/f/forest/needleleaf': 162, '/f/forest_path': 502, '/f/forest_road': 152, '/f/formal_garden': 251, '/f/fountain': 212, '/g/galley': 265, '/g/game_room': 221, '/g/garage/indoor': 171, '/g/garbage_dump': 100, '/g/gas_station': 328, '/g/gazebo/exterior': 803, '/g/general_store/indoor': 101, '/g/general_store/outdoor': 161, '/g/gift_shop': 140, '/g/golf_course': 814, '/g/greenhouse/indoor': 357, '/g/greenhouse/outdoor': 136, '/g/gymnasium/indoor': 372, '/h/hangar/indoor': 205, '/h/hangar/outdoor': 131, '/h/harbor': 250, '/h/hayfield': 144, '/h/heliport': 101, '/h/herb_garden': 221, '/h/highway': 833, '/h/hill': 144, '/h/home_office': 178, '/h/hospital': 120, '/h/hospital_room': 203, '/h/hot_spring': 116, '/h/hot_tub/outdoor': 241, '/h/hotel/outdoor': 120, '/h/hotel_room': 492, '/h/house': 955, '/h/hunting_lodge/outdoor': 103, '/i/ice_cream_parlor': 113, '/i/ice_floe': 207, '/i/ice_shelf': 133, '/i/ice_skating_rink/indoor': 253, '/i/ice_skating_rink/outdoor': 219, '/i/iceberg': 399, '/i/igloo': 166, '/i/industrial_area': 134, '/i/inn/outdoor': 120, '/i/islet': 330, '/j/jacuzzi/indoor': 111, '/j/jail/indoor': 193, '/j/jail_cell': 185, '/j/jewelry_shop': 174, '/k/kasbah': 179, '/k/kennel/indoor': 158, '/k/kennel/outdoor': 207, '/k/kindergarden_classroom': 144, '/k/kitchen': 1746, '/k/kitchenette': 133, '/l/labyrinth/outdoor': 140, '/l/lake/natural': 430, '/l/landfill': 219, '/l/landing_deck': 308, '/l/laundromat': 320, '/l/lecture_room': 192, '/l/library/indoor': 358, '/l/library/outdoor': 102, '/l/lido_deck/outdoor': 138, '/l/lift_bridge': 128, '/l/lighthouse': 489, '/l/limousine_interior': 108, '/l/living_room': 2361, '/l/lobby': 464, '/l/lock_chamber': 101, '/l/locker_room': 365, '/m/mansion': 454, '/m/manufactured_home': 266, '/m/market/indoor': 101, '/m/market/outdoor': 839, '/m/marsh': 193, '/m/martial_arts_gym': 183, '/m/mausoleum': 201, '/m/medina': 104, '/m/moat/water': 110, '/m/monastery/outdoor': 178, '/m/mosque/indoor': 124, '/m/mosque/outdoor': 536, '/m/motel': 254, '/m/mountain': 518, '/m/mountain_snowy': 736, '/m/movie_theater/indoor': 218, '/m/museum/indoor': 248, '/m/music_store': 110, '/m/music_studio': 369, '/n/nuclear_power_plant/outdoor': 116, '/n/nursery': 250, '/o/oast_house': 109, '/o/observatory/outdoor': 202, '/o/ocean': 232, '/o/office': 136, '/o/office_building': 300, '/o/oil_refinery/outdoor': 159, '/o/oilrig': 210, '/o/operating_room': 207, '/o/orchard': 368, '/o/outhouse/outdoor': 310, '/p/pagoda': 200, '/p/palace': 223, '/p/pantry': 565, '/p/park': 143, '/p/parking_garage/indoor': 100, '/p/parking_garage/outdoor': 111, '/p/parking_lot': 439, '/p/parlor': 329, '/p/pasture': 612, '/p/patio': 117, '/p/pavilion': 145, '/p/pharmacy': 159, '/p/phone_booth': 316, '/p/physics_laboratory': 163, '/p/picnic_area': 148, '/p/pilothouse/indoor': 222, '/p/planetarium/outdoor': 115, '/p/playground': 898, '/p/playroom': 244, '/p/plaza': 228, '/p/podium/indoor': 120, '/p/podium/outdoor': 143, '/p/pond': 179, '/p/poolroom/establishment': 125, '/p/poolroom/home': 397, '/p/power_plant/outdoor': 142, '/p/promenade_deck': 102, '/p/pub/indoor': 140, '/p/pulpit': 142, '/p/putting_green': 126, '/r/racecourse': 116, '/r/raceway': 145, '/r/raft': 219, '/r/railroad_track': 128, '/r/rainforest': 231, '/r/reception': 188, '/r/recreation_room': 118, '/r/residential_neighborhood': 110, '/r/restaurant': 796, '/r/restaurant_kitchen': 134, '/r/restaurant_patio': 443, '/r/rice_paddy': 100, '/r/riding_arena': 106, '/r/river': 228, '/r/rock_arch': 109, '/r/rope_bridge': 112, '/r/ruin': 327, '/r/runway': 180, '/s/sandbar': 135, '/s/sandbox': 203, '/s/sauna': 177, '/s/schoolhouse': 139, '/s/sea_cliff': 126, '/s/server_room': 116, '/s/shed': 288, '/s/shoe_shop': 181, '/s/shopfront': 725, '/s/shopping_mall/indoor': 284, '/s/shower': 125, '/s/skatepark': 100, '/s/ski_lodge': 112, '/s/ski_resort': 188, '/s/ski_slope': 260, '/s/sky': 167, '/s/skyscraper': 801, '/s/slum': 192, '/s/snowfield': 189, '/s/squash_court': 116, '/s/stable': 161, '/s/stadium/baseball': 280, '/s/stadium/football': 117, '/s/stage/indoor': 123, '/s/staircase': 680, '/s/street': 458, '/s/subway_interior': 455, '/s/subway_station/platform': 518, '/s/supermarket': 373, '/s/sushi_bar': 119, '/s/swamp': 143, '/s/swimming_pool/indoor': 256, '/s/swimming_pool/outdoor': 428, '/s/synagogue/indoor': 129, '/s/synagogue/outdoor': 147, '/t/television_studio': 197, '/t/temple/east_asia': 202, '/t/temple/south_asia': 155, '/t/tennis_court/indoor': 122, '/t/tennis_court/outdoor': 429, '/t/tent/outdoor': 223, '/t/theater/indoor_procenium': 237, '/t/theater/indoor_seats': 109, '/t/thriftshop': 100, '/t/throne_room': 114, '/t/ticket_booth': 131, '/t/toll_plaza': 103, '/t/topiary_garden': 112, '/t/tower': 704, '/t/toyshop': 388, '/t/track/outdoor': 136, '/t/train_railway': 276, '/t/train_station/platform': 119, '/t/tree_farm': 118, '/t/tree_house': 113, '/t/trench': 105, '/u/underwater/coral_reef': 494, '/u/utility_room': 144, '/v/valley': 145, '/v/van_interior': 131, '/v/vegetable_garden': 496, '/v/veranda': 100, '/v/veterinarians_office': 140, '/v/viaduct': 254, '/v/videostore': 125, '/v/village': 138, '/v/vineyard': 452, '/v/volcano': 126, '/v/volleyball_court/indoor': 107, '/v/volleyball_court/outdoor': 137, '/w/waiting_room': 476, '/w/warehouse/indoor': 777, '/w/water_tower': 396, '/w/waterfall/block': 210, '/w/waterfall/fan': 203, '/w/waterfall/plunge': 106, '/w/watering_hole': 107, '/w/wave': 274, '/w/wet_bar': 135, '/w/wheat_field': 247, '/w/wind_farm': 302, '/w/windmill': 413, '/w/wine_cellar/barrel_storage': 246, '/w/wine_cellar/bottle_storage': 340, '/w/wrestling_ring/indoor': 100, '/y/yard': 201, '/y/youth_hostel': 138}
            {'/a/abbey': {'train': 387, 'test': 97}, '/a/airplane_cabin': {'train': 92, 'test': 23}, '/a/airport_terminal': {'train': 872, 'test': 219}, '/a/alley': {'train': 259, 'test': 65}, '/a/amphitheater': {'train': 252, 'test': 64}, '/a/amusement_arcade': {'train': 175, 'test': 44}, '/a/amusement_park': {'train': 600, 'test': 150}, '/a/anechoic_chamber': {'train': 138, 'test': 35}, '/a/apartment_building/outdoor': {'train': 421, 'test': 106}, '/a/apse/indoor': {'train': 131, 'test': 33}, '/a/aquarium': {'train': 135, 'test': 34}, '/a/aqueduct': {'train': 283, 'test': 71}, '/a/arch': {'train': 174, 'test': 44}, '/a/archive': {'train': 116, 'test': 30}, '/a/arrival_gate/outdoor': {'train': 131, 'test': 33}, '/a/art_gallery': {'train': 226, 'test': 57}, '/a/art_school': {'train': 111, 'test': 28}, '/a/art_studio': {'train': 248, 'test': 62}, '/a/assembly_line': {'train': 127, 'test': 32}, '/a/athletic_field/outdoor': {'train': 91, 'test': 23}, '/a/atrium/public': {'train': 101, 'test': 26}, '/a/attic': {'train': 265, 'test': 67}, '/a/auditorium': {'train': 272, 'test': 68}, '/a/auto_factory': {'train': 193, 'test': 49}, '/b/badlands': {'train': 370, 'test': 93}, '/b/badminton_court/indoor': {'train': 167, 'test': 42}, '/b/baggage_claim': {'train': 176, 'test': 45}, '/b/bakery/shop': {'train': 385, 'test': 97}, '/b/balcony/exterior': {'train': 100, 'test': 25}, '/b/balcony/interior': {'train': 141, 'test': 36}, '/b/ball_pit': {'train': 161, 'test': 41}, '/b/ballroom': {'train': 89, 'test': 23}, '/b/bamboo_forest': {'train': 112, 'test': 29}, '/b/banquet_hall': {'train': 236, 'test': 60}, '/b/bar': {'train': 590, 'test': 148}, '/b/barn': {'train': 223, 'test': 56}, '/b/barndoor': {'train': 112, 'test': 28}, '/b/baseball_field': {'train': 229, 'test': 58}, '/b/basement': {'train': 121, 'test': 31}, '/b/basilica': {'train': 236, 'test': 59}, '/b/basketball_court/outdoor': {'train': 88, 'test': 23}, '/b/bathroom': {'train': 760, 'test': 191}, '/b/batters_box': {'train': 93, 'test': 24}, '/b/bayou': {'train': 132, 'test': 34}, '/b/bazaar/indoor': {'train': 124, 'test': 32}, '/b/bazaar/outdoor': {'train': 88, 'test': 23}, '/b/beach': {'train': 955, 'test': 239}, '/b/beauty_salon': {'train': 439, 'test': 110}, '/b/bedroom': {'train': 1667, 'test': 417}, '/b/berth': {'train': 96, 'test': 24}, '/b/biology_laboratory': {'train': 80, 'test': 20}, '/b/bistro/indoor': {'train': 80, 'test': 20}, '/b/boardwalk': {'train': 115, 'test': 29}, '/b/boat_deck': {'train': 98, 'test': 25}, '/b/boathouse': {'train': 88, 'test': 23}, '/b/bookstore': {'train': 168, 'test': 42}, '/b/booth/indoor': {'train': 386, 'test': 97}, '/b/botanical_garden': {'train': 210, 'test': 53}, '/b/bow_window/indoor': {'train': 168, 'test': 43}, '/b/bow_window/outdoor': {'train': 104, 'test': 27}, '/b/bowling_alley': {'train': 328, 'test': 82}, '/b/boxing_ring': {'train': 99, 'test': 25}, '/b/brewery/indoor': {'train': 99, 'test': 25}, '/b/bridge': {'train': 677, 'test': 170}, '/b/building_facade': {'train': 260, 'test': 65}, '/b/bullring': {'train': 149, 'test': 38}, '/b/burial_chamber': {'train': 89, 'test': 23}, '/b/bus_interior': {'train': 145, 'test': 37}, '/b/butchers_shop': {'train': 181, 'test': 46}, '/b/butte': {'train': 111, 'test': 28}, '/c/cabin/outdoor': {'train': 254, 'test': 64}, '/c/cafeteria': {'train': 83, 'test': 21}, '/c/campsite': {'train': 380, 'test': 96}, '/c/campus': {'train': 139, 'test': 35}, '/c/canal/natural': {'train': 154, 'test': 39}, '/c/canal/urban': {'train': 392, 'test': 98}, '/c/candy_store': {'train': 207, 'test': 52}, '/c/canyon': {'train': 204, 'test': 51}, '/c/car_interior/backseat': {'train': 177, 'test': 45}, '/c/car_interior/frontseat': {'train': 100, 'test': 25}, '/c/carrousel': {'train': 313, 'test': 79}, '/c/casino/indoor': {'train': 472, 'test': 118}, '/c/castle': {'train': 880, 'test': 220}, '/c/catacomb': {'train': 108, 'test': 27}, '/c/cathedral/indoor': {'train': 190, 'test': 48}, '/c/cathedral/outdoor': {'train': 508, 'test': 127}, '/c/cavern/indoor': {'train': 344, 'test': 86}, '/c/cemetery': {'train': 455, 'test': 114}, '/c/chalet': {'train': 251, 'test': 63}, '/c/cheese_factory': {'train': 80, 'test': 21}, '/c/chemistry_lab': {'train': 139, 'test': 35}, '/c/chicken_coop/indoor': {'train': 81, 'test': 21}, '/c/chicken_coop/outdoor': {'train': 229, 'test': 58}, '/c/childs_room': {'train': 127, 'test': 32}, '/c/church/indoor': {'train': 188, 'test': 48}, '/c/church/outdoor': {'train': 736, 'test': 185}, '/c/classroom': {'train': 180, 'test': 45}, '/c/clean_room': {'train': 104, 'test': 26}, '/c/cliff': {'train': 187, 'test': 47}, '/c/cloister/indoor': {'train': 159, 'test': 40}, '/c/closet': {'train': 240, 'test': 60}, '/c/clothing_store': {'train': 268, 'test': 68}, '/c/coast': {'train': 366, 'test': 92}, '/c/cockpit': {'train': 556, 'test': 139}, '/c/coffee_shop': {'train': 128, 'test': 33}, '/c/computer_room': {'train': 148, 'test': 38}, '/c/conference_center': {'train': 84, 'test': 21}, '/c/conference_room': {'train': 666, 'test': 167}, '/c/construction_site': {'train': 421, 'test': 106}, '/c/control_room': {'train': 80, 'test': 20}, '/c/control_tower/outdoor': {'train': 356, 'test': 89}, '/c/corn_field': {'train': 271, 'test': 68}, '/c/corral': {'train': 107, 'test': 27}, '/c/corridor': {'train': 265, 'test': 67}, '/c/cottage_garden': {'train': 183, 'test': 46}, '/c/courthouse': {'train': 300, 'test': 76}, '/c/courtroom': {'train': 117, 'test': 30}, '/c/courtyard': {'train': 80, 'test': 20}, '/c/covered_bridge/exterior': {'train': 452, 'test': 113}, '/c/creek': {'train': 437, 'test': 110}, '/c/crevasse': {'train': 137, 'test': 35}, '/c/crosswalk': {'train': 148, 'test': 38}, '/c/cubicle/office': {'train': 113, 'test': 29}, '/d/dam': {'train': 123, 'test': 31}, '/d/delicatessen': {'train': 193, 'test': 49}, '/d/dentists_office': {'train': 132, 'test': 34}, '/d/desert/sand': {'train': 250, 'test': 63}, '/d/desert/vegetation': {'train': 156, 'test': 39}, '/d/diner/indoor': {'train': 131, 'test': 33}, '/d/diner/outdoor': {'train': 94, 'test': 24}, '/d/dinette/home': {'train': 93, 'test': 24}, '/d/dinette/vehicle': {'train': 157, 'test': 40}, '/d/dining_car': {'train': 194, 'test': 49}, '/d/dining_room': {'train': 944, 'test': 236}, '/d/discotheque': {'train': 109, 'test': 28}, '/d/dock': {'train': 104, 'test': 26}, '/d/doorway/outdoor': {'train': 440, 'test': 110}, '/d/dorm_room': {'train': 132, 'test': 33}, '/d/driveway': {'train': 224, 'test': 57}, '/d/driving_range/outdoor': {'train': 100, 'test': 25}, '/d/drugstore': {'train': 100, 'test': 26}, '/e/electrical_substation': {'train': 80, 'test': 21}, '/e/elevator/door': {'train': 111, 'test': 28}, '/e/elevator/interior': {'train': 101, 'test': 26}, '/e/elevator_shaft': {'train': 96, 'test': 25}, '/e/engine_room': {'train': 100, 'test': 25}, '/e/escalator/indoor': {'train': 315, 'test': 79}, '/e/excavation': {'train': 267, 'test': 67}, '/f/factory/indoor': {'train': 80, 'test': 21}, '/f/fairway': {'train': 207, 'test': 52}, '/f/fastfood_restaurant': {'train': 135, 'test': 34}, '/f/field/cultivated': {'train': 274, 'test': 69}, '/f/field/wild': {'train': 198, 'test': 50}, '/f/fire_escape': {'train': 235, 'test': 59}, '/f/fire_station': {'train': 226, 'test': 57}, '/f/firing_range/indoor': {'train': 88, 'test': 23}, '/f/fishpond': {'train': 133, 'test': 34}, '/f/florist_shop/indoor': {'train': 123, 'test': 31}, '/f/food_court': {'train': 82, 'test': 21}, '/f/forest/broadleaf': {'train': 212, 'test': 53}, '/f/forest/needleleaf': {'train': 129, 'test': 33}, '/f/forest_path': {'train': 401, 'test': 101}, '/f/forest_road': {'train': 121, 'test': 31}, '/f/formal_garden': {'train': 200, 'test': 51}, '/f/fountain': {'train': 169, 'test': 43}, '/g/galley': {'train': 212, 'test': 53}, '/g/game_room': {'train': 176, 'test': 45}, '/g/garage/indoor': {'train': 136, 'test': 35}, '/g/garbage_dump': {'train': 80, 'test': 20}, '/g/gas_station': {'train': 262, 'test': 66}, '/g/gazebo/exterior': {'train': 642, 'test': 161}, '/g/general_store/indoor': {'train': 80, 'test': 21}, '/g/general_store/outdoor': {'train': 128, 'test': 33}, '/g/gift_shop': {'train': 112, 'test': 28}, '/g/golf_course': {'train': 651, 'test': 163}, '/g/greenhouse/indoor': {'train': 285, 'test': 72}, '/g/greenhouse/outdoor': {'train': 108, 'test': 28}, '/g/gymnasium/indoor': {'train': 297, 'test': 75}, '/h/hangar/indoor': {'train': 164, 'test': 41}, '/h/hangar/outdoor': {'train': 104, 'test': 27}, '/h/harbor': {'train': 200, 'test': 50}, '/h/hayfield': {'train': 115, 'test': 29}, '/h/heliport': {'train': 80, 'test': 21}, '/h/herb_garden': {'train': 176, 'test': 45}, '/h/highway': {'train': 666, 'test': 167}, '/h/hill': {'train': 115, 'test': 29}, '/h/home_office': {'train': 142, 'test': 36}, '/h/hospital': {'train': 96, 'test': 24}, '/h/hospital_room': {'train': 162, 'test': 41}, '/h/hot_spring': {'train': 92, 'test': 24}, '/h/hot_tub/outdoor': {'train': 192, 'test': 49}, '/h/hotel/outdoor': {'train': 96, 'test': 24}, '/h/hotel_room': {'train': 393, 'test': 99}, '/h/house': {'train': 764, 'test': 191}, '/h/hunting_lodge/outdoor': {'train': 82, 'test': 21}, '/i/ice_cream_parlor': {'train': 90, 'test': 23}, '/i/ice_floe': {'train': 165, 'test': 42}, '/i/ice_shelf': {'train': 106, 'test': 27}, '/i/ice_skating_rink/indoor': {'train': 202, 'test': 51}, '/i/ice_skating_rink/outdoor': {'train': 175, 'test': 44}, '/i/iceberg': {'train': 319, 'test': 80}, '/i/igloo': {'train': 132, 'test': 34}, '/i/industrial_area': {'train': 107, 'test': 27}, '/i/inn/outdoor': {'train': 96, 'test': 24}, '/i/islet': {'train': 264, 'test': 66}, '/j/jacuzzi/indoor': {'train': 88, 'test': 23}, '/j/jail/indoor': {'train': 154, 'test': 39}, '/j/jail_cell': {'train': 148, 'test': 37}, '/j/jewelry_shop': {'train': 139, 'test': 35}, '/k/kasbah': {'train': 143, 'test': 36}, '/k/kennel/indoor': {'train': 126, 'test': 32}, '/k/kennel/outdoor': {'train': 165, 'test': 42}, '/k/kindergarden_classroom': {'train': 115, 'test': 29}, '/k/kitchen': {'train': 1396, 'test': 350}, '/k/kitchenette': {'train': 106, 'test': 27}, '/l/labyrinth/outdoor': {'train': 112, 'test': 28}, '/l/lake/natural': {'train': 344, 'test': 86}, '/l/landfill': {'train': 175, 'test': 44}, '/l/landing_deck': {'train': 246, 'test': 62}, '/l/laundromat': {'train': 256, 'test': 64}, '/l/lecture_room': {'train': 153, 'test': 39}, '/l/library/indoor': {'train': 286, 'test': 72}, '/l/library/outdoor': {'train': 81, 'test': 21}, '/l/lido_deck/outdoor': {'train': 110, 'test': 28}, '/l/lift_bridge': {'train': 102, 'test': 26}, '/l/lighthouse': {'train': 391, 'test': 98}, '/l/limousine_interior': {'train': 86, 'test': 22}, '/l/living_room': {'train': 1888, 'test': 473}, '/l/lobby': {'train': 371, 'test': 93}, '/l/lock_chamber': {'train': 80, 'test': 21}, '/l/locker_room': {'train': 292, 'test': 73}, '/m/mansion': {'train': 363, 'test': 91}, '/m/manufactured_home': {'train': 212, 'test': 54}, '/m/market/indoor': {'train': 80, 'test': 21}, '/m/market/outdoor': {'train': 671, 'test': 168}, '/m/marsh': {'train': 154, 'test': 39}, '/m/martial_arts_gym': {'train': 146, 'test': 37}, '/m/mausoleum': {'train': 160, 'test': 41}, '/m/medina': {'train': 83, 'test': 21}, '/m/moat/water': {'train': 88, 'test': 22}, '/m/monastery/outdoor': {'train': 142, 'test': 36}, '/m/mosque/indoor': {'train': 99, 'test': 25}, '/m/mosque/outdoor': {'train': 428, 'test': 108}, '/m/motel': {'train': 203, 'test': 51}, '/m/mountain': {'train': 414, 'test': 104}, '/m/mountain_snowy': {'train': 588, 'test': 148}, '/m/movie_theater/indoor': {'train': 174, 'test': 44}, '/m/museum/indoor': {'train': 198, 'test': 50}, '/m/music_store': {'train': 88, 'test': 22}, '/m/music_studio': {'train': 295, 'test': 74}, '/n/nuclear_power_plant/outdoor': {'train': 92, 'test': 24}, '/n/nursery': {'train': 200, 'test': 50}, '/o/oast_house': {'train': 87, 'test': 22}, '/o/observatory/outdoor': {'train': 161, 'test': 41}, '/o/ocean': {'train': 185, 'test': 47}, '/o/office': {'train': 108, 'test': 28}, '/o/office_building': {'train': 240, 'test': 60}, '/o/oil_refinery/outdoor': {'train': 127, 'test': 32}, '/o/oilrig': {'train': 168, 'test': 42}, '/o/operating_room': {'train': 165, 'test': 42}, '/o/orchard': {'train': 294, 'test': 74}, '/o/outhouse/outdoor': {'train': 248, 'test': 62}, '/p/pagoda': {'train': 160, 'test': 40}, '/p/palace': {'train': 178, 'test': 45}, '/p/pantry': {'train': 452, 'test': 113}, '/p/park': {'train': 114, 'test': 29}, '/p/parking_garage/indoor': {'train': 80, 'test': 20}, '/p/parking_garage/outdoor': {'train': 88, 'test': 23}, '/p/parking_lot': {'train': 351, 'test': 88}, '/p/parlor': {'train': 263, 'test': 66}, '/p/pasture': {'train': 489, 'test': 123}, '/p/patio': {'train': 93, 'test': 24}, '/p/pavilion': {'train': 116, 'test': 29}, '/p/pharmacy': {'train': 127, 'test': 32}, '/p/phone_booth': {'train': 252, 'test': 64}, '/p/physics_laboratory': {'train': 130, 'test': 33}, '/p/picnic_area': {'train': 118, 'test': 30}, '/p/pilothouse/indoor': {'train': 177, 'test': 45}, '/p/planetarium/outdoor': {'train': 92, 'test': 23}, '/p/playground': {'train': 718, 'test': 180}, '/p/playroom': {'train': 195, 'test': 49}, '/p/plaza': {'train': 182, 'test': 46}, '/p/podium/indoor': {'train': 96, 'test': 24}, '/p/podium/outdoor': {'train': 114, 'test': 29}, '/p/pond': {'train': 143, 'test': 36}, '/p/poolroom/establishment': {'train': 100, 'test': 25}, '/p/poolroom/home': {'train': 317, 'test': 80}, '/p/power_plant/outdoor': {'train': 113, 'test': 29}, '/p/promenade_deck': {'train': 81, 'test': 21}, '/p/pub/indoor': {'train': 112, 'test': 28}, '/p/pulpit': {'train': 113, 'test': 29}, '/p/putting_green': {'train': 100, 'test': 26}, '/r/racecourse': {'train': 92, 'test': 24}, '/r/raceway': {'train': 116, 'test': 29}, '/r/raft': {'train': 175, 'test': 44}, '/r/railroad_track': {'train': 102, 'test': 26}, '/r/rainforest': {'train': 184, 'test': 47}, '/r/reception': {'train': 150, 'test': 38}, '/r/recreation_room': {'train': 94, 'test': 24}, '/r/residential_neighborhood': {'train': 88, 'test': 22}, '/r/restaurant': {'train': 636, 'test': 160}, '/r/restaurant_kitchen': {'train': 107, 'test': 27}, '/r/restaurant_patio': {'train': 354, 'test': 89}, '/r/rice_paddy': {'train': 80, 'test': 20}, '/r/riding_arena': {'train': 84, 'test': 22}, '/r/river': {'train': 182, 'test': 46}, '/r/rock_arch': {'train': 87, 'test': 22}, '/r/rope_bridge': {'train': 89, 'test': 23}, '/r/ruin': {'train': 261, 'test': 66}, '/r/runway': {'train': 144, 'test': 36}, '/s/sandbar': {'train': 108, 'test': 27}, '/s/sandbox': {'train': 162, 'test': 41}, '/s/sauna': {'train': 141, 'test': 36}, '/s/schoolhouse': {'train': 111, 'test': 28}, '/s/sea_cliff': {'train': 100, 'test': 26}, '/s/server_room': {'train': 92, 'test': 24}, '/s/shed': {'train': 230, 'test': 58}, '/s/shoe_shop': {'train': 144, 'test': 37}, '/s/shopfront': {'train': 580, 'test': 145}, '/s/shopping_mall/indoor': {'train': 227, 'test': 57}, '/s/shower': {'train': 100, 'test': 25}, '/s/skatepark': {'train': 80, 'test': 20}, '/s/ski_lodge': {'train': 89, 'test': 23}, '/s/ski_resort': {'train': 150, 'test': 38}, '/s/ski_slope': {'train': 208, 'test': 52}, '/s/sky': {'train': 133, 'test': 34}, '/s/skyscraper': {'train': 640, 'test': 161}, '/s/slum': {'train': 153, 'test': 39}, '/s/snowfield': {'train': 151, 'test': 38}, '/s/squash_court': {'train': 92, 'test': 24}, '/s/stable': {'train': 128, 'test': 33}, '/s/stadium/baseball': {'train': 224, 'test': 56}, '/s/stadium/football': {'train': 93, 'test': 24}, '/s/stage/indoor': {'train': 98, 'test': 25}, '/s/staircase': {'train': 544, 'test': 136}, '/s/street': {'train': 366, 'test': 92}, '/s/subway_interior': {'train': 364, 'test': 91}, '/s/subway_station/platform': {'train': 414, 'test': 104}, '/s/supermarket': {'train': 298, 'test': 75}, '/s/sushi_bar': {'train': 95, 'test': 24}, '/s/swamp': {'train': 114, 'test': 29}, '/s/swimming_pool/indoor': {'train': 204, 'test': 52}, '/s/swimming_pool/outdoor': {'train': 342, 'test': 86}, '/s/synagogue/indoor': {'train': 103, 'test': 26}, '/s/synagogue/outdoor': {'train': 117, 'test': 30}, '/t/television_studio': {'train': 157, 'test': 40}, '/t/temple/east_asia': {'train': 161, 'test': 41}, '/t/temple/south_asia': {'train': 124, 'test': 31}, '/t/tennis_court/indoor': {'train': 97, 'test': 25}, '/t/tennis_court/outdoor': {'train': 343, 'test': 86}, '/t/tent/outdoor': {'train': 178, 'test': 45}, '/t/theater/indoor_procenium': {'train': 189, 'test': 48}, '/t/theater/indoor_seats': {'train': 87, 'test': 22}, '/t/thriftshop': {'train': 80, 'test': 20}, '/t/throne_room': {'train': 91, 'test': 23}, '/t/ticket_booth': {'train': 104, 'test': 27}, '/t/toll_plaza': {'train': 82, 'test': 21}, '/t/topiary_garden': {'train': 89, 'test': 23}, '/t/tower': {'train': 563, 'test': 141}, '/t/toyshop': {'train': 310, 'test': 78}, '/t/track/outdoor': {'train': 108, 'test': 28}, '/t/train_railway': {'train': 220, 'test': 56}, '/t/train_station/platform': {'train': 95, 'test': 24}, '/t/tree_farm': {'train': 94, 'test': 24}, '/t/tree_house': {'train': 90, 'test': 23}, '/t/trench': {'train': 84, 'test': 21}, '/u/underwater/coral_reef': {'train': 395, 'test': 99}, '/u/utility_room': {'train': 115, 'test': 29}, '/v/valley': {'train': 116, 'test': 29}, '/v/van_interior': {'train': 104, 'test': 27}, '/v/vegetable_garden': {'train': 396, 'test': 100}, '/v/veranda': {'train': 80, 'test': 20}, '/v/veterinarians_office': {'train': 112, 'test': 28}, '/v/viaduct': {'train': 203, 'test': 51}, '/v/videostore': {'train': 100, 'test': 25}, '/v/village': {'train': 110, 'test': 28}, '/v/vineyard': {'train': 361, 'test': 91}, '/v/volcano': {'train': 100, 'test': 26}, '/v/volleyball_court/indoor': {'train': 85, 'test': 22}, '/v/volleyball_court/outdoor': {'train': 109, 'test': 28}, '/w/waiting_room': {'train': 380, 'test': 96}, '/w/warehouse/indoor': {'train': 621, 'test': 156}, '/w/water_tower': {'train': 316, 'test': 80}, '/w/waterfall/block': {'train': 168, 'test': 42}, '/w/waterfall/fan': {'train': 162, 'test': 41}, '/w/waterfall/plunge': {'train': 84, 'test': 22}, '/w/watering_hole': {'train': 85, 'test': 22}, '/w/wave': {'train': 219, 'test': 55}, '/w/wet_bar': {'train': 108, 'test': 27}, '/w/wheat_field': {'train': 197, 'test': 50}, '/w/wind_farm': {'train': 241, 'test': 61}, '/w/windmill': {'train': 330, 'test': 83}, '/w/wine_cellar/barrel_storage': {'train': 196, 'test': 50}, '/w/wine_cellar/bottle_storage': {'train': 272, 'test': 68}, '/w/wrestling_ring/indoor': {'train': 80, 'test': 20}, '/y/yard': {'train': 160, 'test': 41}, '/y/youth_hostel': {'train': 110, 'test': 28}}
"""
class SUN397(Dataset):
    """
        Scenes UNderstanding of 397 Scenes (SUN397) Dataset

        Args: 
            root (string): Root directory where images are downloaded to or better to the extracted sun397 folder.
                            folder data structure should be:
                            data (root)
                                |-sun397
                                |   |-partitions (annotations)
                                |   |-SUN397 (images)
            split (string or list): possible options: 'train', 'test', if list of multiple splits they will be treated as unique split
            split_perc (float): since there is no file of dividing train and test images (there are partitions but they are for other purpose), 
                you choose percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
            partition (float): use it to take only a part of the dataset, keeping the proportion of number of 
                images per classes split_perc will work as well splitting the partion
            aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
            dim_threshold (float): use it to filter images which area is 
                lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
            
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, 
                transform: Optional[Callable] = None, partition_perc: float = 1.0, distribute_images: bool = False, distribute_level: str = 'level3',
                aspect_ratio_threshold: float = None, dim_threshold: int = None):

        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]

        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.partition_perc = partition_perc
        self.distribute_images = distribute_images

        self.distribute_level = distribute_level

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None

        self.metadata, self.targets, self.classes_map, self.classes_count, self.classes_hierarchy, self.filtering_classes_effect, self.total_filtered = self._read_metadata()
        self.classes = list(self.classes_map.keys())


    """
        Read all metadata related to dataset:
        - classes
        - hierarchical classes
    """
    def _read_metadata(self):
        metadata = []
        targets_all = []

        total_images = 0
        # create map of classes { classname: index }
        classes_map = {'level1': {"indoor": 0, "outdoor_natural": 1, "outdoor_man-made": 2},
                        'level2': {"shopping_and_dining": 0, "workplace": 1, "home_or_hotel": 2,
                                    "transportation": 3, "sports_and_leisure": 4, "cultural": 5,
                                    "water_ice_snow": 6, "mountains_hills_desert_sky": 7,
                                    "forest_field_jungle": 8, "man-made_elements": 9,
                                    "transportation": 10, "cultural_or_historical_building_place": 11,
                                    "sportsfields_parks_leisure_spaces": 12, "industrial_and_construction": 13,
                                    "houses_cabins_gardens_and_farms": 14, "commercial_buildings": 15},
                        'level3': {}}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {'level1': {"indoor": 0, "outdoor_natural": 0, "outdoor_man-made": 0},
                        'level2': {"shopping_and_dining": 0, "workplace": 0, "home_or_hotel": 0,
                                    "transportation": 0, "sports_and_leisure": 0, "cultural": 0,
                                    "water_ice_snow": 0, "mountains_hills_desert_sky": 0,
                                    "forest_field_jungle": 0, "man-made_elements": 0,
                                    "transportation": 0, "cultural_or_historical_building_place": 0,
                                    "sportsfields_parks_leisure_spaces": 0, "industrial_and_construction": 0,
                                    "houses_cabins_gardens_and_farms": 0, "commercial_buildings": 0},
                        'level3': {}}
        # will be used to distribute in the same proportion each class into train and test
        classes_splitter = {'level1': {"indoor": 0, "outdoor_natural": 0, "outdoor_man-made": 0},
                            'level2': {"shopping_and_dining": 0, "workplace": 0, "home_or_hotel": 0,
                                        "transportation": 0, "sports_and_leisure": 0, "cultural": 0,
                                        "water_ice_snow": 0, "mountains_hills_desert_sky": 0,
                                        "forest_field_jungle": 0, "man-made_elements": 0,
                                        "transportation": 0, "cultural_or_historical_building_place": 0,
                                        "sportsfields_parks_leisure_spaces": 0, "industrial_and_construction": 0,
                                        "houses_cabins_gardens_and_farms": 0, "commercial_buildings": 0},
                            'level3': {}}

        # structures for tracking filtered images due to thresholds
        filtered_classes_count = {'level1': {"indoor": 0, "outdoor_natural": 0, "outdoor_man-made": 0},
                                'level2': {"shopping_and_dining": 0, "workplace": 0, "home_or_hotel": 0,
                                            "transportation": 0, "sports_and_leisure": 0, "cultural": 0,
                                            "water_ice_snow": 0, "mountains_hills_desert_sky": 0,
                                            "forest_field_jungle": 0, "man-made_elements": 0,
                                            "transportation": 0, "cultural_or_historical_building_place": 0,
                                            "sportsfields_parks_leisure_spaces": 0, "industrial_and_construction": 0,
                                            "houses_cabins_gardens_and_farms": 0, "commercial_buildings": 0},
                                'level3': {}}
        total_filtered = 0

        # will be used to distribute equally images among classes
        distributed_classes_count = {'level1': {"indoor": 0, "outdoor_natural": 0, "outdoor_man-made": 0},
                                'level2': {"shopping_and_dining": 0, "workplace": 0, "home_or_hotel": 0,
                                            "transportation": 0, "sports_and_leisure": 0, "cultural": 0,
                                            "water_ice_snow": 0, "mountains_hills_desert_sky": 0,
                                            "forest_field_jungle": 0, "man-made_elements": 0,
                                            "transportation": 0, "cultural_or_historical_building_place": 0,
                                            "sportsfields_parks_leisure_spaces": 0, "industrial_and_construction": 0,
                                            "houses_cabins_gardens_and_farms": 0, "commercial_buildings": 0},
                                'level3': {}}

        # read level3 classes name and map them
        with open(os.path.join(self.root,"sun397/partitions/ClassName.txt"), 'r') as file:
            class_index = 0
            for line in file:
                line = line.strip('\n')
                classes_map['level3'][line] = class_index
                classes_count['level3'][line] = 0
                classes_splitter['level3'][line] = 0
                filtered_classes_count['level3'][line] = 0
                distributed_classes_count['level3'][line] = 0
                class_index += 1
        
        # creating level 1 and level 2 classes hierarchy
        classes_hierarchy = {"indoor": {"shopping_and_dining": [],
                                        "workplace": [],
                                        "home_or_hotel": [],
                                        "transportation": [],
                                        "sports_and_leisure": [],
                                        "cultural": []},
                            "outdoor_natural": {"water_ice_snow": [],
                                                "mountains_hills_desert_sky": [],
                                                "forest_field_jungle": [],
                                                "man-made_elements": []},
                            "outdoor_man-made": {"transportation": [],
                                                "cultural_or_historical_building_place": [],
                                                "sportsfields_parks_leisure_spaces": [],
                                                "industrial_and_construction": [],
                                                "houses_cabins_gardens_and_farms": [],
                                                "commercial_buildings":[]}}
        classes_hierarchy_reverse = []
        level1_classes = list(classes_hierarchy.keys())
        level2_classes = list()
        level2_classes.append(list(classes_hierarchy['indoor'].keys()))
        level2_classes.append(list(classes_hierarchy['outdoor_natural'].keys()))
        level2_classes.append(list(classes_hierarchy['outdoor_man-made'].keys()))

        # read hierarchy csv in order to create association between level2 and level3 classes and so to level1 ones
        with open(os.path.join(self.root, 'sun397/partitions/hierarchy_classes.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # ignore columns
                    line_count += 1
                else:
                    level1_index = row.index("1", 1, 4) - 1
                    level2_index = 0
                    if level1_index == 0:
                        level2_index = row.index("1", 4, 10) - 4
                    elif level1_index == 1:
                        level2_index = row.index("1", 10, 14) - 10
                    elif level1_index == 2:
                        level2_index = row.index("1", 14, 20) - 14
                    classes_hierarchy[level1_classes[level1_index]][level2_classes[level1_index][level2_index]].append(row[0])
                    classes_hierarchy_reverse.append(list([row[0], level2_classes[level1_index][level2_index], level1_classes[level1_index]]))
                    line_count += 1
            # print('Processed {} lines.'.format(line_count))
            # print(classes_hierarchy)
        
        # calculating classes statistics
        for l3_class_name in classes_map['level3'].keys():
            
            for img in os.listdir(os.path.join(self.root, 'sun397/SUN397', l3_class_name[1:])):
                total_images += 1
                classes_count['level3'][l3_class_name] += 1
                classes_count['level2'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][1]] += 1
                classes_count['level1'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][2]] += 1
                filtered_classes_count['level3'][l3_class_name] += 1
                filtered_classes_count['level2'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][1]] += 1
                filtered_classes_count['level1'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][2]] += 1

                W, H = Image.open(os.path.join(self.root, 'sun397/SUN397', l3_class_name[1:], img)).size

                if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                    # filtered_classes_count[class_name] -= 1
                    filtered_classes_count['level3'][l3_class_name] -= 1
                    filtered_classes_count['level2'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][1]] -= 1
                    filtered_classes_count['level1'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][2]] -= 1
                    total_filtered += 1
                    total_images -= 1
                elif self.area_threshold is not None and (W*H < self.area_threshold):
                    # filtered_classes_count[class_name] -= 1
                    filtered_classes_count['level3'][l3_class_name] -= 1
                    filtered_classes_count['level2'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][1]] -= 1
                    filtered_classes_count['level1'][classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][2]] -= 1
                    total_filtered += 1
                    total_images -= 1

        total_images = int(total_images * self.partition_perc)

        # try to distributed images equally among classes
        if self.distribute_images == True:
            i = 0
            while i < total_images:
                for c in distributed_classes_count[self.distribute_level].keys():
                    if distributed_classes_count[self.distribute_level][c] < int(filtered_classes_count[self.distribute_level][c]):
                        distributed_classes_count[self.distribute_level][c] += 1
                        i += 1
            filtered_classes_count = distributed_classes_count
        else:
            for c in filtered_classes_count[self.distribute_level].keys():
                filtered_classes_count[self.distribute_level][c] = filtered_classes_count[self.distribute_level][c] * self.partition_perc
        
        for l3_class_name in classes_map['level3'].keys():
            for img in os.listdir(os.path.join(self.root, 'sun397/SUN397', l3_class_name[1:])):
                
                l2_class_name = classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][1]
                l1_class_name = classes_hierarchy_reverse[classes_map['level3'][l3_class_name]][2]
                if self.distribute_level == 'level1':
                    current_class_name = l1_class_name
                elif self.distribute_level == 'level2':
                    current_class_name = l2_class_name
                elif self.distribute_level == 'level3':
                    current_class_name = l3_class_name

                W, H = Image.open(os.path.join(self.root, 'sun397/SUN397', l3_class_name[1:], img)).size
                skip = False
                if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                    skip = True
                elif self.area_threshold is not None and (W*H < self.area_threshold):
                    skip = True
                if skip == False:
                    meta = {}
                    if "train" in self.split:
                        if classes_splitter[self.distribute_level][current_class_name] < int(filtered_classes_count[self.distribute_level][current_class_name] * self.split_perc):
                            meta['split'] = "train"
                            meta['img_name'] = img
                            meta['img_folder'] = os.path.join('sun397/SUN397', l3_class_name[1:])
                            meta['target'] = {'level1': l1_class_name, 'level2': l2_class_name, 'level3': l3_class_name}
                            targets_all.append(meta['target'])
                            metadata.append(meta)
                            classes_splitter[self.distribute_level][current_class_name] += 1
                        elif (classes_splitter[self.distribute_level][current_class_name] >= int(filtered_classes_count[self.distribute_level][current_class_name] * self.split_perc) 
                                and classes_splitter[self.distribute_level][current_class_name] < int(filtered_classes_count[self.distribute_level][current_class_name])):
                            if "test" in self.split:
                                meta['split'] = "test"
                                meta['img_name'] = img
                                meta['img_folder'] = os.path.join('sun397/SUN397', l3_class_name[1:])
                                # put relative class index in targets since img_folder is equal to class name
                                meta['target'] = {'level1': l1_class_name, 'level2': l2_class_name, 'level3': l3_class_name}
                                targets_all.append(meta['target'])
                                metadata.append(meta)
                                classes_splitter[self.distribute_level][current_class_name] += 1
        
        # check how much filtering changed classes proportion
        filtering_classes_effect = {}
        filtering_classes_effect_sorted = collections.OrderedDict()
        for key in classes_count[self.distribute_level].keys():
            if round(filtered_classes_count[self.distribute_level][key]/classes_count[self.distribute_level][key], 2) != 1.0:
                filtering_classes_effect[key] = round(filtered_classes_count[self.distribute_level][key]/classes_count[self.distribute_level][key], 2)

        for key, value in sorted(filtering_classes_effect.items(), key=lambda item: item[1]):
            filtering_classes_effect_sorted[key] = value
        # print(filtered_classes_count)
        # print(filtering_classes_effect_sorted)
        # print(total_filtered)

        
        return (metadata, targets_all, classes_map, classes_splitter, classes_hierarchy, 
            filtering_classes_effect_sorted, total_filtered)
                

    def __len__(self):
        return len(self.metadata)
                
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target






    