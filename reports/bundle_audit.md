| ep | split | object | state | run.json | overlay | hand scale L/R | mesh/scan | drift 30f cm | step cm | still from | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | evaluation | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 1 | evaluation | white_pot_lid | REDO | no | no | 0.95/0.82 | 1.45 | 3.2 | 0.41 | 0 | right hand scale 0.82: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); mesh is 1.448x the scanned object's size |
| 3 | evaluation | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 3 | evaluation | wooden_spoon | OK | no | no | 1.07/1.19 |  | 2.1 | 0.63 | 12 |  |
| 5 | evaluation | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 5 | evaluation | wooden_spoon | OK | no | no | 1.05/1.04 |  | 2.0 | 0.47 | never | object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 6 | evaluation | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 6 | evaluation | wooden_spoon | SHOULD | no | no | 1.06/1.14 |  | 7.4 | 0.42 | never | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 8 | evaluation | beige_cup | OK | no | no | 1.25/1.26 | 0.96 | 0.9 | 0.47 | 0 |  |
| 8 | evaluation | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 10 | evaluation | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 10 | evaluation | beige_cup | OK | no | no | 1.13/1.01 | 1.18 | 2.7 | 0.21 | 12 |  |
| 14 | evaluation | blue_cup | SHOULD | no | no | nan/0.91 | 0.97 | 11.1 | 0.37 | 149 | hand_left valid on only 0% of frames; left hand scale nan: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 14 | evaluation | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 15 | evaluation | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 15 | evaluation | wooden_spoon | OK | no | no | 1.04/1.21 |  | 2.6 | 0.47 | 141 | object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 16 | evaluation | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 16 | evaluation | wooden_spoon | REDO | no | no | 1.02/1.70 |  | 1.9 | 0.47 | 0 | hand_left valid on only 77% of frames; right hand scale 1.70: world may be off by that factor; rest footprint 27.9x3.9 cm is a strip: object may sit on an edge (level it) |
| 19 | evaluation | mini_sweeper | SHOULD | no | no | 0.95/1.29 | 1.09 | 7.4 | 0.33 | 189 | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation; rest footprint 14.8x2.7 cm is a strip: object may sit on an edge (level it) |
| 19 | evaluation | mini_dust_pan | MISSING | no | no | / |  |  |  |  |  |
| 20 | evaluation | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 20 | evaluation | water_pitcher | SHOULD | no | no | 1.13/0.93 | 1.07 | 6.8 | 0.39 | 8 | hand_right valid on only 72% of frames; object moves in the first 30 frames (rest contract expects mm-cm) |
| 25 | evaluation | wooden_piece_1 | MISSING | no | no | / |  |  |  |  |  |
| 25 | evaluation | wooden_piece_2 | REDO | no | no | 1.09/1.07 | 11.38 | 5.4 | 0.15 | 121 | hand_left valid on only 92% of frames; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation; rest footprint 37.0x3.6 cm is a strip: object may sit on an edge (level it); mesh is 11.377x the scanned object's size |
| 28 | evaluation | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 28 | evaluation | white_pot | SHOULD | no | no | 1.17/1.13 | 0.85 | 25.1 | 0.61 | 184 | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 30 | evaluation | mini_sweeper | REDO | no | no | 1.42/1.38 | 3.48 | 2.9 | 0.08 | 26 | left hand scale 1.42: world may be off by that factor; right hand scale 1.38: world may be off by that factor; rest footprint 66.0x7.8 cm is a strip: object may sit on an edge (level it); mesh is 3.482x the scanned object's size |
| 30 | evaluation | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 32 | evaluation | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 32 | evaluation | wooden_spoon | SHOULD | no | no | 1.06/1.10 |  | 5.0 | 0.84 | never | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 33 | evaluation | blue_cup | REDO | no | no | 1.03/1.18 | 1.31 | 4.8 | 0.38 | 5 | object moves in the first 30 frames (rest contract expects mm-cm); mesh is 1.311x the scanned object's size |
| 33 | evaluation | wooden_spoon | SHOULD | no | no | 0.92/1.10 |  | 6.3 | 0.94 | 1 | object moves in the first 30 frames (rest contract expects mm-cm) |
| 35 | evaluation | wooden_spoon | SHOULD | no | no | 1.24/1.45 |  | 3.0 | 0.59 | never | right hand scale 1.45: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 36 | evaluation | wooden_spoon | SHOULD | no | no | 1.21/1.15 |  | 8.1 | 0.82 | 391 | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 37 | evaluation | wooden_spoon | SHOULD | no | no | 1.40/1.47 |  | 4.1 | 0.65 | never | left hand scale 1.40: world may be off by that factor; right hand scale 1.47: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 38 | evaluation | mini_dust_pan | REDO | no | no | 1.58/2.62 | 1.24 | 1.7 | 0.63 | 12 | left hand scale 1.58: world may be off by that factor; right hand scale 2.62: world may be off by that factor |
| 0 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 0 | public | white_pot_lid | REDO | no | no | nan/0.89 | 1.41 | 3.7 | 0.28 | 14 | hand_left valid on only 0% of frames; left hand scale nan: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); mesh is 1.411x the scanned object's size |
| 2 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 2 | public | white_pot_lid | REDO | no | no | nan/0.87 | 1.35 | 4.4 | 0.55 | never | hand_left valid on only 0% of frames; left hand scale nan: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation; mesh is 1.35x the scanned object's size |
| 11 | public | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 11 | public | beige_cup | SHOULD | no | no | 1.04/0.97 | 1.20 | 6.5 | 0.43 | never | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 12 | public | blue_cup | SHOULD | no | no | 1.09/1.13 | 1.18 | 4.4 | 0.50 | 149 | object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 12 | public | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 13 | public | blue_cup | SHOULD | no | no | 1.05/nan | 0.90 | 12.2 | 0.42 | never | hand_right valid on only 0% of frames; right hand scale nan: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 13 | public | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 18 | public | mini_sweeper | SHOULD | no | no | 0.98/1.45 | 0.94 | 5.9 | 0.22 | 0 | right hand scale 1.45: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm) |
| 18 | public | mini_dust_pan | MISSING | no | no | / |  |  |  |  |  |
| 21 | public | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 21 | public | water_pitcher | SHOULD | no | no | 1.15/1.08 | 1.15 | 8.2 | 0.29 | 115 | hand_right valid on only 77% of frames; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation |
| 22 | public | blue_cup | MISSING | no | no | / |  |  |  |  |  |
| 22 | public | water_pitcher | SHOULD | no | no | 1.04/0.97 | 1.08 | 3.8 | 0.23 | 27 | object moves in the first 30 frames (rest contract expects mm-cm) |
| 23 | public | wooden_piece_1 | MISSING | no | no | / |  |  |  |  |  |
| 23 | public | wooden_piece_2 | REDO | no | no | 0.80/1.02 | 11.84 | 12.8 | 0.20 | 172 | hand_left valid on only 85% of frames; left hand scale 0.80: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation; rest footprint 22.2x3.0 cm is a strip: object may sit on an edge (level it); mesh is 11.835x the scanned object's size |
| 24 | public | wooden_piece_1 | MISSING | no | no | / |  |  |  |  |  |
| 24 | public | wooden_piece_2 | REDO | no | no | 1.21/nan | 7.08 | 4.7 | 0.39 | 185 | hand_right valid on only 0% of frames; right hand scale nan: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); object never holds still (<1 mm, <0.6 deg per frame for 5 frames) in the first 30 frames: no table is built under it, so it falls in simulation; mesh is 7.075x the scanned object's size |
| 26 | public | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 26 | public | white_pot | OK | no | no | 0.97/1.30 | 0.83 | 2.5 | 0.29 | 22 |  |
| 27 | public | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 27 | public | white_pot | REDO | no | no | 1.25/1.05 | 1.27 | 6.0 | 0.70 | 9 | object moves in the first 30 frames (rest contract expects mm-cm); mesh is 1.27x the scanned object's size |
| 31 | public | mini_sweeper | REDO | no | no | 0.75/1.32 | 4.23 | 9.5 | 0.08 | 25 | left hand scale 0.75: world may be off by that factor; right hand scale 1.32: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm); mesh is 4.231x the scanned object's size |
| 31 | public | plastic_dish_rack | MISSING | no | no | / |  |  |  |  |  |
| 39 | public | mini_dust_pan | MISSING | no | no | / |  |  |  |  |  |
| 40 | public | mini_dust_pan | MISSING | no | no | / |  |  |  |  |  |
| 41 | public | white_pot | SHOULD | no | no | 1.10/0.83 | 1.17 | 3.6 | 0.60 | 0 | right hand scale 0.83: world may be off by that factor; object moves in the first 30 frames (rest contract expects mm-cm) |
| 42 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 43 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 44 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
| 45 | public | white_pot | MISSING | no | no | / |  |  |  |  |  |
