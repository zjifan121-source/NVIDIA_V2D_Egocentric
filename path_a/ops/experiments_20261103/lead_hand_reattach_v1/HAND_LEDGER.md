| 段 | 物体版本 | bob1 用谁 · tracking | Ken own-ref（修复 / 对照） | 接触 对照 | 接触 rawexcl | 接触 kenexcl | 接触 Ken 场景 | 门槛判定 | 建议训练用 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fb11 = v1 | Ken · 0.8026 | 0.77, 0.52 / 0.28 | white_pot/l 0.94；white_pot_lid/r 1.00 | white_pot/l 0.94；white_pot_lid/r 1.00 | — | white_pot/l 0.94；white_pot_lid/r 1.00 | rawexcl +0.00；kenexcl — | Ken 场景（有 policy 证据） |
| 3 | fb11 = v1 | Ken + hand fix · 0.9139 | 0.93, 0.67 / 0.52, 0.19 | wooden_spoon/r 0.98 | wooden_spoon/r 1.00 | white_pot/l 1.00；wooden_spoon/r 1.00 | white_pot/l 1.00；wooden_spoon/r 1.00 | rawexcl +0.00；kenexcl +0.00 | Ken 场景（有 policy 证据）；备选 ours: kenexcl（Ken 修正手 + 排他式贴 + 坐实） |
| 5 | v1 改了 wooden_spoon | Wentao · 0.7627 | — | white_pot/l 1.00；wooden_spoon/r 0.99 | white_pot/l 1.00；wooden_spoon/r 0.93 | — | — | rawexcl -0.06；kenexcl — | ours: control（v1 物体 + 坐实） |
| 6 | fb11 = v1 | Ken · 0.9685 | 0.85, 0.86 / 0.31 | white_pot/l 1.00；wooden_spoon/r 0.97 | white_pot/l 1.00；wooden_spoon/r 1.00 | — | white_pot/l 1.00；wooden_spoon/r 1.00 | rawexcl +0.00；kenexcl — | Ken 场景（有 policy 证据） |
| 8 | fb11 = v1 | Ken · 0.8971 | — | beige_cup/r 0.99；white_pot/l 1.00 | beige_cup/r 1.00；white_pot/l 1.00 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 10 | fb11 = v1 | Ken + hand fix · 0.7201 | 0.71, 0.71, 0.71 / 0.54, 0.60 | — | beige_cup/l 0.33；beige_cup/r 0.34；blue_cup/l 0.99 | — | — | rawexcl —；kenexcl — | Ken 场景（有 policy 证据） |
| 14 | fb11 = v1 | Ken · 0.8453 | 0.87, 0.89 / 0.61 | blue_cup/r 0.60 | blue_cup/l 0.92；blue_cup/r 0.84 | blue_cup/l 0.99；blue_cup/r 0.84 | — | rawexcl +0.00；kenexcl +0.00 | Ken 场景（有 policy 证据）；备选 ours: kenexcl（Ken 修正手 + 排他式贴 + 坐实） |
| 15 | fb11 = v1 | Own stereo (Path A) · 0.9537 | — | wooden_spoon/l 0.39 | wooden_spoon/l 0.83 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 16 | fb11 = v1 | Wentao · 0.9314 | — | 无接触手 | wooden_spoon/l 0.66；wooden_spoon/r 0.42 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 19 | fb11 = v1 | Wentao · 0.8984 | — | mini_dust_pan/l 0.83；mini_sweeper/r 0.99 | mini_dust_pan/l 0.98；mini_sweeper/r 0.99 | — | — | rawexcl -0.01；kenexcl — | ours: rawexcl |
| 20 | fb11 = v1 | Ken + hand fix · 0.9041 | 0.23, 0.73, 0.70 / 0.27, 0.18 | blue_cup/l 1.00；water_pitcher/r 0.59 | blue_cup/l 1.00；water_pitcher/r 1.00 | — | — | rawexcl +0.00；kenexcl — | Ken 场景（有 policy 证据） |
| 25 | fb11 = v1 | Repaired assembly scene · 0.87 | — | wooden_piece_1/l 0.91 | wooden_piece_1/l 0.97 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 28 | fb11 = v1 | Wentao · 0.9058 | — | white_pot/r 0.81 | white_pot/l 0.97；white_pot/r 0.97 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 30 | fb11 = v1 | Wentao · 0.9215 | — | mini_sweeper/r 0.98 | mini_sweeper/r 1.00 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 32 | v1 改了 wooden_spoon | Ken · 0.8947 | 0.84, 0.52 / 0.07 | blue_cup/l 0.99；wooden_spoon/r 0.67 | blue_cup/l 0.99；wooden_spoon/r 0.97 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 33 | v1 改了 blue_cup/wooden_spoon | Ken · 0.901 | — | blue_cup/l 1.00；wooden_spoon/r 0.95 | blue_cup/l 1.00；wooden_spoon/r 1.00 | — | — | rawexcl +0.00；kenexcl — | ours: rawexcl |
| 35 | fb11 = v1 | Ken · 0.9534 | — | control/l 0.66；control/r 0.60 | — | — | — | rawexcl —；kenexcl — | ours: control（fb11 手 + 坐实） |
| 36 | fb11 = v1 | Own stereo (Path A) · 0.9217 | — | control/l 0.30；control/r 0.68 | — | — | — | rawexcl —；kenexcl — | ours: control（fb11 手 + 坐实） |
| 37 | fb11 = v1 | Own stereo (Path A) · 0.9905 | — | control/r 0.36 | — | — | — | rawexcl —；kenexcl — | ours: control（fb11 手 + 坐实） |
| 38 | fb11 = v1 | Wentao · 0.9482 | — | control/r 1.00 | — | — | — | rawexcl —；kenexcl — | ours: control（fb11 手 + 坐实） |
