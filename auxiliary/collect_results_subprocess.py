# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import subprocess

phases = ["early", "late"]
locations = ["bonn", "heppenheim", "sinsheim", "mosbach"]
script_file = "/media/E132-Projekte/Projects/2025_MartinezMora_TimeAwareVODetection/ctp_dynamics_extraction/germany_exps/collect_results.py"
folder = "/media/E132-Projekte/Projects/2025_MartinezMora_TimeAwareVODetection/models_stroke_amsterdamproject"

for phase in phases:
    for location in locations:
        subprocess.run(
            [
                sys.executable,
                script_file,
                "--f",
                os.path.join(
                    folder, f"folders_external_results_{location}_{phase}.json"
                ),
                "--o",
                os.path.join(
                    folder, f"external_results_{location}_{phase}_with_retraining.json"
                ),
            ]
        )
