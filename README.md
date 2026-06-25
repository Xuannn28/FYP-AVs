# Smart V2X Cooperative Perception in Autonomous Vehicles
### Optimizing Cooperative Perception in Autonomous Vehicles through Semantic Data Analytics
 
**Monash University FIT3162/FIT3164 Final Year Project**
 
| | |
|---|---|
| **Team** | Goh Jia Xuan, Ooi Pei Shuen, Ng Li Xian |
| **Supervisor** | Dr. Tan Chee Keong |
| **Final Report** | [FIT3164_Final_Written_Report.pdf](./docs/FIT3164_Final_Written_Report.pdf) |
| **Final Presentation** | [Final Presentation Slides](./docs/FIT3164_Final_Presentation_Slides.pdf) |
 
---
 
## Overview
 
A vehicle cannot detect what it cannot see. Physical occlusion caused by
large vehicles, buildings, and road geometry creates blind spots that no
single onboard sensor can resolve alone. This project addresses that
limitation through **cooperative perception** — an ego vehicle requests and
fuses detection data from a Road Side Unit (RSU) via Vehicle-to-Everything
(V2X) communication, activating assistance only when needed and minimising
bandwidth by transmitting detection results rather than raw images.
 
The system is validated through two independent strands of work:
 
1. **Dataset Analysis** (`CoopPerception/`) — statistical proof that
   cooperative perception improves scene coverage, using OPV2V (simulated
   LiDAR) and DAIR-V2X (real-world camera) benchmark datasets
2. **Hardware Prototype** (`final_demo/`) — a physical two-node Raspberry Pi
   system demonstrating cooperative perception, confidence-triggered RSU
   activation, and transmission optimisation on real camera hardware
---
 
## Repository Structure
---
 
## Key Results
 
### Dataset Validation (R1)
 
| Dataset | Single-Agent | Cooperative | Gain | Statistical Test |
|---|---|---|---|---|
| **OPV2V** (simulated LiDAR, 407 frames, 16 scenarios) | 7.77% ± 1.38% | 15.22% ± 4.33% | **+7.44%** | Paired t-test p = 9.67×10⁻¹⁴⁵, Cohen's d = 2.011 |
| **DAIR-V2X** (real-world camera, 46 pairs) | 55.9% ± 13.5% | 100% (ground truth) | **+44.1%** | Descriptive (see [methodology note](./CoopPerception/README.md#why-dair-v2x-is-descriptive)) |
 
![OPV2V Scene Coverage](./CoopPerception/results/per_scenario_coverage.png)
![DAIR-V2X Scene Coverage](./CoopPerception/results_dairv2x/coverage_bar.png)
