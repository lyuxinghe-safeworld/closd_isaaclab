# CLoSD-IsaacLab

This project ports CLoSD's text-to-motion closed-loop pipeline to Isaac Lab using ProtoMotions' motion tracker. Given a natural-language prompt (e.g. "a person walks forward"), a DiP diffusion model generates a 263-dim HumanML3D motion sequence, which is converted to joint rotations and fed in real time to a ProtoMotions SMPL tracker running inside Isaac Lab — closing the loop between language, diffusion planning, and physics simulation.

---

## Prerequisites

- GCP VM with an NVIDIA GPU (A100 / T4 / V100 recommended)
- TurboVNC installed and configured (required for Isaac Lab headless display)
- Python 3.10+
- The following repos cloned under `~/code/`:

| Repo | Path |
|------|------|
| CLoSD | `~/code/CLoSD` |
| CLoSD_t2m_standalone | `~/code/CLoSD_t2m_standalone` |
| ProtoMotions | `~/code/ProtoMotions` |
| closd_isaaclab (this repo) | `~/code/closd_isaaclab` |

---

## Setup

### 1. Verify repos are present

```bash
ls ~/code/CLoSD ~/code/CLoSD_t2m_standalone ~/code/ProtoMotions ~/code/closd_isaaclab
```

### 2. Download the ProtoMotions motion tracker checkpoint

The tracker checkpoint is stored via Git LFS. Pull it with:

```bash
cd ~/code/ProtoMotions
git lfs pull
```

If Git LFS is unavailable, download directly with curl:

```bash
mkdir -p ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl
curl -L -o ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    <CHECKPOINT_URL>
```

Verify the file exists:

```bash
ls -lh ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt
```

### 3. Download CLoSD DiP model weights

The DiP checkpoint is auto-downloaded from HuggingFace on first run. To trigger the download in advance:

```bash
cd ~/code/CLoSD
python -c "from closd.diffusion_planner.model_util import load_model_wo_clip; print('OK')"
```

The checkpoint will be saved to:

```
~/code/CLoSD/closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt
```

### 4. Install this package

Activate the `env_isaaclab` environment and install in editable mode:

```bash
source ~/code/env_isaaclab/bin/activate
pip install -e "~/code/closd_isaaclab[dev]"
```

### 5. Set environment variables

```bash
export PYTHONPATH="$HOME/code/CLoSD:$HOME/code/CLoSD_t2m_standalone:$HOME/code/ProtoMotions:$PYTHONPATH"
export DISPLAY=:1   # adjust to match your active VNC display
```

---

## TurboVNC Notes

Isaac Lab requires a display. On a headless GCP VM, use TurboVNC:

```bash
# Start the VNC server on display :1
vncserver :1 -geometry 1920x1080 -depth 24

# Export the display in your shell
export DISPLAY=:1

# (Optional) Use VirtualGL for GPU-accelerated rendering
vglrun python scripts/verify_tracking.py
```

To stop the server:

```bash
vncserver -kill :1
```

Connect from your local machine with a TurboVNC client pointed at `<VM_IP>:5901`.

---

## Verification Steps

Run these in order to validate each layer of the stack independently.

### Step 1 — Diffusion only (no simulator)

Generates a skeleton animation MP4 from a text prompt. No Isaac Lab or display needed.

```bash
python scripts/verify_diffusion.py --prompt "a person walks forward" --num-seconds 8
```

Outputs: `motion.pt`, `xyz.pt`, and `a-person-walks-forward.mp4` in the current directory.

### Step 2 — Rotation solver

Tests 6D rotation round-trips and forward-kinematics consistency against a ProtoMotions `.motion` file.

```bash
python scripts/verify_rotations.py \
    --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion
```

All checks should print `PASS`.

### Step 3 — Motion tracking (needs DISPLAY)

Loads the pretrained SMPL tracker and runs it on an offline `.motion` file inside Isaac Lab.

```bash
# Required env vars for GCP VMs
export DISPLAY=:1
export LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python scripts/verify_tracking.py \
    --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --motion-file ~/code/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion \
    --num-envs 1
```

### Step 4 — Full closed-loop pipeline (needs DISPLAY)

Runs the complete text → diffusion → joint rotations → Isaac Lab tracker pipeline.

```bash
# Required env vars for GCP VMs (same as Step 3)
export DISPLAY=:1
export LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Must run from ProtoMotions dir (USD assets use relative paths)
cd ~/code/ProtoMotions
python ~/code/closd_isaaclab/scripts/run_closd_isaaclab.py \
    --prompt "a person walks forward" \
    --tracker-checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --num-envs 1
```

---

## Architecture

```
Text prompt
    │
    ▼
DiP diffusion model (CLoSD)
    │  263-dim HumanML3D motion sequence
    ▼
HML conversion  →  22-joint 3D positions
    │
    ▼
Rotation solver  →  6D joint rotations (SMPL skeleton)
    │
    ▼
Robot state builder  →  Isaac Lab joint state tensor
    │
    ▼
ProtoMotions motion tracker  (Isaac Lab physics sim)
    │
    └──► observation  ──► (future: re-condition diffusion)
```

The `closd_isaaclab` package bridges the three repos:

| Module | Role |
|--------|------|
| `diffusion/motion_provider` | Wraps CLoSD DiP inference |
| `diffusion/hml_conversion` | HumanML3D → 3D joint positions |
| `diffusion/rotation_solver` | 3D positions → 6D rotations |
| `diffusion/robot_state_builder` | Rotations → Isaac Lab state tensor |
| `integration/closd_motion_lib` | ProtoMotions-compatible motion library |
| `integration/closd_motion_manager` | Closed-loop episode management |
| `utils/coord_transform` | Coordinate frame utilities |
| `utils/fps_convert` | Frame-rate resampling |

---

## Running Tests

```bash
source ~/code/env_isaaclab/bin/activate
cd ~/code/closd_isaaclab
pytest tests/ -v
```
