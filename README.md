

# MEFormer – Proposal Quality Analysis and Auxiliary Loss Extension

This repository is an **extension** of the original [MEFormer](https://github.com/hanchaa/MEFormer) multimodal 3D object detection framework.  
Please **follow the installation and environment setup instructions** from the original repository before proceeding with any custom scripts or experiments provided here.

---

## 🔧 Setup Instructions

1. **Clone the original MEFormer repo and install dependencies**:
    - Follow: https://github.com/hanchaa/MEFormer

2. **Clone this repo into your workspace or copy the files into your existing MEFormer directory**.

3. **Install Python dependencies** (if not already done):
    ```bash
    pip install -r requirements.txt
    ```

4. **Make shell scripts executable** (optional):
    
    ```bash
    chmod +x train_proposal.sh train_without_pme.sh run_proposal_test.sh
    ```
    
    Use `environment.yml` if you want to recreate the entire Conda environment (including channels and dependencies). You can recreate it using:
    
    ```
    conda env create -f environment.yml
    ```

---

## 📁 Directory Highlights

- `proposal_dumps.pkl`  
  → Stores intermediate proposals extracted before decoder/NMS.

- `final_detections.pkl`  
  → Stores final MEFormer predictions after full decoding.

- `figures/`  
  → Contains recall curves and visual comparisons of proposals vs. final detections.

- `analysis/`  
  → Contains proposal recall evaluation code.

- `test_results/`  
  → Optionally used to store inference visualizations.

- `train_proposal.sh` / `train_without_pme.sh`  
  → Training scripts for experiments with/without PME module or auxiliary loss.

---

## 📈 Custom Features Implemented

### 1. **Proposal Quality Analysis**
- Analyze raw 3D proposals (`proposal_dumps.pkl`) and compare with final detections (`final_detections.pkl`).
- Plots:
    - BEV visualizations across multiple frames.
    - Raw proposal generation scripts and SLURM training files
    - 

### 2. **Auxiliary Loss Implementation**
- Custom auxiliary loss supervises the predicted center points of the decoder output.
- Controlled by `lambda` weight parameter (e.g., 0.1 or 0.05).
- Results summarized in training logs and plotted.

---

## 📝 Notes

- Ensure your `MEFormer` config and checkpoint paths are correctly updated in the `.sh` scripts.
- Use the same dataset structure (nuScenes) as in the original repo.
- Figures were generated using `wandb` and matplotlib — logs are stored in `wandb/` if available.

---

## 📜 License

This extension is provided under the same license terms as the original MEFormer repository.  
Refer to the `LICENSE` file for details.

---

For questions or academic use, cite the original MEFormer paper and refer to the GitHub repository:  
🔗 https://github.com/hanchaa/MEFormer