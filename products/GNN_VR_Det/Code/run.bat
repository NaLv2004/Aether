@echo off
echo ==================================================
echo Step 4: Comparing Wideband Sparse-GNN with Wideband Baseline
echo Fixed Parameters: N=256, K=8, F=16, f_c=0.1e12, B=10e9
echo Evaluation SNR List: [-5, 0, 5, 10, 15] dB
echo Number of Test Samples: 5000
echo ==================================================

echo.
echo --- [Baseline] Running Wideband VR-MMSE and VR-ZF ---
python baselines_wideband.py --num_samples 5000 --N 256 --K 8 --F 16 --f_c 0.1e12 --B 10e9 --batch_size 50 --snr_list -5 0 5 10 15

echo.
echo --- [Proposed] Running Fully-Trained Wideband Sparse-GNN (S=128) ---
python sparse_gnn_wideband.py --N 256 --K 8 --F 16 --S 128 --f_c 0.1e12 --B_bw 10e9 --num_train 1000 --num_val 2000 --num_test 5000 --epochs 100 --batch_size 16 --lr 0.001 --hidden_dim 64 --num_layers 4 --snr_min -5 --snr_max 15 --train_snr 10.0 --snr_list -5 0 5 10 15

echo.
echo ==================================================
echo Step 4 Execution Completed!
echo ==================================================