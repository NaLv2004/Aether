@echo off
echo === COMPARATIVE EVALUATION: AREA SIZE SENSITIVITY ===

echo [SCENARIO: DENSE - 0.5km]
python evaluation.py --area_size 0.5 --model_path qat_gnn_model.pth --test_samples 1000

echo.
echo [SCENARIO: STANDARD - 1.0km]
python evaluation.py --area_size 1.0 --model_path qat_gnn_model.pth --test_samples 1000

echo.
echo [SCENARIO: SPARSE - 2.0km]
python evaluation.py --area_size 2.0 --model_path qat_gnn_model.pth --test_samples 1000

echo.
echo === BIT DISTRIBUTION ANALYSIS (Using joint_qat script for stats) ===
python joint_qat.py --area_size 0.5 --epochs 0 --pretrained_model qat_gnn_model.pth
python joint_qat.py --area_size 2.0 --epochs 0 --pretrained_model qat_gnn_model.pth