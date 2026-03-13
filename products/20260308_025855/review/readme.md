## [Current Step]
# Comprehensive Evaluation V14 (Rebuttal Version)

This evaluation script (`eval_v14_rebuttal.py`) addresses multiple reviewer comments:

1. **M8 (Multiple Scenarios / Single Scenario Bias):** Modifies the evaluation framework to average metrics across 5 different large-scale fading geometries (seeds 42 to 46). Rather than evaluating on a single fixed AP/UE spatial distribution, tests are now repeatedly simulated over `seed`, `seed+1`, ... `seed+4`, and BER/bits results are averaged out to report generalized performance.
2. **M6 (AP Dropout Validation):** Includes a new `Task 6: AP Dropout Robustness` to test the resiliency of the model when link fail-overs happen. It evaluates the network at 10dBm P_tx with an AP random dropout rate of `[0.0, 0.1, 0.3, 0.5]`. Local inputs (`v`, `H`, `y`) are zero-masked according to the dropout probability before being processed. BER degradation is directly calculated and reported.
3. **M5 (Stronger/Inadequate Baselines):** 
   - **M5(a)**: Enhances the `C-MMSE-MxQ` unlearned baseline by replacing flat 8-bit uniform quantization with a smart bit-allocation algorithm proportional to local channel strengths. The given budget ensures fair comparison against the Gumbel-Softmax learned intelligence.
   - **M5(b)**: Introduces `MMSE-PIC` (Parallel Interference Cancellation), a classical strong iterative benchmark added to Perfect CSI environments (`tau=0`).

## Execution Command-Line Arguments
- `--model_path`: Path to load the previously trained state dict (default: `new_joint_model_v13.pth`).
- `--L`, `--N`, `--K`: Dimensions defining the setup (default: 16 APs, 4 Antennas, 8 Users).
- `--area_size`: Physical simulation footprint.
- `--bandwidth`, `--noise_psd`: Fundamental physical layer metrics.
- `--test_samples`: Number of Monte-Carlo test samples to smooth variations (will be automatically subdivided equally amongst the 5 geometry seeds).
- `--hidden_dim`: Defines internal network dimensions.
- `--seed`: Base anchor random seed used for generating reproducible spatial distributions.