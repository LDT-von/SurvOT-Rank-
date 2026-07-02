@echo off
setlocal
cd /d "C:\Users\cwnu\Desktop\新建文件夹 (2)\45_otehv2_rankevent"

set PYTHON=python
set GPU=0

set DATA_ROOT=E:\TCGA-WSI-data\blca\uni
set DATA_PATH=..\SlotSPE\dataset_csv
set RESULT_DIR=.\results_v45_5fold_30ep
set SLOTSPE_DIR=..\SlotSPE

if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

echo [%date% %time%] starting V45 5-fold x 30 epoch
echo [%date% %time%] DATA_ROOT=%DATA_ROOT%
echo [%date% %time%] DATA_PATH=%DATA_PATH%
echo [%date% %time%] RESULT_DIR=%RESULT_DIR%
echo [%date% %time%] SLOTSPE_DIR=%SLOTSPE_DIR%

%PYTHON% -u train.py ^
  --n_classes 4 --num_patches 2048 --encoding_dim 1024 ^
  --max_epochs 30 --batch_size 4 --seed 3 --study blca ^
  --rna_format Pathways --label_col survival_months_dss --bag_loss nll_surv ^
  --signature combine --slot_num_wsi 8 --slot_num_omics 8 ^
  --slot_iters 5 --temperature 0.01 --topk_ratio 0.25 ^
  --top_k_method parallel_topk_st --k_start 0 --k_end 5 ^
  --lr 5e-4 --gpu %GPU% --num_workers 4 ^
  --data_root_dir "%DATA_ROOT%" --data_path "%DATA_PATH%" ^
  --results_dir "%RESULT_DIR%" --specific_simple "v45_5fold_30ep" ^
  --otehv2_eps 0.05 --otehv2_iter 50 --otehv2_warmup 5 ^
  --otehv2_num_events 24 --otehv2_heads 4 --otehv2_layers 4 ^
  --otehv2_dropout 0.1 ^
  --lambda_otehv2_ot 0.06 --lambda_otehv2_div 0.01 ^
  --lambda_otehv2_event_surv 0.25 --lambda_otehv2_recon 0.2 ^
  --lambda_rankevent_per_event 0.15 --lambda_rankevent_rank 0.15 ^
  --lambda_rankevent_global_cons 0.02 --lambda_rankevent_gate_ent 0.005 ^
  --rankevent_eps_start 0.10 --rankevent_eps_end 0.05 ^
  --rankevent_eps_anneal_epochs 12 --rankevent_global_init -2.0

echo [%date% %time%] DONE
