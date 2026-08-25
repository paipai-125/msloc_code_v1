


# 冯越小模型实验部分如下
python DeMamba/train.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42

python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/method/eval \
  --device-ids 0 \
  --val-batch-size 16

python evaluate_long.py \
    --gt_file "../MSLoc_data/test_all_1209_0119_long.json" \
    --infer_file "../MSLoc_data/DeMamba/full/method/eval/predictions.json"













# MSLoc：XCLIP 神经元探测与 DeMamba 训练实验

本说明覆盖当前第一阶段实验：冻结本地预训练的 XCLIP，利用真假视频对探测三类敏感神经元，将最终固定的 768 维特征直接送入 Mamba 和四分类头训练，并与 XCLIP baseline 对比。

所有命令均在服务器的 `MSLoc_code` 目录执行；命令中的所有路径均为相对路径。数据、预训练模型、缓存、中间结果、模型权重、评测与可视化结果均写入同级目录 `../MSLoc_data`。

## 1. 环境安装与检查

```bash
conda create -n msloc python=3.10 -y
conda activate msloc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r DeMamba/requirements.txt
```

安装 FFmpeg：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

确认本地 XCLIP 可以离线加载：

```bash
python -c "from transformers import XCLIPVisionModel; m=XCLIPVisionModel.from_pretrained('../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16', local_files_only=True); print('offline XCLIP loaded:', m.config.hidden_size, m.config.num_hidden_layers)"
```

预期输出包含 `offline XCLIP loaded: 768 12`。


## 2. 下载数据集和模型权重

```bash
cd ..
modelscope login --token ms-412c41b7-1f64-483e-9ab2-f81cc7c04525
modelscope download --dataset L67plus/TASLE --local-dir ./
cat MSLoc_assets.tar.gz.part-* > MSLoc_data.tar.gz
tar -xzf MSLoc_data.tar.gz
```

解压后目录如下，**注意文件夹名称需要改成 `MSLoc_data`**

```text
MSLoc_data/
├── data
├── DeMamba
└── Trace
```

## 3. 抽帧

```bash
python DeMamba/Preprocess/video2frame.py \
  --input_root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output_root ../MSLoc_data/DeMamba/video_frames \
  --num_workers 8
```

## 4. 生成全量配置文件

```bash
mkdir -p ../MSLoc_data/DeMamba/full/configs

cat > ../MSLoc_data/DeMamba/full/configs/xclip_baseline_full.yaml <<'YAML'
model: 'XCLIP_DeMamba_4'
tuning_mode: 'lp'
task: 'many2many'

save_dir: '../MSLoc_data/DeMamba/full/baseline/results'
xclip_model_path: '../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16'

max_epoch: 10
bath_per_epoch: 1000
train_batch_size: 2
val_batch_size: 2
num_workers: 2
lr: 0.000001

train_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json'
test_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json'
dataset_base_path: '../MSLoc_data/DeMamba/video_frames'
window_length: 2.0
frames_per_window: 8
mode: 'four_class'
transform_config: {
  crop_youku: True,
  normalization: 'clip'
}
YAML

cat > ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml <<'YAML'
model: 'XCLIP_NeuronDeMamba_4'
tuning_mode: 'lp'
task: 'many2many'

save_dir: '../MSLoc_data/DeMamba/full/method/results'
neuron_indices_path: '../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_indices.json'
xclip_model_path: '../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16'

max_epoch: 10
bath_per_epoch: 1000
train_batch_size: 2
val_batch_size: 2
num_workers: 2
lr: 0.000001

train_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json'
test_json_path: '../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json'
dataset_base_path: '../MSLoc_data/DeMamba/video_frames'
window_length: 2.0
frames_per_window: 8
mode: 'four_class'
transform_config: {
  crop_youku: True,
  normalization: 'clip'
}
YAML
```

## 5. 构造全量神经元探测对

```bash
mkdir -p ../MSLoc_data/DeMamba/full/method/neuron_probe

python DeMamba/build_probe_pairs.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --output ../MSLoc_data/DeMamba/full/method/neuron_probe/train_pairs_full.jsonl \
  --window-length 2.0 \
  --boundary-min-context 0.50
```

## 6. 探测并保存最终 768 个神经元

```bash
python DeMamba/probe_xclip_neurons.py \
  --pairs ../MSLoc_data/DeMamba/full/method/neuron_probe/train_pairs_full.jsonl \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --model-path ../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16 \
  --output-dir ../MSLoc_data/DeMamba/full/method/neuron_probe \
  --top-ratio 0.10 \
  --final-neuron-count 768 \
  --min-neurons-per-target 256 \
  --frames-per-window 8 \
  --crop-youku \
  --amp \
  --strict
```

## 7. 训练与评测

训练使用 PyTorch `DataParallel`。单卡使用 `--device-ids 0`，单机 8 卡使用 `--device-ids 0,1,2,3,4,5,6,7`；不要使用 `torchrun`。


### 7.1 Baseline：全维冻结 XCLIP 特征 + Mamba + 分类头

训练：

```bash
python DeMamba/train.py \
  --config ../MSLoc_data/DeMamba/full/XCLIP_Tasle_baseline_full.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 16 \
  --val-batch-size 16 \
  --max-epoch 10 \
  --seed 3407
```

评测：

```bash
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/XCLIP_Tasle_baseline_full.yaml \
  --model_path ../MSLoc_data/DeMamba/full/baseline/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/baseline/eval \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16
```

### 7.2 方法：768 个探测神经元 + Mamba + 分类头

训练：

```bash
python DeMamba/train.py \
  --config ../MSLoc_data/DeMamba/full/XCLIP_Tasle_neurons_full.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 16 \
  --val-batch-size 16 \
  --max-epoch 10 \
  --seed 3407
```

评测：

```bash
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/XCLIP_Tasle_neurons_full.yaml \
  --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/method/eval \
  --device-ids 0 \
  --val-batch-size 16
```

`tuning_mode: lp` 在本实验中表示编码器冻结：XCLIP 不训练，Mamba 和分类头训练。学习率调度器在每个 epoch 的全部 `optimizer.step()` 完成后才调用。


## 8. 可视化

神经元探测可视化：生成 fake、R2F、F2R 三个目标的逐层得分热力图、最终 768 神经元的层分布图和 CSV。

```bash
python DeMamba/visualize_xclip_neuron_heatmaps.py \
  --scores ../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_scores.npz \
  --indices ../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_indices.json \
  --output-dir ../MSLoc_data/DeMamba/full/method/neuron_probe/visualizations
```

预测时间线可视化：

```bash
python DeMamba/visualize_predictions.py \
  --predictions ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval/visualizations \
  --max-videos 30
```

## 9. 可选：小样本调试

全量流程不需要调用本节。若只想验证代码和环境，可以使用 `DeMamba/make_paired_subset.py` 生成小样本

```bash
python DeMamba/make_paired_subset.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --output ../MSLoc_data/DeMamba/debug/train_20.json \
  --fake-count 20
```