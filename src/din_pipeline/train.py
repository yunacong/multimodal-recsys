"""DIN 训练脚本

支持:
  - GPU/MPS/CPU 自动选择
  - AUC 评估
  - 早停 (patience=2)
  - 模型保存 (best by val AUC)
  - 日志输出
"""

import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import DINDataset, collate_fn
from model import DIN


def get_device():
    """自动选择设备: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device):
    """在 val/test 集上评估, 返回 AUC + Loss"""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            batch_device = {k: v.to(device) for k, v in batch.items() if k != "label"}
            
            logits = model(batch_device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    auc = roc_auc_score(all_labels, all_logits)
    avg_loss = total_loss / n_batches
    
    model.train()
    return {"auc": auc, "loss": avg_loss, "n_samples": len(all_labels)}


def train_one_epoch(model, loader, optimizer, device, log_interval=100):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    t_start = time.time()
    
    all_logits = []
    all_labels = []
    
    for step, batch in enumerate(loader):
        labels = batch["label"].to(device)
        batch_device = {k: v.to(device) for k, v in batch.items() if k != "label"}
        
        optimizer.zero_grad()
        logits = model(batch_device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            samples_per_sec = (step + 1) * loader.batch_size / elapsed
            avg_loss = total_loss / n_batches
            print(f"  Step {step+1}/{len(loader)}: loss {avg_loss:.4f} | {samples_per_sec:.0f} samples/sec | elapsed {elapsed:.0f}s")
    
    # 训练 AUC (作为参考)
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    train_auc = roc_auc_score(all_labels, all_logits)
    
    return {
        "loss": total_loss / n_batches,
        "auc": train_auc,
        "time": time.time() - t_start,
    }


def main(args):
    print("=" * 70)
    print("DIN 训练")
    print("=" * 70)
    
    # ====== 配置 ======
    DIN_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Num workers: {args.num_workers}")
    
    # ====== 数据 ======
    print("\n加载数据...")
    t0 = time.time()
    train_ds = DINDataset(DIN_DIR / "din_train.npz")
    val_ds = DINDataset(DIN_DIR / "din_val.npz")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"  train: {len(train_ds):,}, batches: {len(train_loader):,}")
    print(f"  val:   {len(val_ds):,}, batches: {len(val_loader):,}")
    print(f"  加载耗时: {time.time()-t0:.1f}s")
    
    # ====== 模型 ======
    with open(DIN_DIR / "meta.json") as f:
        meta = json.load(f)
    n_items = meta["n_items"] + 1  # +1 for padding idx 0
    
    print(f"\n模型: n_items={n_items}, emb_dim={args.emb_dim}, mlp_hidden={args.mlp_hidden}")
    model = DIN(
        n_items=n_items,
        emb_dim=args.emb_dim,
        mlp_hidden=args.mlp_hidden,
        seq_len=meta["seq_len"],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ====== 训练循环 ======
    history = []
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        print("\nTraining...")
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, log_interval=args.log_interval)
        print(f"  Train: loss {train_metrics['loss']:.4f}, AUC {train_metrics['auc']:.4f}, time {train_metrics['time']:.0f}s")
        
        # Eval
        print("\nEvaluating on val...")
        t0 = time.time()
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val: loss {val_metrics['loss']:.4f}, AUC {val_metrics['auc']:.4f}, time {time.time()-t0:.0f}s")
        
        # 保存历史
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "time": train_metrics["time"],
        })
        
        # 早停 + 保存最佳模型
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            model_path = OUT_DIR / "din_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_metrics["auc"],
                "args": vars(args),
            }, model_path)
            print(f"  ✅ 新最佳! val AUC {best_val_auc:.4f}, 保存到 {model_path}")
        else:
            patience_counter += 1
            print(f"  ⚠️  Val AUC 没提升, patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n🛑 早停 (patience={args.patience} 用完)")
                break
        
        # 保存训练历史
        with open(OUT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🎉 训练完成!")
    print(f"{'='*70}")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"\n对比基准:")
    print(f"  LightGBM v3-mpnet: AUC 0.8122")
    print(f"  DIN (本次):        AUC {best_val_auc:.4f}")
    print(f"  Delta:             {best_val_auc - 0.8122:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/din", type=str)
    parser.add_argument("--out_dir", default="models/din", type=str)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--mlp_hidden", default=64, type=int)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()
    
    main(args)
