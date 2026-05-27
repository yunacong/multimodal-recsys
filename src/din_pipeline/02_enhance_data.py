
'''
Enhanced DIN 数据 - 加 sub_cat / brand / price 辅助特征
'''
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('/Users/wangfengyuan/projects/multimodal-recsys')
DIN_DIR = PROJECT_ROOT / 'data' / 'processed' / 'din'
DATA_PROC = PROJECT_ROOT / 'data' / 'processed'

print('=' * 70)
print('Enhanced DIN 数据 - 加辅助特征')
print('=' * 70)

# Step 1: 加载 item meta
print()
print('[1/5] 加载 item meta 特征')
t0 = time.time()
item_meta = pd.read_csv(DATA_PROC / 'item_meta_features.csv')
print(f'  shape: {item_meta.shape}')
print(f'  columns: {list(item_meta.columns)}')

item_vocab = pd.read_csv(DIN_DIR / 'item_vocab.csv')
item_vocab.rename(columns={'item_id': 'parent_asin'}, inplace=True)

meta_with_idx = item_vocab.merge(item_meta, on='parent_asin', how='left')
print(f'  merge: {meta_with_idx.shape}')
cov = meta_with_idx['sub_category_id'].notna().sum() / len(meta_with_idx) * 100
print(f'  覆盖率: {cov:.1f}%')

# Step 2: 构造 idx -> features 的 array
n_items = int(meta_with_idx['idx'].max()) + 1
sub_cat_arr = np.full(n_items, -1, dtype=np.int32)
brand_arr = np.full(n_items, -1, dtype=np.int32)
price_arr = np.zeros(n_items, dtype=np.float32)
price_missing_arr = np.ones(n_items, dtype=np.int8)

for _, row in meta_with_idx.iterrows():
    idx = int(row['idx'])
    if pd.notna(row.get('sub_category_id')):
        sub_cat_arr[idx] = int(row['sub_category_id'])
    if pd.notna(row.get('brand_id')):
        brand_arr[idx] = int(row['brand_id'])
    p = row.get('price')
    if pd.notna(p) and p > 0:
        price_arr[idx] = float(p)
        price_missing_arr[idx] = 0

# 处理 -1 → max+1 (作为 'unknown' 类别)
n_sub_cats = int(sub_cat_arr.max()) + 2
n_brands = int(brand_arr.max()) + 2

sub_cat_arr[sub_cat_arr == -1] = n_sub_cats - 1
brand_arr[brand_arr == -1] = n_brands - 1

print(f'  n_sub_cats: {n_sub_cats}')
print(f'  n_brands: {n_brands}')
print(f'  price 缺失: {int(price_missing_arr.sum()):,}/{n_items:,}')

# Price 归一化
prices_valid = price_arr[price_missing_arr == 0]
log_prices = np.log1p(prices_valid)
log_p_min = float(log_prices.min())
log_p_max = float(log_prices.max())
print(f'  price 范围: {float(prices_valid.min()):.2f} - {float(prices_valid.max()):.2f}')
print(f'  log_price 范围: {log_p_min:.2f} - {log_p_max:.2f}')

price_norm = np.zeros(n_items, dtype=np.float32)
mask = price_missing_arr == 0
price_norm[mask] = (np.log1p(price_arr[mask]) - log_p_min) / (log_p_max - log_p_min + 1e-9)
print(f'  耗时: {time.time()-t0:.1f}s')


def enhance_dataset(npz_path, name):
    print(f'\n[{name}] 加载 + 增强...')
    data = dict(np.load(npz_path))
    n = len(data['label'])
    
    data['target_sub_cat'] = sub_cat_arr[data['target']].astype(np.int32)
    data['target_brand'] = brand_arr[data['target']].astype(np.int32)
    data['target_price'] = price_norm[data['target']].astype(np.float32)
    
    L = data['history'].shape[1]
    history_flat = data['history'].flatten()
    
    data['history_sub_cat'] = sub_cat_arr[history_flat].reshape(n, L).astype(np.int32)
    data['history_brand'] = brand_arr[history_flat].reshape(n, L).astype(np.int32)
    data['history_price'] = price_norm[history_flat].reshape(n, L).astype(np.float32)
    
    print(f'  [{name}] enhanced: {n:,} 行, 字段数: {len(data)}')
    return data


print()
print('[2/5] Enhance train')
train_enhanced = enhance_dataset(DIN_DIR / 'din_train.npz', 'train')

print()
print('[3/5] Enhance val')
val_enhanced = enhance_dataset(DIN_DIR / 'din_val.npz', 'val')

print()
print('[4/5] Enhance test')
test_enhanced = enhance_dataset(DIN_DIR / 'din_test.npz', 'test')

print()
print('[5/5] 保存 enhanced 数据')
t0 = time.time()
np.savez_compressed(DIN_DIR / 'din_train_enhanced.npz', **train_enhanced)
np.savez_compressed(DIN_DIR / 'din_val_enhanced.npz', **val_enhanced)
np.savez_compressed(DIN_DIR / 'din_test_enhanced.npz', **test_enhanced)

for name in ['train', 'val', 'test']:
    p = DIN_DIR / f'din_{name}_enhanced.npz'
    print(f'  din_{name}_enhanced.npz: {p.stat().st_size/1e6:.0f} MB')

with open(DIN_DIR / 'meta.json') as f:
    meta = json.load(f)

meta['enhanced'] = True
meta['n_sub_cats'] = int(n_sub_cats)
meta['n_brands'] = int(n_brands)
meta['price_log_min'] = log_p_min
meta['price_log_max'] = log_p_max

with open(DIN_DIR / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print()
print('=' * 70)
print('🎉 Enhanced DIN 数据完成!')
print('=' * 70)
print(f'保存耗时: {time.time()-t0:.1f}s')
print(json.dumps(meta, indent=2))
