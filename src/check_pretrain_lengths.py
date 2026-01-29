import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from config.model_config import MODEL_NAME

def analyze_pretrain_data(data_root, tokenizer_name=None):
    """
    Phân tích độ dài của samples trong pretrain dataset
    """
    data_root = Path(data_root)
    if tokenizer_name is None:
        tokenizer_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Statistics
    stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'max_words': 0,
        'max_tokens': 0,
        'min_words': float('inf'),
        'min_tokens': float('inf'),
        'avg_words': 0,
        'avg_tokens': 0,
        'sum_words': 0,
        'sum_tokens': 0,
        'length_distribution': {
            '0-100': 0,
            '100-500': 0,
            '500-1000': 0,
            '1000-4000': 0,
            '4000-5000': 0,
            '5000-10000': 0,
            '10000-20000': 0,
            '20000-50000': 0,
            '50000+': 0
        },
        'longest_samples': []  # Top 10 longest samples
    }
    
    # Find all data directories
    data_dirs = sorted([d for d in data_root.iterdir() 
                       if d.is_dir() and d.name.startswith('data_')])
    
    if not data_dirs:
        data_dirs = [data_root]
    
    print("=" * 80)
    print("🔍 ANALYZING PRETRAIN DATASET LENGTHS")
    print("=" * 80)
    
    all_samples_info = []
    
    for data_dir in data_dirs:
        print(f"\n📂 Processing directory: {data_dir}")
        
        imdb_files = sorted(data_dir.glob("imdb_pretrain_p*.npy"))
        pdf_root = data_dir / "pdfs"
        
        if not imdb_files:
            continue
        
        for imdb_file in imdb_files:
            print(f"\n   📄 Analyzing {imdb_file.name}...")
            data = np.load(imdb_file, allow_pickle=True)
            
            # Skip metadata
            if len(data) > 0 and not isinstance(data[0], dict):
                data = data[1:]
            
            for sample in tqdm(data, desc="   Processing samples"):
                stats['total_samples'] += 1
                
                if not isinstance(sample, dict):
                    continue
                
                image_id = sample.get('image_id')
                ocr_tokens = sample.get('ocr_tokens')
                
                if not image_id or not ocr_tokens:
                    continue
                
                # Check PDF exists
                pdf_path = pdf_root / image_id / f"{image_id}.pdf"
                if not pdf_path.exists():
                    continue
                
                stats['valid_samples'] += 1
                
                # Calculate lengths
                num_words = len(ocr_tokens)
                text = " ".join(ocr_tokens)
                
                # Tokenize to get token count
                encoding = tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_tensors="pt"
                )
                num_tokens = encoding['input_ids'].shape[1]
                
                # Update statistics
                stats['sum_words'] += num_words
                stats['sum_tokens'] += num_tokens
                stats['max_words'] = max(stats['max_words'], num_words)
                stats['max_tokens'] = max(stats['max_tokens'], num_tokens)
                stats['min_words'] = min(stats['min_words'], num_words)
                stats['min_tokens'] = min(stats['min_tokens'], num_tokens)
                
                # Update distribution
                if num_tokens < 100:
                    stats['length_distribution']['0-100'] += 1
                elif num_tokens < 500:
                    stats['length_distribution']['100-500'] += 1
                elif num_tokens < 1000:
                    stats['length_distribution']['500-1000'] += 1
                elif num_tokens < 4000:
                    stats['length_distribution']['1000-4000'] += 1
                elif num_tokens < 5000:
                    stats['length_distribution']['4000-5000'] += 1
                elif num_tokens < 10000:
                    stats['length_distribution']['5000-10000'] += 1
                elif num_tokens < 20000:
                    stats['length_distribution']['10000-20000'] += 1
                elif num_tokens < 50000:
                    stats['length_distribution']['20000-50000'] += 1
                else:
                    stats['length_distribution']['50000+'] += 1
                
                # Store sample info
                sample_info = {
                    'doc_id': image_id,
                    'num_words': num_words,
                    'num_tokens': num_tokens,
                    'file': imdb_file.name,
                    'dir': data_dir.name
                }
                all_samples_info.append(sample_info)
    
    # Calculate averages
    if stats['valid_samples'] > 0:
        stats['avg_words'] = stats['sum_words'] / stats['valid_samples']
        stats['avg_tokens'] = stats['sum_tokens'] / stats['valid_samples']
    
    # Get top 10 longest samples
    stats['longest_samples'] = sorted(
        all_samples_info, 
        key=lambda x: x['num_tokens'], 
        reverse=True
    )[:200]
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total samples scanned: {stats['total_samples']:,}")
    print(f"   Valid samples: {stats['valid_samples']:,}")
    
    print(f"\n📏 Length Statistics:")
    print(f"   Max words: {stats['max_words']:,}")
    print(f"   Max tokens: {stats['max_tokens']:,}")
    print(f"   Min words: {stats['min_words']:,}")
    print(f"   Min tokens: {stats['min_tokens']:,}")
    print(f"   Avg words: {stats['avg_words']:,.1f}")
    print(f"   Avg tokens: {stats['avg_tokens']:,.1f}")
    
    print(f"\n📊 Token Length Distribution:")
    for range_key, count in stats['length_distribution'].items():
        percentage = (count / stats['valid_samples'] * 100) if stats['valid_samples'] > 0 else 0
        print(f"   {range_key:15} tokens: {count:6,} samples ({percentage:5.2f}%)")
    
    print(f"\n🔥 Top 10 Longest Samples:")
    print(f"   {'Rank':<6} {'Doc ID':<30} {'Words':<10} {'Tokens':<10} {'File'}")
    print(f"   {'-'*6} {'-'*30} {'-'*10} {'-'*10} {'-'*40}")
    for i, sample in enumerate(stats['longest_samples'], 1):
        print(f"   {i:<6} {sample['doc_id']:<30} {sample['num_words']:<10,} "
              f"{sample['num_tokens']:<10,} {sample['file']}")
    
    # Memory estimation for longest sample
    longest = stats['longest_samples'][0] if stats['longest_samples'] else None
    if longest:
        num_tokens = longest['num_tokens']
        attention_memory_gb = (num_tokens ** 2 * 12 * 4) / 1e9  # 12 heads, FP32
        activations_memory_gb = (num_tokens * 768 * 12 * 4) / 1e9  # 12 layers
        total_memory_gb = attention_memory_gb + activations_memory_gb + 2.64  # params + grad + optimizer
        
        print(f"\n💾 Memory Estimation for Longest Sample:")
        print(f"   Tokens: {num_tokens:,}")
        print(f"   Attention memory: {attention_memory_gb:.2f} GB")
        print(f"   Activations memory: {activations_memory_gb:.2f} GB")
        print(f"   Total estimated: {total_memory_gb:.2f} GB")
        
        if total_memory_gb > 40:
            print(f"   ⚠️  WARNING: This will OOM on A100 40GB!")
        if total_memory_gb > 80:
            print(f"   🚨 CRITICAL: This will OOM on A100 80GB!")
    
    # Save detailed report
    report_path = Path("./logs/pretrain_length_analysis_p00.json")
    with open(report_path, 'w') as f:
        # Convert to serializable format
        report = {
            'summary': {
                'total_samples': stats['total_samples'],
                'valid_samples': stats['valid_samples'],
                'max_words': stats['max_words'],
                'max_tokens': stats['max_tokens'],
                'min_words': stats['min_words'],
                'min_tokens': stats['min_tokens'],
                'avg_words': stats['avg_words'],
                'avg_tokens': stats['avg_tokens'],
            },
            'distribution': stats['length_distribution'],
            'longest_samples': stats['longest_samples']
        }
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    import os
    
    # Get data path from environment or use default
    data_root = os.getenv("PRETRAIN_DATA_PATH", "../data")
    
    print(f"Data root: {data_root}")
    print(f"Tokenizer: {MODEL_NAME}")
    
    stats = analyze_pretrain_data(data_root)