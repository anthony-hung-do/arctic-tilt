from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import numpy as np
import os

PRETRAIN_DATA_PATH = os.getenv("PRETRAIN_DATA_PATH", "../data")

class DocVQADataset(Dataset):
    def __init__(self, ds, tokenizer, max_source_length=512, max_target_length=128,
                 resize_scale=(512, 384), transform=None, language='en',
                 use_chunked_processing=True):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.resize_scale = resize_scale
        self.transform = transform if transform is not None else ToTensor()
        self.language = language
        self.use_chunked_processing = use_chunked_processing

    def __len__(self):
        return len(self.ds)

    def normalize_bbox(self, bbox, width, height):
        """Chuẩn hóa bounding box về [0, 1000]"""
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def __getitem__(self, idx):
        item = self.ds[idx]

        # 1. Xử lý hình ảnh
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        original_width, original_height = image.size
        resized_image = image.copy().resize(self.resize_scale)
        img_tensor = self.transform(resized_image)

        # 2. Lấy text và bounding boxes từ OCR
        words = item['words']
        bboxes = item['bounding_boxes']

        # 3. Lấy câu hỏi
        question = item['query'][self.language]

        # 4. Lấy câu trả lời đầu tiên
        answers = item['answers']
        target_text = answers[0] if answers else ""

        # 5. Tạo input text: question + OCR text
        ocr_text = " ".join(words)
        input_text = f"question: {question} context: {ocr_text}"

        if self.use_chunked_processing:
            # Với chunked processing: KHÔNG truncate, KHÔNG pad
            # Để ChunkedProcessor xử lý việc chia chunks
            source_encoding = self.tokenizer(
                input_text,
                truncation=False,
                padding=False,
                return_tensors="pt"
            )

            # Target vẫn pad như bình thường
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            # Tạo bounding boxes cho độ dài thực tế (không pad)
            actual_length = source_encoding['input_ids'].shape[1]

        else:
            # Không có chunked processing: Truncate và pad như cũ
            source_encoding = self.tokenizer(
                input_text,
                max_length=self.max_source_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            actual_length = self.max_source_length

        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 8. Chuẩn hóa bounding boxes
        normalized_bboxes = []
        for i, bbox in enumerate(bboxes):
            if i >= len(words):
                break
            norm_bbox = self.normalize_bbox(bbox, original_width, original_height)
            normalized_bboxes.append(norm_bbox)

        # 9. Tạo bounding boxes cho từng token
        # Tokenize words để lấy word_ids
        question_part = f"question: {question} context: "
        question_tokens = self.tokenizer.encode(question_part, add_special_tokens=False)
        question_length = len(question_tokens)

        word_encoding = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        word_ids = word_encoding.word_ids()

        # Tạo bbox array cho actual sequence length
        bbox_array = []
        for i in range(actual_length):
            if i < question_length:
                # Question tokens → padding bbox
                bbox_array.append([0, 0, 0, 0])
            else:
                # Context tokens → real bbox
                context_idx = i - question_length
                if context_idx < len(word_ids) and word_ids[context_idx] is not None:
                    word_idx = word_ids[context_idx]
                    if word_idx < len(normalized_bboxes):
                        bbox_array.append(normalized_bboxes[word_idx])
                    else:
                        bbox_array.append([0, 0, 0, 0])
                else:
                    bbox_array.append([0, 0, 0, 0])

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'bboxes': torch.tensor(bbox_array, dtype=torch.long),
            'pixel_values': img_tensor,
            'question': question,
            'answer': target_text,
            'doc_id': item['id'],
            'original_length': actual_length  # ← Track độ dài thực tế
        }
    

import torch
import torch.nn as nn
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import pdf2image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from tqdm import tqdm
import random
from torchvision import transforms
from typing import Tuple, List

class PretrainDataset(Dataset):
    """
    Self-supervised pretraining dataset for Arctic-TILT
    Uses Masked Language Modeling (MLM) approach
    """
    def __init__(
        self,
        data_root,
        tokenizer,
        max_length=512,
        mlm_probability=0.15,
        transform=None,
        use_chunked_processing=True,
        max_tokens_limit=5000 
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.max_tokens_limit = max_tokens_limit
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1)
        ])
        self.use_chunked_processing = use_chunked_processing

        # Filter valid samples
        # self.valid_samples = self._filter_valid_samples()
        # print(f"✅ Loaded {len(self.valid_samples)}/{len(self.data)} valid pretrain samples")
        self.samples_info = self._load_all_imdb_files()
        print(f"✅ Loaded {len(self.samples_info)} valid pretrain samples from all files")

    def _load_all_imdb_files(self):
        """Load and validate samples from all imdb_pretrain_p*.npy files"""
        all_samples = []
        
        # Find all data_* directories
        data_dirs = sorted([d for d in self.data_root.iterdir() 
                           if d.is_dir() and d.name.startswith('data_')])
        
        if not data_dirs:
            print("⚠️ No data_* directories found, trying legacy structure...")
            # Fallback to legacy structure
            data_dirs = [self.data_root]
        
        for data_dir in data_dirs:
            print(f"\n📂 Processing directory: {data_dir}")
            
            # Find imdb files in this directory
            imdb_files = sorted(data_dir.glob("imdb_pretrain_p*.npy"))
            
            if not imdb_files:
                print(f"   ⚠️ No imdb files found in {data_dir}")
                continue
            
            pdf_root = data_dir / "pdfs"
            if not pdf_root.exists():
                print(f"   ⚠️ PDF directory not found: {pdf_root}")
                continue
            
            # Process each imdb file
            for imdb_file in imdb_files:
                print(f"   📄 Loading {imdb_file.name}...")
                samples = self._load_and_validate_imdb(imdb_file, pdf_root)
                all_samples.extend(samples)
                print(f"      ✓ Added {len(samples)} valid samples")
        
        return all_samples
    
    def _load_and_validate_imdb(self, imdb_file, pdf_root):
        """Load imdb file and validate samples"""
        data = np.load(imdb_file, allow_pickle=True)
        
        # Skip metadata at index 0 if present
        if len(data) > 0 and not isinstance(data[0], dict):
            data = data[1:]
        
        valid_samples = []
        skipped = 0
        skipped_too_long = 0
        
        for idx, sample in enumerate(data):
            if not isinstance(sample, dict):
                skipped += 1
                continue
            
            image_id = sample.get('image_id')
            if not image_id:
                skipped += 1
                continue
            
            # Check if PDF exists
            pdf_path = pdf_root / image_id / f"{image_id}.pdf"
            if not pdf_path.exists():
                skipped += 1
                continue
            
            # Check if OCR tokens exist
            ocr_tokens = sample.get('ocr_tokens')
            if not ocr_tokens or len(ocr_tokens) == 0:
                skipped += 1
                continue

            if len(ocr_tokens) >= 1400:
                text = " ".join(ocr_tokens)
                num_tokens = len(self.tokenizer.encode(text, add_special_tokens=True))
                
                if num_tokens > self.max_tokens_limit:
                    skipped_too_long += 1
                    continue
            
            # Store sample info with paths
            valid_samples.append({
                'sample': sample,
                'pdf_root': pdf_root,
                'image_id': image_id
            })
        
        if skipped > 0:
            print(f"      ⚠️ Skipped {skipped} invalid/missing samples")
        if skipped_too_long > 0:
            print(f"      ⚠️ Skipped {skipped_too_long} samples exceeding {self.max_tokens_limit} tokens")
        
        return valid_samples
    
    # def _filter_valid_samples(self):
    #     """Filter samples that have valid PDF and OCR data"""
    #     valid = []
    #     for idx, sample in enumerate(tqdm(self.data, desc="Filtering valid samples")):
    #         if not isinstance(sample, dict):
    #             continue

    #         image_id = sample.get('image_id')
    #         if not image_id:
    #             continue

    #         # Check if PDF exists
    #         pdf_path = self.pdf_root / image_id / f"{image_id}.pdf"
    #         if not pdf_path.exists():
    #             continue

    #         # Check if OCR tokens exist
    #         ocr_tokens = sample.get('ocr_tokens')
    #         if not ocr_tokens or len(ocr_tokens) == 0:
    #             continue

    #         valid.append(idx)

    #     return valid

    def __len__(self):
        # return len(self.valid_samples)
        return len(self.samples_info)

    def _load_page_image(self, pdf_root, image_id, page_num=0):
        """Load PDF page as image"""
        pdf_path = pdf_root / image_id / f"{image_id}.pdf"
        try:
            images = pdf2image.convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=150
            )

            if not images:
                return None
            
            # Convert sang PIL Image và đóng ngay
            image = images[0].copy()
            
            # Giải phóng danh sách images
            del images
            
            return image
            
        except Exception as e:
            print(f"⚠️ Error loading PDF {pdf_path}: {e}")
            return None

    def _mask_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for MLM pretraining
        """
        labels = input_ids.clone()

        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask(
                [val.item()],
                already_has_special_tokens=True
            )[0]
            for val in input_ids
        ], dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Get masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for non-masked tokens to -100
        labels[~masked_indices] = -100

        # 80% of the time: replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time: replace with random token
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time: keep original token

        return input_ids, labels

    def normalize_bbox(self, bbox, width, height):
        """Normalize bbox to [0, 1000]"""
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def __getitem__(self, idx):
        sample_info = self.samples_info[idx]
        sample = sample_info['sample']
        pdf_root = sample_info['pdf_root']
        image_id = sample_info['image_id']

        # Get data
        ocr_tokens = sample['ocr_tokens']
        ocr_boxes = sample['ocr_normalized_boxes']
        page_num = sample.get('ucsf_document_page', 0)

        # Load image
        image = self._load_page_image(pdf_root, image_id, page_num)
        if image is None:
            # Return dummy data if image fails to load
            return self._get_dummy_sample()

        # Resize image
        original_width, original_height = image.size
        resized_image = image.resize((512, 384))
        img_tensor = self.transform(resized_image)

        del image
        del resized_image

        # Create text input (no question, just OCR text for pretraining)
        text = " ".join(ocr_tokens)

        # Tokenize
        if self.use_chunked_processing:
            encoding = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors="pt"
            )
            actual_length = encoding['input_ids'].shape[1]
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            actual_length = self.max_length

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Apply MLM masking
        masked_input_ids, labels = self._mask_tokens(input_ids)

        # Process bounding boxes
        normalized_bboxes = []
        for bbox in ocr_boxes:
            # left = bbox[0]    # x-coordinate (normalized)
            # top = bbox[1]     # y-coordinate (normalized)
            # width = bbox[2]   # width (normalized)
            # height = bbox[3]  # height (normalized)

            # # Convert to pixel coordinates
            # x_min = int(left * original_width)
            # y_min = int(top * original_height)
            # x_max = int((left + width) * original_width)
            # y_max = int((top + height) * original_height)

            # norm_bbox = self.normalize_bbox(
            #     [x_min, y_min, x_max, y_max],
            #     original_width,
            #     original_height
            # )

            norm_bbox = [
                int(bbox[0] * 1000),  # left
                int(bbox[1] * 1000),  # top
                int(bbox[2] * 1000),  # right
                int(bbox[3] * 1000),  # bottom
            ]

            normalized_bboxes.append(norm_bbox)

        # Align bboxes with tokens
        word_encoding = self.tokenizer(ocr_tokens, is_split_into_words=True, add_special_tokens=False)
        word_ids = word_encoding.word_ids()

        bbox_array = []
        for i in range(actual_length):
            if i < len(word_ids) and word_ids[i] is not None:
                word_idx = word_ids[i]
                if word_idx < len(normalized_bboxes):
                    bbox_array.append(normalized_bboxes[word_idx])
                else:
                    bbox_array.append([0, 0, 0, 0])
            else:
                bbox_array.append([0, 0, 0, 0])

        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'bboxes': torch.tensor(bbox_array, dtype=torch.long),
            'pixel_values': img_tensor,
            'original_length': actual_length,
            'doc_id': image_id,
        }

    def _get_dummy_sample(self):
        """Return dummy sample in case of errors"""
        dummy_length = 128
        return {
            'input_ids': torch.zeros(dummy_length, dtype=torch.long),
            'attention_mask': torch.zeros(dummy_length, dtype=torch.long),
            'labels': torch.full((dummy_length,), -100, dtype=torch.long),
            'bboxes': torch.zeros(dummy_length, 4, dtype=torch.long),
            'pixel_values': torch.zeros(3, 384, 512),
            'original_length': dummy_length,
            'doc_id': 'dummy',
        }

