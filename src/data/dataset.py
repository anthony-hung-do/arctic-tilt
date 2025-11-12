from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import numpy as np

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