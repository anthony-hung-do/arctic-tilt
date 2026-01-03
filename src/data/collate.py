import torch

class DocVQACollateFn(object):
    def __init__(self, tokenizer, max_source_length=512, max_target_length=128, use_chunked_processing=True):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_chunked_processing = use_chunked_processing

    def __call__(self, batch):
        if self.use_chunked_processing:
            # Variable-length sequences → cần pad trong collate_fn
            return self._collate_variable_length(batch)
        else:
            # Fixed-length sequences → chỉ cần stack
            return self._collate_fixed_length(batch)

    def _collate_variable_length(self, batch):
        """Xử lý variable-length sequences cho chunked processing"""

        # Tìm max length trong batch
        max_length = max(item['original_length'] for item in batch)

        batch_dict = {}

        # Pad sequences đến max_length trong batch
        input_ids_list = []
        attention_mask_list = []
        bboxes_list = []

        for item in batch:
            current_len = item['original_length']
            pad_len = max_length - current_len

            if pad_len > 0:
                # Pad input_ids
                padded_input_ids = torch.cat([
                    item['input_ids'],
                    torch.full((pad_len,), self.tokenizer.pad_token_id)
                ])

                # Pad attention_mask
                padded_attention_mask = torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len)
                ])

                # Pad bboxes
                padding_bbox = torch.zeros(pad_len, 4, dtype=torch.long)
                padded_bboxes = torch.cat([item['bboxes'], padding_bbox])
            else:
                padded_input_ids = item['input_ids']
                padded_attention_mask = item['attention_mask']
                padded_bboxes = item['bboxes']

            input_ids_list.append(padded_input_ids)
            attention_mask_list.append(padded_attention_mask)
            bboxes_list.append(padded_bboxes)

        # Stack tensors
        batch_dict['input_ids'] = torch.stack(input_ids_list)
        batch_dict['attention_mask'] = torch.stack(attention_mask_list)
        batch_dict['bboxes'] = torch.stack(bboxes_list)
        batch_dict['labels'] = torch.stack([item['labels'] for item in batch])
        batch_dict['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])

        # Metadata
        batch_dict['questions'] = [item['question'] for item in batch]
        batch_dict['answers'] = [item['answer'] for item in batch]
        batch_dict['doc_ids'] = [item['doc_id'] for item in batch]

        return batch_dict

    def _collate_fixed_length(self, batch):
        """Xử lý fixed-length sequences (original logic)"""
        batch_dict = {}

        # Stack các tensors
        batch_dict['input_ids'] = torch.stack([item['input_ids'] for item in batch])
        batch_dict['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
        batch_dict['labels'] = torch.stack([item['labels'] for item in batch])
        batch_dict['bboxes'] = torch.stack([item['bboxes'] for item in batch])
        batch_dict['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])

        # Metadata (không cần tensor)
        batch_dict['questions'] = [item['question'] for item in batch]
        batch_dict['answers'] = [item['answer'] for item in batch]
        batch_dict['doc_ids'] = [item['doc_id'] for item in batch]

        return batch_dict
    

class PretrainCollateFn:
    """Collate function for pretrain dataloader"""
    def __init__(self, tokenizer, use_chunked_processing=True):
        self.tokenizer = tokenizer
        self.use_chunked_processing = use_chunked_processing

    def __call__(self, batch):
        if self.use_chunked_processing:
            return self._collate_variable_length(batch)
        else:
            return self._collate_fixed_length(batch)

    def _collate_variable_length(self, batch):
        """Handle variable length sequences"""
        max_length = max(item['original_length'] for item in batch)

        batch_dict = {}
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        bboxes_list = []

        for item in batch:
            current_len = item['original_length']
            pad_len = max_length - current_len

            if pad_len > 0:
                # Pad input_ids
                padded_input_ids = torch.cat([
                    item['input_ids'],
                    torch.full((pad_len,), self.tokenizer.pad_token_id)
                ])

                # Pad attention_mask
                padded_attention_mask = torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len)
                ])

                # Pad labels
                padded_labels = torch.cat([
                    item['labels'],
                    torch.full((pad_len,), -100)
                ])

                # Pad bboxes
                padding_bbox = torch.zeros(pad_len, 4, dtype=torch.long)
                padded_bboxes = torch.cat([item['bboxes'], padding_bbox])
            else:
                padded_input_ids = item['input_ids']
                padded_attention_mask = item['attention_mask']
                padded_labels = item['labels']
                padded_bboxes = item['bboxes']

            input_ids_list.append(padded_input_ids)
            attention_mask_list.append(padded_attention_mask)
            labels_list.append(padded_labels)
            bboxes_list.append(padded_bboxes)

        batch_dict['input_ids'] = torch.stack(input_ids_list)
        batch_dict['attention_mask'] = torch.stack(attention_mask_list)
        batch_dict['labels'] = torch.stack(labels_list)
        batch_dict['bboxes'] = torch.stack(bboxes_list)
        batch_dict['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
        batch_dict['doc_ids'] = [item['doc_id'] for item in batch]

        return batch_dict

    def _collate_fixed_length(self, batch):
        """Handle fixed length sequences"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'bboxes': torch.stack([item['bboxes'] for item in batch]),
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'doc_ids': [item['doc_id'] for item in batch]
        }