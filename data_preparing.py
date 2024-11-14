from transformers import AutoTokenizer
from datasets import load_from_disk
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader


def load_tokenizer(model_name='mistralai/Mistral-7B-v0.1'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token.")
        else:
            raise ValueError("Tokenizer does not have an eos_token to use as pad_token.")
    else:
        print("Tokenizer already has a pad_token.")
    return tokenizer




def prepare_data(dataset, tokenizer, max_length=256):
    # Tokenize the dataset without truncation
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    
    # Tokenize each split
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    

    # Function to group texts into chunks
    def group_texts(examples):
        concatenated_examples = {'input_ids': sum(examples['input_ids'], [])}
        total_length = len(concatenated_examples['input_ids'])
        block_size = max_length
        total_length = (total_length // block_size) * block_size
        concatenated_examples['input_ids'] = concatenated_examples['input_ids'][:total_length]
        result = {
            'input_ids': [concatenated_examples['input_ids'][i: i + block_size]
                          for i in range(0, total_length, block_size)]
        }
        return result

    # Group texts in each split
    final_datasets = {}
    for split in tokenized_datasets.keys():
        # Group texts
        grouped_dataset = tokenized_datasets[split].map(
            group_texts,
            batched=True,
            remove_columns=tokenized_datasets[split].column_names,
            desc=f"Grouping texts in {split} split",
        )

        # Flatten and set format
        grouped_dataset = grouped_dataset.flatten()
        grouped_dataset.set_format(type='torch', columns=['input_ids'])
        final_datasets[split] = grouped_dataset.shuffle()

    return final_datasets




def save_dataset(path_to_save, return_dataloader=False, save_dataset=True):
    dataset = load_dataset("ashaba1in/small_openwebtext")
    tokenizer = load_tokenizer()
    final_dataset = prepare_data(dataset, tokenizer, max_length=256)
    train_dataset = final_dataset['train']
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        )
    if save_dataset:
        final_dataset_dict = DatasetDict(final_dataset)
        final_dataset_dict.save_to_disk(path_to_save)

    if return_dataloader:
        return train_dataloader
    
    return None




def load_data(path_to_save):
    final_dataset = load_from_disk(path_to_save)
    train_dataset = final_dataset['train']
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )
    return train_dataloader



