import torch
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoConfig, AutoModel

def bert(text):

    label2id = {'FALSE':0,'MISLEADING':1,'TRUE':2}
    id2label = {v:k for k, v in label2id.items()}

    bert_sequential_config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    MAX_LEN = 64

    model_save_name = 'classifier_multilab.pt'
    # path = F"/content/drive/My Drive/Files/{model_save_name}" 
    path = "MicrosoftEssexBert/Files/" + model_save_name
    model = torch.load(path)




    ### Define the model 
    bert_sequential_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path="bert-base-uncased",
                config=bert_sequential_config,
            )

    #### Load the model 
    bert_sequential_model.load_state_dict(torch.load(path))

    device = torch.device("cuda:0")
    bert_sequential_model = bert_sequential_model.to(device)

    def convert_news_to_features(news):
        input_ids = [
            bert_tokenizer.encode(news, add_special_tokens=True)
        ]

        input_ids = pad_sequences(
            input_ids,
            maxlen=MAX_LEN,
            dtype="long", 
            value=bert_tokenizer.pad_token_id,
            padding="post",
            truncating="post"
        )

        input_ids = torch.tensor(input_ids)
        attention_masks = (input_ids > 0).int()

        return TensorDataset(input_ids, attention_masks)

    example = convert_news_to_features(text)


    dl = DataLoader(
        example,
        batch_size=32,
        num_workers=0)

    bert_sequential_model = bert_sequential_model.eval()
    
    spans_list = []
    labels_list = []
    probabilities_list = []

    with torch.no_grad():
        for batch in dl:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = bert_sequential_model(
                    input_ids, 
                    attention_mask=attention_mask 
                )
            _, predictions = torch.max(outputs[0], dim=1)
            probabilities = torch.sigmoid(outputs[0])

    # print(predictions)
    # print(outputs)
    return predictions[0].item()

# texto = "teste"
# result = bert(texto)
# print(result)