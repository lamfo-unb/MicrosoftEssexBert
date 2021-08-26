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
    path = "../MicrosoftEssexBert/Files/" + model_save_name
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
    # print(probabilities)
    return predictions[0].item()

# texto = "teste"
# result = bert(texto)
# print(result)



def train_bert():
    import torch
    import pandas as pd

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import AutoConfig, AutoModel
    from keras.preprocessing.sequence import pad_sequences
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    label2id = {'FALSE':0,'MISLEADING':1,'TRUE':2}
    id2label = {v:k for k, v in label2id.items()}

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    bert_sequential_config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    df_final = pd.read_csv('covid_news_final.csv')

    MAX_LEN = 64


    def convert_examples_to_features(tweets, labels):
        input_ids = [
            bert_tokenizer.encode(tweet, add_special_tokens=True) for tweet in tweets
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
        labels = torch.tensor([label2id[label] for label in labels])

        return TensorDataset(input_ids, attention_masks, labels)
    
    dataset = convert_examples_to_features(df_final.Title, list(df_final.Label))

    from sklearn.model_selection import train_test_split

    train_data, val_data, train_labels, val_labels = train_test_split(
        dataset,
        list(df_final.Label), 
        random_state=1234,
        test_size=0.2
    )

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

    from torch.utils.data import (
    DataLoader,
    TensorDataset,
    RandomSampler,
    SequentialSampler,
)

    BATCH_SZ = 64

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        dataset=train_data,
        sampler=train_sampler,
        batch_size=BATCH_SZ
    )

    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(
        dataset=val_data,
        sampler=val_sampler,
        batch_size=BATCH_SZ
    )

    model_save_name = 'classifier_multilab.pt'
    # path = F"/content/drive/My Drive/Files/{model_save_name}" 
    path = "../MicrosoftEssexBert/Files/" + model_save_name
    model = torch.load(path)

    ### Define the model 
    bert_sequential_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path="bert-base-uncased",
                config=bert_sequential_config,
            )

    #### Load the model 
    bert_sequential_model.load_state_dict(torch.load(path))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Moving model to device: {device}")
    bert_sequential_model = bert_sequential_model.to(device)

    from torch.optim import SGD
    from torch.optim import Adadelta
    from tqdm import tqdm

    # define a learning rate
    LR=5e-4
    optimizer = SGD(bert_sequential_model.parameters(), lr=LR)
    optimizer_ad = Adadelta(bert_sequential_model.parameters(), lr=LR)

    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    EPOCHS = 2
    loss = []

    for epoch in range(EPOCHS):
        batch_loss = 0
        # The model is in training model now; while in evaluation mode,
        # we change this to .eval()
        bert_sequential_model.train()

        for batch in tqdm(train_dataloader):
            # move the input data to device
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            # pass the input to the model
            outputs = bert_sequential_model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # set model gradients to 0, so that optmizer won't accumulate
            # them over subsequent training iterations
            optimizer.zero_grad()
            loss = outputs[0]

            # obtain loss, and backprop
            batch_loss += loss.item()
            loss.backward()
            #clip gradient norms to avoid any exploding gradient problems
            # torch.nn.utils.clip_grad_norm_(bert_sequential_model.parameters(), 1.0)
            optimizer.step()

        epoch_train_loss = batch_loss / len(train_dataloader)  
        print(f"epoch: {epoch+1}, train_loss: {epoch_train_loss}")
        
        # At the end of each epoch, we will also run the model 
        # on the validation dataset
        val_loss, val_accuracy = 0, 0
        true_labels, predictions = [], []

        for val_batch in val_dataloader:
            val_batch = tuple(t.to(device) for t in val_batch)
            input_ids, attention_mask, labels = val_batch
            
            with torch.no_grad():        
                outputs = bert_sequential_model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
                )
            
            val_loss += loss.item()
            
            # convert predictions and gold labels to numpy arrays so that
            # we can compute evaluation metrics like accuracy and f1
            label_ids = labels.to('cpu').numpy()
            preds = outputs[1].detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            true_labels.extend(label_ids)
            predictions.extend(preds)
        
        acc = f1_score(y_true=true_labels, y_pred=predictions, average='micro')
        f1 = f1_score(y_true=true_labels, y_pred=predictions, average='macro')

        print(f"epoch: {epoch+1} val loss: {val_loss}, accuracy:{acc}, f1:{f1}")

    model_save_name = 'classifier_multilab.pt'
    # path = F"/content/drive/My Drive/Files/{model_save_name}" 
    path = "../MicrosoftEssexBert/Files/" + model_save_name
    torch.save(bert_sequential_model.state_dict(), path)



    