import torch
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-xls-r-1b-russian", device=device)
output_dir = "my/CPU_model"

# first of all, you need to define your model's token set
# tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
tokens = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ы', 'э', 'ю', 'я']

token_set = TokenSet(tokens)

learning_rate=3e-4
max_steps=0
eval_steps=None
per_device_train_batch_size=8
per_device_eval_batch_size=8

# the lines below will load the training and model arguments objects, 
# you can check the source code (huggingsound.trainer.TrainingArguments and huggingsound.trainer.ModelArguments) to see all the available arguments
training_args = TrainingArguments(
    learning_rate=learning_rate,
    max_steps=max_steps,
    eval_steps=eval_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
)
model_args = ModelArguments(
    activation_dropout=0.1,
    hidden_dropout=0.1,
) 

# define your train/eval data
train_data = [
    {"path": "/content/drive/MyDrive/sbx_train_data/e8ef0e7b8ccf0c2148f92b57cb025811.mp3", "transcription": "где мой заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-15-5-50.mp3", "transcription": "Алматы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-20-27.mp3", "transcription": "Нужен заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-24-3.mp3", "transcription": "гранд на гоголя"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-25-46.mp3", "transcription": "меломан гранд"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-26-16.mp3", "transcription": "гранд на панфилова"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-36-35.mp3", "transcription": "гранд"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-37-42.mp3", "transcription": "гоголя 58"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-38-24.mp3", "transcription": "меломан на гоголя 58"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-12-16-39-17.mp3", "transcription": "гоголя панфилова"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-27-27.mp3", "transcription": "меломан на панфилова"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-30-58.mp3", "transcription": "мега центр"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-31-7.mp3", "transcription": "мега марвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-30-43.mp3", "transcription": "мега алмата"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-31-18.mp3", "transcription": "мега меломан"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-34-26.mp3", "transcription": "большая мега марвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-35-42.mp3", "transcription": "мега на розыбакиева"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-36-1.mp3", "transcription": "марвин достык"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-12-37-36.mp3", "transcription": "достык плаза"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-13-16.mp3", "transcription": "меломан достык плаза"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-14-50.mp3", "transcription": "самал два дом сто одиннадцать достык плаза"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-15-55.mp3", "transcription": "достык жолдасбекова"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-53-14.mp3", "transcription": "мега парк"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-53-34.mp3", "transcription": "мега парк на сейфуллина"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-53-56.mp3", "transcription": "мега на сейфуллина"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-54-16.mp3", "transcription": "мега на сейфуллина макатаева"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-54-31.mp3", "transcription": "мега парк марвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-54-45.mp3", "transcription": "мега парк меломан"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-54-58.mp3", "transcription": "марвин москва"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-55-38.mp3", "transcription": "меломан в микрорайоне 8 дом 37 дробь 1"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-55-51.mp3", "transcription": "меломан есентай"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-56-4.mp3", "transcription": "меломан есентай"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-56-16.mp3", "transcription": "меломан есентай молл"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-57-23.mp3", "transcription": "марвин силквей алматы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-15-57-36.mp3", "transcription": "марвин силквей алматы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-2-31.mp3", "transcription": "аутлеет марвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-8-47.mp3", "transcription": "где мой заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-10-59.mp3", "transcription": "мой заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-11-10.mp3", "transcription": "найти товар"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-11-28.mp3", "transcription": "оформи мне заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-11-43.mp3", "transcription": "найди товар"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-11-55.mp3", "transcription": "узнать адрес"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-12-11.mp3", "transcription": "хочу узнать адрес"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-27-12.mp3", "transcription": "оператор"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-27-57.mp3", "transcription": "привет"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-28-14.mp3", "transcription": "пока"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-29-32.mp3", "transcription": "добрый день"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-17-16-29-58.mp3", "transcription": "до свидания"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-14-58-0.mp3", "transcription": "пропустил"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-14-58-18.mp3", "transcription": "звонок"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-14-58-37.mp3", "transcription": "пропущенный"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-14-59-13.mp3", "transcription": "график"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-5-31.mp3", "transcription": "график работы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-2-44.mp3", "transcription": "время работы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-2-2.mp3", "transcription": "соединить с мэнеджером"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-1-15.mp3", "transcription": "соединить с менеджером"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-1-4.mp3", "transcription": "расписание"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-0-15.mp3", "transcription": "расписание работы"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-12-26.mp3", "transcription": "комфорт"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-13-3.mp3", "transcription": "усть-каменогорск"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-13-26.mp3", "transcription": "усть каман"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-13-44.mp3", "transcription": "Алтай"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-14-10.mp3", "transcription": "Зыряновск"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-14-33.mp3", "transcription": "Меломан"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-14-47.mp3", "transcription": "Марвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-45-59.mp3", "transcription": "АДК"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-46-58.mp3", "transcription": "Сатпаева девяносто"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-47-14.mp3", "transcription": "Сатпаева"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-48-16.mp3", "transcription": "Толеби"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-48-34.mp3", "transcription": "Толеби аутлэт"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-49-11.mp3", "transcription": "Толеби аутлет"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-49-39.mp3", "transcription": "Заказ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-49-54.mp3", "transcription": "Заказе"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-58-14.mp3", "transcription": "республика акжол"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-58-39.mp3", "transcription": "республика"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-59-4.mp3", "transcription": "коргальжинское шоссе"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-59-20.mp3", "transcription": "ханшатыр"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-0-49.mp3", "transcription": "шатёр"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-1-6.mp3", "transcription": "сарыарка"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-1-34.mp3", "transcription": "гринмолл"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-2-17.mp3", "transcription": "ситимолл"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-2-54.mp3", "transcription": "керуенсити"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-3-13.mp3", "transcription": "актау"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-4-0.mp3", "transcription": "трцактау"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-4-15.mp3", "transcription": "тцактау"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-4-38.mp3", "transcription": "семей"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-4-51.mp3", "transcription": "семеймарвин"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-6-33.mp3", "transcription": "уральск"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-6-53.mp3", "transcription": "уральскситицентр"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-7-18.mp3", "transcription": "мега"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-7-53.mp3", "transcription": "мегапланет"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-8-12.mp3", "transcription": "атыраусарыарка"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-8-31.mp3", "transcription": "караганда"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-9-4.mp3", "transcription": "актобе"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-9-21.mp3", "transcription": "костанай"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-9-46.mp3", "transcription": "павлодар"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-10-14.mp3", "transcription": "петропавловск"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-10-32.mp3", "transcription": "тараз"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-10-48.mp3", "transcription": "ука"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-11-39.mp3", "transcription": "укаАДК"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-16-12-7.mp3", "transcription": "ЦУМ"},
    {"path": "/content/drive/MyDrive/sbx_train_data/ttsmaker-file-2024-7-19-15-57-52.mp3", "transcription": "кошкарбаева"},

]

# and finally, fine-tune your model
model.finetune(
    output_dir, 
    train_data=train_data, 
    # eval_data=eval_data, # the eval_data is optional
    token_set=token_set, 
    training_args=training_args,
    model_args=model_args,
)
