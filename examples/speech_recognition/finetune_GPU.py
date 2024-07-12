import torch
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-xls-r-1b-russian", device=device)
output_dir = "my/GPU_model"

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
    {"path": "./files/group_question1.mp3", "transcription": "остались вопросы, ответе да связат с оператором"},
    {"path": "./files/5addressKomfort.mp3", "transcription": "магазин комфорт расположен по адресу проспект суюнбая дом два график работы с восми часов до двадцати двух часов вечера"},
    {"path": "./files/4scheduleKomfort.mp3", "transcription": "комфорт работает с восми утра до семи вечера"},
    {"path": "./files/4addresMag.mp3", "transcription": "адрес какого магазига вам требуется"},
    {"path": "./files/4.3supportItem.mp3", "transcription": "вас слушает служба поддержки марвин проблемы с доставкой оплатой сайтом"},
    {"path": "./files/4.2scheduleMeloman.mp3", "transcription": "меломан работает с восьми утра до семи вечера"},
    {"path": "./files/4.1scheduleMarwin.mp3", "transcription": "марвин работает с восми утра до семи вечера"},
    {"path": "./files/4.1raspysanieMag.mp3", "transcription": "график какого магазина вам нужен"},
    {"path": "./files/3transfer_operator.mp3", "transcription": "соеденяю с оперетором"},
    {"path": "./files/3.1assistant.mp3", "transcription": "что вас интересует время работы магазинов наличие товара адрес магазина"},
    {"path": "./files/2menu_ru.mp3", "transcription": "пропустили наш звонок нажмите один перейти к виртуальному боту нажмите два"},
    {"path": "./files/1hello.mp3", "transcription": "вас привестствует телебот марвин для выбора русского языка нажмите один для выбора казахского языка нажмите два"} ,
    {"path": "./files/0repeatRu.mp3", "transcription": "повторите"},
    {"path": "./files/00goodbyeRu.mp3", "transcription": "желаем вам доброго дня"},
]
eval_data = [
    {"path": "/path/to/sagan2.mp3", "transcription": "absence of evidence is not evidence of absence"},
    {"path": "/path/to/asimov2.wav", "transcription": "the true delight is in the finding out rather than in the knowing"},
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
